import datetime as dt
import multiprocessing as mp
import signal
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Event, Queue
from pathlib import Path
from queue import Empty

import polars as pl
from loguru import logger

from metrion.config_reader.config_schema import LiveAttributionConfig
from metrion.data_record import DataRecord
from metrion.data_storage.data_storage_factory import DataStorageFactory
from metrion.data_storage.parquet_data_storage import _RotatingParquetWriter
from metrion.energy_attribution.energy_attribution import DyncamicEnergyAttributionModel
from metrion.energy_attribution.wide_result import add_up, denormalize, identity_of, to_wide
from metrion.entity.entity import Entity
from metrion.entity.logical_entity import LogicalEntity
from metrion.measurement import Measurement
from metrion.queue_util import close_and_join_feeder, drain_until_sentinel
from metrion.util import (
    IpcAsyncWriter,
    read_temp_ipc,
    transform_measuremnent_list_to_polars,
)


@dataclass
class ForgottenEntities:
    """Ids of threads the registry has let go of, sent to the saver on the entities queue."""

    ids: list[str]


class DataSaver:
    """Stores data records in the specified storage backend."""

    def __init__(
        self,
        entities_queue: Queue,
        measurements_queue: Queue,
        stop_event: Event,  # type: ignore
        storage_type: str,
        storage_path: str,
        live_config: LiveAttributionConfig,
        hardware: Sequence[Entity],
        store_raw: bool = True,
    ):
        self._entities_queue = entities_queue
        self._measurements_queue = measurements_queue
        self._stop_event = stop_event
        self._live = live_config
        self.dyn_eam = DyncamicEnergyAttributionModel(cutoff=live_config.cutoff)
        # The hardware is detected once at start and never changes, so the model gets it here, once.
        self.dyn_eam.update_entites(hardware)
        self._storage_type = storage_type
        self._storage_path = storage_path
        self._store_raw = store_raw

        self._received_sent_e = False
        self._received_sent_m = False

        # Live attribution. The writer is opened in store_data, inside the saver process.
        self._energy_writer: _RotatingParquetWriter | None = None
        # Windows some component has not attributed yet; the next cycles add to them.
        self._open_windows = to_wide([], live_config.window)
        # Identity per logical_entity_id for the result stream, the latest version of each thread.
        self._identities: dict[str, dict] = {}
        # (drop after, ids) of the threads the registry let go of, oldest first.
        self._forgotten: deque[tuple[dt.datetime, list[str]]] = deque()

    def _collect_attribute_energies(
        self, measurements: Sequence[Measurement] | pl.DataFrame, force: bool = False
    ) -> list[pl.DataFrame]:
        """Feed the attribution model and return its non-empty result frames.

        attribute_energy returns three triples (CPU, DRAM, GPU), each of them
        (total, idle, dynamic), so up to nine frames in the measurement schema.
        """
        self.dyn_eam.update_measurements(measurements)

        cpu_result, dram_result, gpu_result = self.dyn_eam.attribute_energy(force=force)

        return [df for df in (*cpu_result, *dram_result, *gpu_result) if not df.is_empty()]

    def _run_live_attribution(self, measurement_dfs: list[pl.DataFrame], force: bool = False) -> None:
        """Attribute what arrived and write the windows that every component has passed.

        A window is written once, when complete; until then the rows of later cycles are added
        to it. force writes all open windows, for the shutdown.
        """
        if not self._live.enabled:
            return
        # The TCP monitor's traffic frames share the queue; they feed no component and carry two more columns.
        measurement_dfs = [df for df in measurement_dfs if "src_endpoint" not in df.columns]

        # New measurements arrive once per timing.cycle, the loop runs every 0.5 s; a call without
        # them would only repeat the same filtering.
        if measurement_dfs or force:
            measurements = pl.concat(measurement_dfs, how="vertical_relaxed") if measurement_dfs else pl.DataFrame()
            frames = self._collect_attribute_energies(measurements, force=force)
            wide = add_up([self._open_windows, to_wide(frames, self._live.window)])

            until = dt.datetime.max if force else self.dyn_eam.attributed_until()
            if until is None:
                self._open_windows = wide
            else:
                if self._energy_writer is not None:
                    self._energy_writer.write(denormalize(wide.filter(pl.col("window_end") <= until), self._identities))
                self._open_windows = wide.filter(pl.col("window_end") > until)
                self._drop_forgotten(until)

        if self._energy_writer is not None:
            self._energy_writer.flush_aged()

    def _remember_identities(self, entities: Sequence[Entity]) -> None:
        if not self._live.enabled:
            return
        for entity in entities:
            if isinstance(entity, LogicalEntity):
                # Later versions win: the registry resends a thread once its name or cmdline changes.
                self._identities[entity.id] = identity_of(entity)

    def _forget(self, ids: list[str]) -> None:
        if not self._live.enabled:
            return
        # A thread's energy ends before the registry let go of it, so its last window ends at most
        # one window later.
        drop_after = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(seconds=self._live.window)
        self._forgotten.append((drop_after, ids))

    def _drop_forgotten(self, until: dt.datetime) -> None:
        """Drop the identities of forgotten threads whose windows have all been written."""
        while self._forgotten and self._forgotten[0][0] <= until:
            _, ids = self._forgotten.popleft()
            for logical_entity_id in ids:
                self._identities.pop(logical_entity_id, None)

    def store_data(self):
        ds_factory = DataStorageFactory()
        self._data_storage = ds_factory.create_data_storage(self._storage_type, self._storage_path)
        if self._live.enabled:
            # The result stream has few rows; without the age flush its first file would take hours.
            self._energy_writer = _RotatingParquetWriter(
                Path(self._live.path), rotate_rows=200_000, max_age_s=self._live.window
            )

        while not self._stop_event.is_set():
            try:
                self._process_entities()
                measurement_dfs = self._process_measurements()
                self._run_live_attribution(measurement_dfs)

                # sleep to avoid busy waiting
                time.sleep(0.5)

            except Exception as e:
                logger.opt(exception=True).warning(f"Error in data saving loop: {e}")
                time.sleep(1)

        self._stop()

    def _process_entities(self) -> None:
        items = []
        while True:
            try:
                item = self._entities_queue.get_nowait()
            except Empty:
                break
            if item == DataCollector.SENT_E:
                self._received_sent_e = True
                continue
            items.append(item)

        self._take_entities(items)

    def _take_entities(self, items: Sequence) -> None:
        """Store the entities among the queue items and note the threads the registry let go of."""
        entities: list[Entity] = []
        for item in items:
            if isinstance(item, ForgottenEntities):
                self._forget(item.ids)
            else:
                entities.extend(item)

        if entities:
            self._data_storage.save_entities(entities)
            self._remember_identities(entities)

    def _process_measurements(self) -> list[pl.DataFrame]:
        dfs = []
        while True:
            try:
                path = self._measurements_queue.get_nowait()
                if path == DataCollector.SENT_M:
                    self._received_sent_m = True
                    continue
                df = read_temp_ipc(path)
                dfs.append(df)
            except Empty:
                break
            except Exception as e:
                logger.error(f"Error processing measurement path: {e}")
                break

        if dfs and self._store_raw:
            self._data_storage.save_measurements(dfs)
        return dfs

    def _stop(self, deadline_s: float = 20.0):
        # Final drain of queues using sentinels
        if not self._received_sent_e:
            items, seen = drain_until_sentinel(
                self._entities_queue, DataCollector.SENT_E, deadline_s=deadline_s, name="entities"
            )
            self._take_entities(items)
            self._received_sent_e = seen
            if not seen:
                logger.warning("Entity sentinel not received, some entities may be missing")

        measurement_dfs: list[pl.DataFrame] = []
        if not self._received_sent_m:
            items, seen = drain_until_sentinel(
                self._measurements_queue, DataCollector.SENT_M, deadline_s=deadline_s, name="measurements"
            )
            for path in items:
                df = read_temp_ipc(path)
                if self._store_raw:
                    self._data_storage.save_measurements([df])
                measurement_dfs.append(df)
            self._received_sent_m = seen
            if not seen:
                logger.warning("Measurement sentinel not received, some measurements may be missing")

        # What has not passed the cutoff yet would otherwise be lost with the process.
        try:
            self._run_live_attribution(measurement_dfs, force=True)
        except Exception as e:
            logger.opt(exception=True).warning(f"Final live attribution failed: {e}")
        if self._energy_writer is not None:
            self._energy_writer.flush()

        # The queues belong to the producer; closing them here would only block this side.
        self._data_storage.close()


def start_data_saver(data_saver: DataSaver):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    data_saver.store_data()


class DataCollector:
    """Collects data records and manages their storage using configurable strategies."""

    SENT_E = ("__EOF__", "entities")
    SENT_M = ("__EOF__", "measurements")

    def __init__(
        self,
        storage_type: str,
        storage_path: str,
        flush_interval: float = 2.0,
        live_config: LiveAttributionConfig | None = None,
        hardware: Sequence[Entity] = (),
        store_raw: bool = True,
    ):
        """
        Initialize the DataCollector.

        Args:
            storage_type: Backend to write to, 'duckdb' or 'parquet'
            storage_path: Target the backend is opened with
            flush_interval: Seconds the IPC writer holds rows before writing a file
            live_config: Attribution while the agent runs; off when not given
            hardware: Detected CPUs, DRAM and GPUs, whose energy the attribution splits
            store_raw: Write the measurements themselves; off keeps only the result stream
        """
        self._ctx = mp.get_context("spawn")

        self._entities_queue = self._ctx.Queue()
        self._measurements_queue = self._ctx.Queue()
        self._storage_type = storage_type
        self._storage_path = storage_path
        self._live_config = live_config or LiveAttributionConfig()
        self._hardware = hardware
        self._store_raw = store_raw

        # Buffering for measurements
        self._measurement_buffer = []
        self._buffer_row_count = 0
        self._max_buffer_rows = 1_000_000

        self._stop_event = self._ctx.Event()

        self.process = None

        # IPC writer
        self.ipc_writer = IpcAsyncWriter(
            out_paths_queue=self._measurements_queue,
            assign_ids=True,
            max_age_s=flush_interval,
        )

        self._start_collection_process()

    def _start_collection_process(self):
        """Start the collection process for both strategies."""
        self.data_saver = DataSaver(
            self._entities_queue,
            self._measurements_queue,
            self._stop_event,
            self._storage_type,
            self._storage_path,
            self._live_config,
            self._hardware,
            self._store_raw,
        )

        self.process = self._ctx.Process(
            target=start_data_saver,
            name="data_saver_process",
            args=(self.data_saver,),
            daemon=True,
        )
        self.process.start()

    def add(self, value: DataRecord):
        """
        Add a data record to the collection queues.

        Args:
            value: DataRecord containing entities and measurements to add
        """
        if value._entities:
            self._entities_queue.put_nowait(value._entities)

        if value._measurements:
            measurements = value._measurements
            self._measurement_buffer.extend(measurements)
            self._flush_measurements()

    def forget(self, ids: list[str]) -> None:
        """Tell the saver which threads the registry has let go of."""
        if ids:
            self._entities_queue.put_nowait(ForgottenEntities(ids))

    def _flush_measurements(self):
        """Transform buffered measurements to Polars DataFrame and write to IPC."""
        if not self._measurement_buffer:
            return

        try:
            list_df_m, list_df_comm = transform_measuremnent_list_to_polars(self._measurement_buffer)
            if len(list_df_m) > 0:
                self.ipc_writer.enqueue(list_df_m, "measurement")
            if len(list_df_comm) > 0:
                self.ipc_writer.enqueue(list_df_comm, "communication_measurement")
        except Exception as e:
            logger.error(f"Error flushing measurements: {e}")
        finally:
            # Clear buffer regardless of success to avoid stuck memory
            self._measurement_buffer = []
            self._buffer_row_count = 0

    def stop(self, join_timeout: float = 60.0):
        """Stop the data collection process gracefully."""
        # Flush any remaining measurements
        self._flush_measurements()
        self.ipc_writer.close()

        self._entities_queue.put(DataCollector.SENT_E)
        self._measurements_queue.put(DataCollector.SENT_M)

        # Let the feeder threads push the sentinels through before the saver is told to
        # wind down. cancel_join_thread() here would be allowed to drop them.
        close_and_join_feeder(self._entities_queue, name="entities")
        close_and_join_feeder(self._measurements_queue, name="measurements")

        self._stop_event.set()
        logger.debug("Saving final data...")

        if self.process:
            self.process.join(timeout=join_timeout)
            if self.process.is_alive():
                logger.warning(f"Saver process still running after {join_timeout}s, terminating...")
                self.process.terminate()
                self.process.join(timeout=5)
            if self.process.is_alive():
                logger.warning("Saver process ignored SIGTERM, killing it")
                self.process.kill()
                self.process.join(timeout=5)
            self.process = None

        logger.info("DataCollector stopped.")
