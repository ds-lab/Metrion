import multiprocessing as mp
import signal
import time
from collections.abc import Sequence
from multiprocessing import Event, Queue
from queue import Empty

import polars as pl
from loguru import logger

from metrion.data_record import DataRecord
from metrion.data_storage.data_storage_factory import DataStorageFactory
from metrion.energy_attribution.energy_attribution import DyncamicEnergyAttributionModel
from metrion.entity.entity import Entity
from metrion.measurement import Measurement
from metrion.util import (
    IpcAsyncWriter,
    read_temp_ipc,
    transform_measuremnent_list_to_polars,
)


class DataSaver:
    """Stores data records in the specified storage backend."""

    def __init__(
        self,
        entities_queue: Queue,
        measurements_queue: Queue,
        stop_event: Event,  # type: ignore
        storage_type: str,
        storage_path: str,
    ):
        self._entities_queue = entities_queue
        self._measurements_queue = measurements_queue
        self._stop_event = stop_event
        self.dyn_eam = DyncamicEnergyAttributionModel(cutoff=180.0)
        self._storage_type = storage_type
        self._storage_path = storage_path

        self._received_sent_e = False
        self._received_sent_m = False

    def _collect_attribute_energies(
        self, entities: Sequence[Entity], measurements: Sequence[Measurement] | pl.DataFrame, force: bool = False
    ):
        self.dyn_eam.update_entites(entities)
        self.dyn_eam.update_measurements(measurements)

        (
            (cpu_pid_total_energy, cpu_pid_idle_energy, cpu_pid_dynamic_energy),
            (dram_pid_total_energy, dram_pid_idle_energy, dram_pid_dynamic_energy),
        ) = self.dyn_eam.attribute_energy(force=force)

        energies = [
            cpu_pid_total_energy,
            cpu_pid_idle_energy,
            cpu_pid_dynamic_energy,
            dram_pid_total_energy,
            dram_pid_idle_energy,
            dram_pid_dynamic_energy,
        ]

        return energies

    def store_data(self):
        ds_factory = DataStorageFactory()
        self._data_storage = ds_factory.create_data_storage(self._storage_type, db_url=self._storage_path)

        while not self._stop_event.is_set():
            try:
                # Process entities
                self._process_entities()

                # Process measurements
                self._process_measurements()

                # sleep to avoid busy waiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in data saving loop: {e}")
                time.sleep(1)

        self._stop()

    def _process_entities(self):
        entities = []
        while True:
            try:
                item = self._entities_queue.get_nowait()
                if item == DataCollector.SENT_E:
                    self._received_sent_e = True
                    continue
                entities.extend(item)
            except Empty:
                break

        if entities:
            self._data_storage.save_entities(entities)

    def _process_measurements(self):
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

        if len(dfs) == 0:
            return
        self._data_storage.save_measurements(dfs)
        del dfs

    def _stop(self):
        # Final drain of queues using sentinels
        current_iter_e = 0
        while not self._received_sent_e:
            current_iter_e += 1
            if current_iter_e > 1000:
                logger.warning("Stuck waiting for ENTITY sentinel, breaking loop")
                break
            try:
                item = self._entities_queue.get(timeout=1)

                if item == DataCollector.SENT_E:
                    self._received_sent_e = True
                    break
                self._data_storage.save_entities(item)
            except Empty:
                continue

        current_iter_m = 0
        while not self._received_sent_m:
            current_iter_m += 1
            if current_iter_m > 1000:  # Safety break
                logger.warning("Stuck waiting for MEASUREMENT sentinel, breaking loop")
                break
            try:
                path = self._measurements_queue.get(timeout=1)
                if path == DataCollector.SENT_M:
                    self._received_sent_m = True
                    break
                df = read_temp_ipc(path)
                self._data_storage.save_measurements([df])
                del df
            except Empty:
                continue

        # Energies calculation logic from original _stop was commented out mostly
        # energies = []
        # self._data_storage.save_data_record(DataRecord(entities, energies), progressbar=True)
        # Use simple close

        self._data_storage.close()
        self._entities_queue.close()
        self._measurements_queue.close()
        self._entities_queue.join_thread()
        self._measurements_queue.join_thread()


def start_data_saver(data_saver: DataSaver):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    data_saver.store_data()


class DataCollector:
    """Collects data records and manages their storage using configurable strategies."""

    SENT_E = ("__EOF__", "entities")
    SENT_M = ("__EOF__", "measurements")

    def __init__(self, storage_type: str, storage_path: str):
        """
        Initialize the DataCollector.

        Args:
            strategy: Dictionary with 'type' ('interval' or 'max_items') and 'value'
            data_storage: Storage interface for persisting data records
        """
        self._ctx = mp.get_context("spawn")

        self._entities_queue = self._ctx.Queue()
        self._measurements_queue = self._ctx.Queue()
        self._storage_type = storage_type
        self._storage_path = storage_path

        # Buffering for measurements
        self._measurement_buffer = []
        self._buffer_row_count = 0
        self._max_buffer_rows = 1_000_000

        self._stop_event = self._ctx.Event()

        self.process = None

        # IPC writer
        self.ipc_writer = IpcAsyncWriter(
            out_paths_queue=self._measurements_queue,
        )

        self._start_collection_process()

    def _start_collection_process(self):
        """Start the collection process for both strategies."""
        self.data_saver = DataSaver(
            self._entities_queue, self._measurements_queue, self._stop_event, self._storage_type, self._storage_path
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

    def stop(self):
        """Stop the data collection process gracefully."""
        # Flush any remaining measurements
        self._flush_measurements()
        self.ipc_writer.close()

        self._entities_queue.put(DataCollector.SENT_E)
        self._measurements_queue.put(DataCollector.SENT_M)
        self._stop_event.set()
        logger.debug("Saving final data...")

        self._entities_queue.close()
        self._measurements_queue.close()
        self._entities_queue.cancel_join_thread()
        self._measurements_queue.cancel_join_thread()

        if self.process:
            self.process.join(timeout=600)
            if self.process.is_alive():
                logger.warning("Process didn't stop gracefully, terminating...")
                self.process.kill()
                self.process.join(timeout=5)
            self.process = None

        # close queues
        self._entities_queue.close()
        self._measurements_queue.close()

        logger.info("DataCollector stopped.")
