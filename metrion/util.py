import multiprocessing as mp
import os
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from queue import Queue
from threading import Thread

import polars as pl
from loguru import logger

from metrion.entity.entity import Entity
from metrion.measurement import CommunicationMeasurement, Measurement


class IpcAsyncWriterV2:
    """
    Background IPC writer.
    Enqueue Polars DataFrames; writer batches up to `max_rows` then writes IPC to disk
    and emits the file path to `out_paths_queue`.
    """

    def __init__(self, out_paths_queue: mp.Queue, max_rows: int = 500_000):
        self._out_paths_queue = out_paths_queue
        self._max_rows = int(max_rows)

        self._batch: list[pl.DataFrame] = []
        self._batch_rows: int = 0

        self._sentinel = object()

        self._jobs: Queue[object] = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, dfs: pl.DataFrame | Sequence[pl.DataFrame]) -> None:
        if isinstance(dfs, pl.DataFrame):
            self._jobs.put([dfs])
        else:
            self._jobs.put(list(dfs))

    def close(self) -> None:
        self._jobs.put(self._sentinel)
        self._thread.join()

    def _emit_path(self, path: str | Path) -> None:
        self._out_paths_queue.put(str(path))

    def _write_df(self, df: pl.DataFrame) -> None:
        self._emit_path(write_temp_ipc(df))

    def _flush_batch(self) -> None:
        if not self._batch:
            return
        self._write_df(pl.concat(self._batch, how="vertical"))
        self._batch.clear()
        self._batch_rows = 0

    def _run(self) -> None:
        import logging

        log = logging.getLogger(__name__)

        while True:
            item = self._jobs.get()
            try:
                if item is self._sentinel:
                    self._flush_batch()
                    return

                if not isinstance(item, list):
                    raise TypeError(f"Expected list[pl.DataFrame], got {type(item)!r}")

                for df in item:
                    if not isinstance(df, pl.DataFrame):
                        raise TypeError(f"Expected pl.DataFrame, got {type(df)!r}")

                    # Huge df: flush current batch, then write chunked.
                    if df.height > self._max_rows:
                        self._flush_batch()
                        for chunk in split_large_df(df, self._max_rows):
                            self._write_df(chunk)
                        continue

                    # Would overflow: flush first.
                    if self._batch_rows + df.height > self._max_rows:
                        self._flush_batch()

                    self._batch.append(df)
                    self._batch_rows += df.height

            except Exception:
                log.exception("IPC writer failed")
            finally:
                self._jobs.task_done()


class IpcAsyncWriter:
    _SENTINEL = object()

    """
    Background IPC writer.
    - enqueue(dfs): returns quickly (unless queue is full -> backpressure)
    - worker writes df to a temp file and pushes the path to out_paths_queue
    """

    def __init__(
        self,
        out_paths_queue: mp.Queue,
        max_rows: int = 500_000,
    ):
        self._out_paths_queue = out_paths_queue
        self._jobs: Queue[object] = Queue()
        self._max_rows = max_rows

        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, dfs: Sequence[pl.DataFrame], df_type: str) -> None:
        self._jobs.put((dfs, df_type))

    def close(self) -> None:
        self._jobs.put(self._SENTINEL)
        self._thread.join()

    def _emit_path(self, path: Path) -> None:
        self._out_paths_queue.put(str(path))

    def _flush_measurements(self, measurement_dfs: list[pl.DataFrame]) -> None:
        if not measurement_dfs:
            return
        try:
            measurement_df = pl.concat(measurement_dfs, how="vertical")
            path = Path(write_temp_ipc(measurement_df))
            self._emit_path(path)
        except Exception as e:
            logger.warning(f"IPC writer failed during measurement flush with {e}")
            schema = [df.schema for df in measurement_dfs]
            print(schema)

    def _run(self) -> None:
        running_measurement_rows = 0
        measurement_dfs: list[pl.DataFrame] = []

        while True:
            item = self._jobs.get()
            try:
                if item is self._SENTINEL:
                    # flush leftovers before exit
                    self._flush_measurements(measurement_dfs)
                    return

                dfs, df_type = item
                if not isinstance(df_type, str):
                    raise TypeError("df_type must be a str")

                # Accept any sequence/iterable of dfs
                if not isinstance(dfs, Sequence):
                    raise TypeError("dfs must be a Sequence[pl.DataFrame]")

                if df_type == "measurement":
                    for df in dfs:
                        df = _add_id_to_measurement_df(df)
                        if df.height > self._max_rows:
                            # flush pending small ones first to keep batching consistent
                            self._flush_measurements(measurement_dfs)
                            measurement_dfs = []
                            running_measurement_rows = 0

                            for chunk in split_large_df(df, self._max_rows):
                                path = write_temp_ipc(chunk)
                                self._emit_path(path)
                            continue

                        # flush BEFORE overflow, so written batch stays <= _max_rows
                        if running_measurement_rows + df.height > self._max_rows:
                            self._flush_measurements(measurement_dfs)
                            measurement_dfs = []
                            running_measurement_rows = 0

                        measurement_dfs.append(df)
                        running_measurement_rows += df.height

                elif df_type == "communication_measurement":
                    comm_df = pl.concat(dfs, how="vertical")
                    comm_df = _add_id_to_measurement_df(comm_df)
                    path = write_temp_ipc(comm_df)
                    self._emit_path(path)

                else:
                    raise ValueError(f"Unknown df_type: {df_type}")

            except Exception:
                logger.exception("IPC writer failed")
            finally:
                self._jobs.task_done()


def check_entity_type(entities: Sequence[Entity], expected_type: type | None = None) -> None:
    """Check all entities are of the same concrete type and that this type is compatible with expected_type."""
    if not entities:
        raise ValueError("entities must not be empty")

    first_cls = type(entities[0])

    if not all(type(entity) is first_cls for entity in entities):
        raise ValueError("All entities must be of the same concrete type")

    if expected_type:
        if not issubclass(first_cls, expected_type):
            raise ValueError(f"Entities are of type {first_cls.__name__}, expected {expected_type.__name__}")


def split_large_df(df: pl.DataFrame, max_rows: int = 300_000) -> list[pl.DataFrame]:
    """Split a large DataFrame into smaller DataFrames."""
    if df.height <= max_rows:
        return [df]

    dfs = []
    for i in range(0, df.height, max_rows):
        dfs.append(df.slice(i, max_rows))
    return dfs


def write_temp_ipc(df: pl.DataFrame) -> str:
    tmpdir = Path(tempfile.gettempdir())
    fname = f"batch_{uuid.uuid4().hex}.ipc"
    path = tmpdir / fname
    df.write_ipc(path, compression="uncompressed")
    return str(path)


def write_tmp_parquet(df: pl.DataFrame) -> str:
    tmpdir = Path(tempfile.gettempdir())
    fname = f"batch_{uuid.uuid4().hex}.parquet"
    path = tmpdir / fname
    df.write_parquet(path, compression="uncompressed")
    return str(path)


def read_temp_ipc(path: str) -> pl.DataFrame:
    path = Path(path)
    df = pl.read_ipc(path)
    Path(path).unlink(missing_ok=True)
    if "start_ts" in df.columns and "end_ts" in df.columns:
        if df["start_ts"].dtype == pl.String and df["end_ts"].dtype == pl.String:
            df = df.with_columns(
                pl.col("start_ts").str.to_datetime("%Y-%m-%d %H:%M:%S%.f"),
                pl.col("end_ts").str.to_datetime("%Y-%m-%d %H:%M:%S%.f"),
            )
    return df


def _generate_id(n: int) -> pl.Series:
    ids: list[int] = []
    gen = Measurement.get_snowflake_generator()

    while len(ids) < n:
        generated_id = next(gen)
        if generated_id is not None:
            ids.append(generated_id)

    return pl.Series("id", ids, dtype=pl.Int64)


def _add_id_to_measurement_df(df: pl.DataFrame) -> pl.DataFrame:
    if "id" not in df.columns:
        n = df.height
        id_series = _generate_id(n)
        df = df.with_columns(id_series)
    else:
        df = df.with_columns(pl.col("id").cast(pl.Int64))

        missing_ids = df.filter(pl.col("id").is_null())
        having_ids = df.filter(pl.col("id").is_not_null())
        n_missing = missing_ids.height

        if n_missing > 0:
            id_series = _generate_id(n_missing)
            missing_with_ids = missing_ids.with_columns(id_series)
            df = pl.concat([having_ids, missing_with_ids])

    df = df.with_columns(pl.col("id").cast(pl.Int64))
    return df


def transform_measuremnent_list_to_polars(
    measurements: Sequence[Measurement | pl.DataFrame],
) -> tuple[Sequence[pl.DataFrame], Sequence[pl.DataFrame]]:
    """Transform a list of measurement dicts to a Polars DataFrame."""

    if len(measurements) == 0:
        return [], []

    data_measurements_df = []
    data_communication_measurements_df = []

    for m in measurements:
        if isinstance(m, Measurement):
            m = m.to_df()
            data_measurements_df.append(m)
        elif isinstance(m, CommunicationMeasurement):
            m = m.to_df()
            data_communication_measurements_df.append(m)
        elif isinstance(m, pl.DataFrame):
            if "src_endpoint" in m.columns and "dst_endpoint" in m.columns:
                data_communication_measurements_df.append(m)
            else:
                data_measurements_df.append(m)
            continue
        else:
            raise ValueError("measurements must be a list of Measurement or Polars DataFrame")

    return data_measurements_df, data_communication_measurements_df


def build_cgroup_id_map(root="/sys/fs/cgroup") -> dict[int, str]:
    id2path = {}
    for dirpath, _, _ in os.walk(root):
        # cgroup name
        rel = os.path.relpath(dirpath, root)
        path = "/" if rel == "." else "/" + rel

        # inode of directory
        try:
            st = os.stat(dirpath)
        except FileNotFoundError:
            continue
        id2path[st.st_ino] = path

    return id2path
