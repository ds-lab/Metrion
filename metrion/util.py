import multiprocessing as mp
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import polars as pl
from loguru import logger

from metrion.entity.entity import Entity
from metrion.measurement import CommunicationMeasurement, Measurement


@dataclass
class _Batch:
    """Frames waiting to be written as one IPC file, plus when the first of them arrived."""

    dfs: list[pl.DataFrame] = field(default_factory=list)
    rows: int = 0
    oldest: float | None = None

    def add(self, df: pl.DataFrame) -> None:
        if not self.dfs:
            self.oldest = time.monotonic()
        self.dfs.append(df)
        self.rows += df.height

    def clear(self) -> None:
        self.dfs = []
        self.rows = 0
        self.oldest = None

    def is_older_than(self, max_age_s: float) -> bool:
        return self.oldest is not None and (time.monotonic() - self.oldest) >= max_age_s


class IpcAsyncWriter:
    """
    Background IPC writer.

    Enqueue Polars DataFrames; the writer keeps one batch per ``df_type`` and writes it to a
    temporary IPC file once the batch reaches ``max_rows`` rows or its oldest frame is
    ``max_age_s`` old, then puts the file path on ``out_paths_queue``.

    Parameters
    ----------
    out_paths_queue : mp.Queue
        Receives the path of every file the writer produces.
    max_rows : int
        Row count at which a batch is written out.
    max_age_s : float
        Age of the oldest buffered frame at which a batch is written out, even if the row
        threshold was never reached.
    assign_ids : bool
        Give every row a snowflake ``id`` before buffering it. Set it for measurement frames;
        raw event frames do not need one.
    """

    _SENTINEL = object()

    def __init__(
        self,
        out_paths_queue: mp.Queue,
        max_rows: int = 500_000,
        max_age_s: float = 2.0,
        assign_ids: bool = False,
    ):
        self._out_paths_queue = out_paths_queue
        self._max_rows = int(max_rows)
        self._max_age_s = float(max_age_s)
        self._poll_interval = max(0.01, min(0.25, self._max_age_s / 4))
        self._assign_ids = assign_ids

        self._batches: dict[str, _Batch] = {}

        self._jobs: Queue[object] = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, dfs: pl.DataFrame | Sequence[pl.DataFrame], df_type: str = "measurement") -> None:
        if isinstance(dfs, pl.DataFrame):
            dfs = [dfs]
        if not isinstance(dfs, Sequence):
            raise TypeError(f"Expected pl.DataFrame or Sequence[pl.DataFrame], got {type(dfs)!r}")
        if not isinstance(df_type, str):
            raise TypeError(f"df_type must be a str, got {type(df_type)!r}")
        self._jobs.put((list(dfs), df_type))

    def close(self) -> None:
        self._jobs.put(self._SENTINEL)
        self._thread.join()

    def _emit_path(self, path: str | Path) -> None:
        self._out_paths_queue.put(str(path))

    def _write_df(self, df: pl.DataFrame) -> None:
        self._emit_path(write_temp_ipc(df))

    def _batch_for(self, df_type: str) -> _Batch:
        return self._batches.setdefault(df_type, _Batch())

    def _flush(self, df_type: str) -> None:
        batch = self._batch_for(df_type)
        if not batch.dfs:
            return
        try:
            self._write_df(pl.concat(batch.dfs, how="vertical"))
        except Exception:
            schemas = [df.schema for df in batch.dfs]
            logger.opt(exception=True).warning(f"IPC writer dropped a {df_type} batch, schemas: {schemas}")
        finally:
            # A batch that cannot be written would otherwise be retried forever and keep growing.
            batch.clear()

    def _flush_all(self) -> None:
        for df_type in list(self._batches):
            self._flush(df_type)

    def _flush_aged(self) -> None:
        for df_type, batch in list(self._batches.items()):
            if batch.is_older_than(self._max_age_s):
                self._flush(df_type)

    def _buffer(self, dfs: Sequence[pl.DataFrame], df_type: str) -> None:
        batch = self._batch_for(df_type)
        for df in dfs:
            if not isinstance(df, pl.DataFrame):
                raise TypeError(f"Expected pl.DataFrame, got {type(df)!r}")
            if self._assign_ids:
                df = _add_id_to_measurement_df(df)

            # Huge frame: write out what is pending, then write the frame in chunks.
            if df.height > self._max_rows:
                self._flush(df_type)
                for chunk in split_large_df(df, self._max_rows):
                    self._write_df(chunk)
                continue

            # Would overflow: write out what is pending first.
            if batch.rows + df.height > self._max_rows:
                self._flush(df_type)

            batch.add(df)

    def _run(self) -> None:
        while True:
            try:
                item = self._jobs.get(timeout=self._poll_interval)
            except Empty:
                # No new work. Only check whether a batch has aged out; nothing was taken
                # off the queue, so no task_done() here.
                self._flush_aged()
                continue

            try:
                if item is self._SENTINEL:
                    self._flush_all()
                    return

                dfs, df_type = item
                self._buffer(dfs, df_type)
                self._flush_aged()

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


def _generate_id(n) -> pl.Series:
    ids = []
    gen = Measurement.get_snowflake_generator()
    for _ in range(n):
        id = next(gen)
        ids.append(id)
    return pl.Series("id", ids)


def _add_id_to_measurement_df(df: pl.DataFrame) -> pl.DataFrame:
    if "id" not in df.columns:
        n = df.height
        id_series = _generate_id(n)
        df = df.with_columns(id_series)
    else:
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
