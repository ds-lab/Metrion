import json
import time
from collections.abc import Sequence
from pathlib import Path

import polars as pl
from loguru import logger

from metrion.data_record import DataRecord
from metrion.data_storage.i_data_storage import DataStorageI
from metrion.entity.entity import Entity
from metrion.measurement import CommunicationMeasurement, Measurement
from metrion.util import _add_id_to_measurement_df, naive_utc_timestamps


class _RotatingParquetWriter:
    """Collects frames and writes one numbered Parquet file per `rotate_rows` rows."""

    def __init__(self, directory: Path, rotate_rows: int, compression: str = "zstd", max_age_s: float | None = None):
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._rotate_rows = rotate_rows
        self._compression = compression
        self._max_age_s = max_age_s
        self._pending: list[pl.DataFrame] = []
        self._pending_rows = 0
        self._oldest_pending: float | None = None
        # Continue behind the parts an earlier run left here; starting at zero again would
        # overwrite them.
        existing = [int(path.stem.split("-")[1]) for path in self._dir.glob("part-*.parquet")]
        self._part = max(existing) + 1 if existing else 0
        self._total_rows = 0

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def write(self, df: pl.DataFrame) -> None:
        if df.is_empty():
            return

        if not self._pending:
            self._oldest_pending = time.monotonic()
        self._pending.append(df)
        self._pending_rows += df.height
        self._total_rows += df.height

        if self._pending_rows >= self._rotate_rows:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return

        path = self._dir / f"part-{self._part:05d}.parquet"
        # diagonal, not vertical: a frame with a missing column must not cost the whole
        # pending buffer. The missing values become null and are filtered on read.
        pl.concat(self._pending, how="diagonal_relaxed").write_parquet(path, compression=self._compression)

        self._pending.clear()
        self._pending_rows = 0
        self._oldest_pending = None
        self._part += 1

    def flush_aged(self) -> None:
        """Write the buffer out once its oldest row has waited max_age_s.

        For streams with a low row rate: without it the first file appears only when
        rotate_rows is reached, which for the attribution results can take hours.
        """
        if self._max_age_s is None or self._oldest_pending is None:
            return
        if time.monotonic() - self._oldest_pending >= self._max_age_s:
            self.flush()


class ParquetDataStorage(DataStorageI):
    """Append-only Parquet sink.

    Nothing reads the sink while a run is in progress, so it carries no primary keys, no
    upserts and no deduplication. Repeated entity versions are resolved on read.
    """

    MEASUREMENTS_DIR = "measurements"
    COMM_MEASUREMENTS_DIR = "communication_measurements"

    def __init__(self, path: str, rotate_rows: int = 200_000):
        self._root = Path(path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._rotate_rows = rotate_rows
        self._writers: dict[str, _RotatingParquetWriter] = {}
        self._entity_seq = 0
        logger.info(f"ParquetDataStorage writes to {self._root.resolve()}")

    def _writer(self, name: str) -> _RotatingParquetWriter:
        if name not in self._writers:
            self._writers[name] = _RotatingParquetWriter(self._root / name, self._rotate_rows)
        return self._writers[name]

    def save_measurements(self, measurements: Sequence[Measurement | pl.DataFrame], progressbar: bool = False) -> None:
        frames: list[pl.DataFrame] = []
        comm_frames: list[pl.DataFrame] = []
        objects: list[Measurement] = []
        comm_objects: list[CommunicationMeasurement] = []

        for measurement in measurements:
            if isinstance(measurement, pl.DataFrame):
                is_comm = "src_endpoint" in measurement.columns and "dst_endpoint" in measurement.columns
                target = comm_frames if is_comm else frames
                target.append(_add_id_to_measurement_df(measurement))
            elif isinstance(measurement, CommunicationMeasurement):
                comm_objects.append(measurement)
            elif isinstance(measurement, Measurement):
                objects.append(measurement)
            else:
                logger.warning(f"Unknown measurement type: {type(measurement)}")

        if objects:
            frames.append(pl.concat([m.to_df() for m in objects], how="vertical"))
        if comm_objects:
            comm_frames.append(pl.concat([m.to_df() for m in comm_objects], how="vertical"))

        for name, batch in ((self.MEASUREMENTS_DIR, frames), (self.COMM_MEASUREMENTS_DIR, comm_frames)):
            if not batch:
                continue
            writer = self._writer(name)
            for frame in batch:
                writer.write(naive_utc_timestamps(frame))

    def save_entities(self, entities: Sequence[Entity], progressbar: bool = False) -> None:
        """Write entities append-only, one stream per concrete type.

        Deliberately without deduplication: the same entity may reappear in later cycles
        with richer metadata. The versions are merged on read.
        """
        by_type: dict[str, list[pl.DataFrame]] = {}
        for entity in entities:
            # CPUs carry their cores as sub-objects; the DB path writes those as well.
            for item in (entity, *getattr(entity, "cores", ())):
                frame = item._to_df().with_columns(pl.lit(type(item).__name__).alias("_entity_type"))
                by_type.setdefault(type(item).__name__.lower(), []).append(frame)

        # One sequence number per call, so a later version of an entity can be told from an
        # earlier one on read. A timestamp would not be enough: two calls can share a
        # microsecond, and DuckDB is read with preserve_insertion_order off, so there is no
        # implicit order to fall back on.
        self._entity_seq += 1
        for name, frames in by_type.items():
            df = pl.concat(frames, how="diagonal_relaxed").with_columns(
                pl.lit(self._entity_seq, dtype=pl.Int64).alias("_seq")
            )
            self._writer(f"entities/{name}").write(df)

    def save_data_record(self, data_record: DataRecord, progressbar: bool = False) -> None:
        if len(data_record.entities) > 0:
            self.save_entities(data_record.entities)
        if len(data_record.measurements) > 0:
            self.save_measurements(data_record.measurements)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.flush()

        streams = {name: writer.total_rows for name, writer in self._writers.items()}
        (self._root / "manifest.json").write_text(json.dumps({"streams": streams}, indent=2))
        logger.info(f"ParquetDataStorage closed: {streams}")
