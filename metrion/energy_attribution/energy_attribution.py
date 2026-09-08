"""In-memory energy attribution routines using Polars."""

# isort: skip_file
from __future__ import annotations

import datetime as dt
from datetime import UTC
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path

import polars as pl
from loguru import logger

from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.entity import Entity
from metrion.entity.gpu import GPU
from metrion.measurement import Measurement
from metrion.metric import (
    CPU_PKG_PID_TOTAL_ENERGY,
    CPU_PKG_PID_IDLE_ENERGY,
    CPU_PKG_PID_DYNAMIC_ENERGY,
    DRAM_SOCKET_PID_DYNAMIC_ENERGY,
    DRAM_SOCKET_PID_IDLE_ENERGY,
    DRAM_SOCKET_PID_TOTAL_ENERGY,
    GPU_DEVICE_PID_TOTAL_ENERGY,
    GPU_DEVICE_PID_IDLE_ENERGY,
    GPU_DEVICE_PID_DYNAMIC_ENERGY,
)
from metrion.util import naive_utc_timestamps


class MeasurementBuffer:
    """Hybrid memory/disk buffer for measurements with automatic spill.

    Stores measurements in memory up to a configurable limit, then spills
    oldest data to Parquet files on disk. Required for handling late-arriving
    measurements (up to 60+ seconds delay) without unbounded memory growth.

    Parameters
    ----------
    memory_limit_rows : int
        Maximum rows to keep in memory before spilling to disk.
    spill_dir : Path, optional
        Directory for spill files. Defaults to system temp directory.

    Examples
    --------
    >>> buffer = MeasurementBuffer(memory_limit_rows=100_000)
    >>> buffer.add(new_measurements_df)
    >>> all_data = buffer.get_all()
    >>> buffer.remove_before(cutoff_time)
    """

    def __init__(self, memory_limit_rows: int = 500_000, spill_dir: Path | None = None) -> None:
        self._memory_df = pl.DataFrame()
        self._spill_dir = spill_dir or Path(tempfile.gettempdir()) / "metrion_spill"
        self._spill_files: list[Path] = []
        self._memory_limit = memory_limit_rows

    def add(self, df: pl.DataFrame) -> None:
        """Add measurements to buffer, spilling to disk if needed.

        Parameters
        ----------
        df : pl.DataFrame
            Measurements to add with end_ts column for ordering.
        """
        if df.is_empty():
            return
        self._memory_df = pl.concat([self._memory_df, df])
        if self._memory_df.height > self._memory_limit:
            self._spill_to_disk()

    def _spill_to_disk(self) -> None:
        """Write oldest half of buffer to Parquet file."""
        self._spill_dir.mkdir(parents=True, exist_ok=True)
        half = self._memory_df.height // 2
        # Sort by end_ts to spill oldest first
        sorted_df = self._memory_df.sort("end_ts")
        to_spill = sorted_df.head(half)
        self._memory_df = sorted_df.tail(self._memory_df.height - half)

        spill_path = self._spill_dir / f"spill_{dt.datetime.now(UTC).timestamp()}.parquet"
        to_spill.write_parquet(spill_path)
        self._spill_files.append(spill_path)
        logger.debug(f"Spilled {half} measurement rows to {spill_path}")

    def get_all(self) -> pl.DataFrame:
        """Read all data from disk and memory.

        Returns
        -------
        pl.DataFrame
            Combined measurements from all spill files and memory.
        """
        dfs = []
        for f in self._spill_files:
            if f.exists():
                dfs.append(pl.read_parquet(f))
        dfs.append(self._memory_df)
        return pl.concat(dfs) if dfs else pl.DataFrame()

    def remove_before(self, cutoff: dt.datetime) -> None:
        """Remove measurements older than cutoff from memory and disk.

        Parameters
        ----------
        cutoff : dt.datetime
            Remove all measurements with end_ts <= cutoff.
        """
        self._memory_df = self._memory_df.filter(pl.col("end_ts") > cutoff)
        # Clean up spill files that are entirely before cutoff
        for f in self._spill_files[:]:
            if not f.exists():
                self._spill_files.remove(f)
                continue
            df = pl.read_parquet(f)
            remaining = df.filter(pl.col("end_ts") > cutoff)
            if remaining.is_empty():
                f.unlink()
                self._spill_files.remove(f)
                logger.debug(f"Removed spill file {f}")
            elif remaining.height < df.height:
                # Partial cleanup - rewrite file
                remaining.write_parquet(f)

    def height(self) -> int:
        """Total row count across memory and disk."""
        disk_rows = sum(pl.read_parquet(f).height for f in self._spill_files if f.exists())
        return disk_rows + self._memory_df.height

    def is_empty(self) -> bool:
        """Check if buffer has no data."""
        return self._memory_df.is_empty() and not self._spill_files

    def cleanup(self) -> None:
        """Remove all spill files."""
        for f in self._spill_files:
            if f.exists():
                f.unlink()
        self._spill_files.clear()


class EnergyAttributionModelBase(ABC):
    """Abstract base class for energy attribution models.

    Provides shared utilities for time bound inference and empty result
    generation used by CPU, DRAM, and GPU attribution models.
    """

    @classmethod
    def _determine_start_end_times(
        cls,
        measurement_df: pl.DataFrame,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
    ) -> tuple[dt.datetime | None, dt.datetime | None]:
        """Infer time bounds from measurements if not provided.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements with start_ts and end_ts columns.
        start_time : datetime, optional
            Explicit start time, or None to infer from data.
        end_time : datetime, optional
            Explicit end time, or None to infer from data.

        Returns
        -------
        tuple[datetime | None, datetime | None]
            (start_time, end_time) bounds for filtering.
        """
        if start_time is None:
            inferred_start = measurement_df.get_column("start_ts").min()
            if inferred_start is None:
                return None, None
            if not isinstance(inferred_start, dt.datetime):
                raise TypeError("measurement.start_ts must be datetime-like")
            start_time = inferred_start
        if end_time is None:
            inferred_end = measurement_df.get_column("end_ts").max()
            if inferred_end is None:
                return None, None
            if not isinstance(inferred_end, dt.datetime):
                raise TypeError("measurement.end_ts must be datetime-like")
            end_time = inferred_end
        return start_time, end_time

    @staticmethod
    def _empty_result() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Return empty result DataFrames with standard schema.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            Three empty DataFrames with (metric_id, physical_entity_id,
            logical_entity_id, value, start_ts, end_ts) columns.
        """
        empty_df = pl.DataFrame(
            schema={
                "metric_id": pl.String,
                "physical_entity_id": pl.String,
                "logical_entity_id": pl.String,
                "value": pl.Float64,
                "start_ts": pl.Datetime,
                "end_ts": pl.Datetime,
            }
        )
        return (empty_df, empty_df, empty_df)

    @staticmethod
    def _join_overlapping(
        measurements: pl.DataFrame | pl.LazyFrame,
        intervals: pl.DataFrame | pl.LazyFrame,
        predicates: Sequence[pl.Expr],
        value_col: str | None = None,
        meas_start: str = "meas_start",
        meas_end: str = "meas_end",
        interval_start: str = "interval_start",
        interval_end: str = "interval_end",
    ) -> pl.DataFrame | pl.LazyFrame:
        """Join measurements to every energy interval they overlap.

        Parameters
        ----------
        measurements : pl.DataFrame or pl.LazyFrame
            Left frame, carrying the measurement window.
        intervals : pl.DataFrame or pl.LazyFrame
            Right frame, carrying the energy interval window.
        predicates : Sequence[pl.Expr]
            Additional join conditions, usually the entity binding. A column present in
            both frames is addressed with the ``_right`` suffix on the interval side.
        value_col : str, optional
            Additive column, scaled by the share of the measurement that falls into the
            interval. Leave at None for rates and levels, which must not be split.
        meas_start, meas_end, interval_start, interval_end : str
            Column names of the two windows.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            One row per measurement and overlapped interval. With ``value_col`` set, the
            shares of a single measurement add up to its original value. The measurement
            window columns keep their original, unclipped values and therefore repeat
            across the rows of one measurement; the interval columns say which slice a
            row belongs to, so group and join on those.
        """
        # Point measurements (start_ts == end_ts) overlap no half-open interval, so they
        # get the smallest representable width before the join.
        measurements = measurements.with_columns(
            pl.when(pl.col(meas_end) <= pl.col(meas_start))
            .then(pl.col(meas_start) + pl.duration(microseconds=1))
            .otherwise(pl.col(meas_end))
            .alias(meas_end)
        )

        joined = measurements.join_where(
            intervals,
            *predicates,
            pl.col(meas_start) < pl.col(interval_end),
            pl.col(meas_end) > pl.col(interval_start),
        )

        if value_col is None:
            return joined

        return (
            joined.with_columns(
                pl.max_horizontal(
                    pl.col(meas_start).dt.timestamp("ns"),
                    pl.col(interval_start).dt.timestamp("ns"),
                ).alias("_overlap_start_ns"),
                pl.min_horizontal(
                    pl.col(meas_end).dt.timestamp("ns"),
                    pl.col(interval_end).dt.timestamp("ns"),
                ).alias("_overlap_end_ns"),
                (pl.col(meas_end).dt.timestamp("ns") - pl.col(meas_start).dt.timestamp("ns")).alias("_span_ns"),
            )
            .with_columns(
                (
                    pl.col(value_col) * (pl.col("_overlap_end_ns") - pl.col("_overlap_start_ns")) / pl.col("_span_ns")
                ).alias(value_col)
            )
            .drop(["_overlap_start_ns", "_overlap_end_ns", "_span_ns"])
        )

    @abstractmethod
    def attribute_energy(
        self,
        measurement_df: pl.DataFrame,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute energy to processes.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) DataFrames.
        """
        pass


class CPUEnergyAttributionModel(EnergyAttributionModelBase):
    """CPU package energy attribution using RAPL counters.

    Distributes CPU package energy to processes based on CPU work and
    execution time metrics. Dynamic energy is proportional to work done,
    idle energy is proportional to execution time.

    Attributes
    ----------
    _PKG_METRIC : str
        Metric ID for CPU package energy counter.
    _WORK_METRIC : str
        Metric ID for per-process CPU work.
    _TIME_METRIC : str
        Metric ID for per-process execution time.
    _ENERGY_WRAP_CPU : float
        Energy counter wrap-around value in Joules.
    """

    _PKG_METRIC = "CPU_SOCKET_PKG_ENERGY"
    _WORK_METRIC = "CPU_CORE_PID_WORK"
    _TIME_METRIC = "CPU_CORE_PID_TIME"
    _ENERGY_WRAP_CPU = 262_143_328_850.0 / 1_000_000.0

    def __init__(self) -> None:
        pass

    def attribute_energy(
        self,
        measurement_df: pl.DataFrame,
        cpu_df: pl.DataFrame | None = None,
        core_df: pl.DataFrame | None = None,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute CPU package energy to processes.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements with CPU_SOCKET_PKG_ENERGY and work/time metrics.
        cpu_df : pl.DataFrame, optional
            CPU entity data with idle_power_package.
        core_df : pl.DataFrame, optional
            Core entity data for socket mapping.
        start_time : datetime, optional
            Filter start time.
        end_time : datetime, optional
            Filter end time.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) per process.
        """
        if cpu_df is None or core_df is None:
            return self._empty_result()
        return self._attribute_pkg_energy(measurement_df, cpu_df, core_df, start_time, end_time)

    def _split_over_intervals(
        self, measurements: pl.DataFrame, energy_intervals: pl.DataFrame, value_col: str
    ) -> pl.DataFrame | pl.LazyFrame:
        """Distribute measurements over the energy intervals they overlap.

        A measurement touching two intervals appears in both, each time carrying the
        share of its duration that falls into that interval, so the shares add up to the
        original value.

        Parameters
        ----------
        measurements : pl.DataFrame
            Columns package_id, meas_start, meas_end and value_col.
        energy_intervals : pl.DataFrame
            Columns package_id, interval_start, interval_end.
        value_col : str
            Name of the column to split.
        """
        return self._join_overlapping(
            measurements,
            energy_intervals,
            [pl.col("package_id") == pl.col("package_id_right")],
            value_col=value_col,
        )

    def _attribute_pkg_energy(
        self,
        measurement_df: pl.DataFrame,
        cpu_df: pl.DataFrame,
        core_df: pl.DataFrame,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if measurement_df.is_empty():
            return self._empty_result()

        start_time, end_time = self._determine_start_end_times(measurement_df, start_time=start_time, end_time=end_time)

        # Closed bounds on purpose: these samples are not overlapped, they anchor the
        # intervals. The sample at end_time is the lead value of the last interval, and
        # dropping it would lose that interval's energy.
        energy_samples = measurement_df.filter(
            (pl.col("metric_id") == self._PKG_METRIC)
            & (pl.col("start_ts") >= start_time)
            & (pl.col("start_ts") <= end_time)
        ).select(
            pl.col("value"),
            pl.col("metric_id"),
            pl.col("physical_entity_id").alias("package_id"),
            pl.col("logical_entity_id"),
            pl.col("start_ts"),
            pl.col("end_ts"),
        )

        if energy_samples.is_empty():
            return self._empty_result()

        energy_intervals = (
            energy_samples.sort(["package_id", "start_ts"])
            .with_columns(
                pl.col("start_ts").alias("interval_start"),
                pl.col("start_ts").shift(-1).over(partition_by="package_id", order_by="start_ts").alias("interval_end"),
                pl.col("value").shift(-1).over(partition_by="package_id", order_by="start_ts").alias("lead_value"),
            )
            .with_columns(
                pl.when(pl.col("lead_value").is_null())
                .then(0.0)
                .when(pl.col("lead_value") >= pl.col("value"))
                .then(pl.col("lead_value") - pl.col("value"))
                .otherwise(pl.lit(self._ENERGY_WRAP_CPU) - pl.col("value") + pl.col("lead_value"))
                .alias("energy_j")
            )
            .filter(pl.col("interval_end").is_not_null())
            .select(["package_id", "interval_start", "interval_end", "energy_j"])
            .sort(["package_id", "interval_start"])
        )

        # measurements holds the cpu core
        # cpu core holds a the parent_id which is the socket id

        core_mapping = core_df.select(["id", "parent_id"]).rename({"id": "core_id", "parent_id": "package_id"})

        # Overlap against the half-open window [start_time, end_time), matching the
        # convention of _join_overlapping. Both bounds exclusive: a measurement ending
        # exactly at start_time or starting exactly at end_time overlaps no interval
        # inside the window, and with closed bounds it would land in this window and in
        # the neighbouring one.
        measurement_core = (
            measurement_df.filter(
                pl.col("metric_id").is_in([self._WORK_METRIC, self._TIME_METRIC])
                & (pl.col("end_ts") > start_time)
                & (pl.col("start_ts") < end_time)
                & pl.col("value").is_not_null()
                & (pl.col("value").abs() < 1e307)
            )
            .join(core_mapping, left_on="physical_entity_id", right_on="core_id", how="inner")
            .select(
                [
                    "logical_entity_id",
                    "metric_id",
                    "package_id",
                    pl.col("start_ts").alias("meas_start"),
                    pl.col("end_ts").alias("meas_end"),
                    pl.col("value").alias("meas_value"),
                ]
            )
        )

        # Work and time must be split the same way, otherwise the numerator and the
        # denominator of the later share no longer describe the same set of measurements.
        work_measurements = self._split_over_intervals(
            measurement_core.filter(pl.col("metric_id") == self._WORK_METRIC),
            energy_intervals,
            "meas_value",
        ).select(
            [
                "interval_start",
                "interval_end",
                "logical_entity_id",
                "package_id",
                pl.col("meas_value").alias("work_value"),
            ]
        )

        if work_measurements.is_empty():
            return self._empty_result()

        time_measurements = self._split_over_intervals(
            measurement_core.filter(pl.col("metric_id") == self._TIME_METRIC),
            energy_intervals,
            "meas_value",
        ).select(
            [
                "interval_start",
                "interval_end",
                "logical_entity_id",
                "package_id",
                pl.col("meas_value").alias("exec_time_ms"),
            ]
        )

        work_per_thread = work_measurements.group_by(
            ["interval_start", "interval_end", "logical_entity_id", "package_id"]
        ).agg(pl.col("work_value").sum().alias("work"))

        work_per_socket = work_per_thread.group_by(["interval_start", "interval_end", "package_id"]).agg(
            pl.col("work").sum().alias("socket_work")
        )

        time_per_thread = (
            time_measurements.with_columns((pl.col("exec_time_ms") / 1000.0).alias("exec_time_s"))
            .group_by(["interval_start", "interval_end", "logical_entity_id", "package_id"])
            .agg(pl.col("exec_time_s").sum())
        )

        time_per_socket = time_per_thread.group_by(["interval_start", "interval_end", "package_id"]).agg(
            pl.col("exec_time_s").sum().alias("socket_time_s")
        )

        cpu_idle = cpu_df.select([pl.col("id").alias("package_id"), "idle_power_package"])

        activity = (
            work_per_thread.join(
                energy_intervals,
                on=["package_id", "interval_start", "interval_end"],
                how="inner",
            )
            .join(
                time_per_thread,
                on=["package_id", "interval_start", "interval_end", "logical_entity_id"],
            )
            .join(cpu_idle, on="package_id")
            .with_columns(
                [
                    pl.col("exec_time_s").fill_null(0.0),
                    pl.col("idle_power_package").fill_null(0.0),
                    (
                        (
                            pl.col("interval_end").dt.timestamp(time_unit="ns")
                            - pl.col("interval_start").dt.timestamp(time_unit="ns")
                        )
                        / 1_000_000_000.0
                    ).alias("interval_seconds"),
                ]
            )
            .with_columns((pl.col("idle_power_package") * pl.col("interval_seconds")).alias("idle_energy_j"))
        )

        attribution = (
            activity.join(work_per_socket, on=["package_id", "interval_start", "interval_end"])
            .join(time_per_socket, on=["package_id", "interval_start", "interval_end"])
            .with_columns(
                [
                    pl.col("socket_work").fill_null(0.0),
                    pl.col("socket_time_s").fill_null(0.0),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col("socket_work") <= 0)
                    .then(0.0)
                    .otherwise(
                        (pl.col("work") / pl.col("socket_work")) * (pl.col("energy_j") - pl.col("idle_energy_j"))
                    )
                    .alias("active_energy_j"),
                    pl.when(pl.col("socket_time_s") <= 0)
                    .then(0.0)
                    .otherwise((pl.col("exec_time_s") / pl.col("socket_time_s")) * pl.col("idle_energy_j"))
                    .alias("idle_energy_share_j"),
                ]
            )
        )

        total_energy_per_socket = (
            attribution.group_by(["interval_start", "interval_end", "logical_entity_id", "package_id"])
            .agg(
                [
                    pl.col("active_energy_j").sum(),
                    pl.col("idle_energy_share_j").sum().alias("idle_energy_j"),
                ]
            )
            .with_columns((pl.col("active_energy_j") + pl.col("idle_energy_j")).alias("total_energy_j"))
            .select(
                [
                    "interval_start",
                    "interval_end",
                    "logical_entity_id",
                    "package_id",
                    "total_energy_j",
                    "active_energy_j",
                    "idle_energy_j",
                ]
            )
            .sort(["interval_start", "logical_entity_id"])
        )

        total_energy_df = total_energy_per_socket.select(
            pl.lit(CPU_PKG_PID_TOTAL_ENERGY.name).alias("metric_id"),
            pl.col("package_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("total_energy_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        idle_energy_df = total_energy_per_socket.select(
            pl.lit(CPU_PKG_PID_IDLE_ENERGY.name).alias("metric_id"),
            pl.col("package_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("idle_energy_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        dynamic_energy_df = total_energy_per_socket.select(
            pl.lit(CPU_PKG_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
            pl.col("package_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("active_energy_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        res = (total_energy_df, idle_energy_df, dynamic_energy_df)

        return res


class DRAMEnergyAttributionModel(EnergyAttributionModelBase):
    """DRAM energy attribution using RAPL counters and memory access patterns.

    Distributes DRAM energy to processes based on memory read patterns
    (L3 cache misses to local and remote DRAM). Idle energy is split
    equally among processes with memory presence.

    Attributes
    ----------
    _DRAM_METRIC : str
        Metric ID for DRAM energy counter.
    _MEMORY_METRIC : str
        Metric ID for per-process memory presence.
    _READ_LOCAL_METRIC, _READ_REMOTE_METRIC : str
        Metric IDs for per-process L3 misses to local and remote DRAM.
    _ENERGY_WRAP_DRAM : float
        Energy counter wrap-around value in Joules.
    """

    _DRAM_METRIC = "DRAM_SOCKET_TOTAL_ENERGY"
    _MEMORY_METRIC = "DRAM_SOCKET_PID_MEMORY"
    _READ_LOCAL_METRIC = "CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM"
    _READ_REMOTE_METRIC = "CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM"
    _ENERGY_WRAP_DRAM = 65_712_999_613.0 / 1_000_000.0

    def __init__(self) -> None:
        pass

    def attribute_energy(
        self,
        measurement_df: pl.DataFrame,
        core_df: pl.DataFrame | None = None,
        dram_df: pl.DataFrame | None = None,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute DRAM energy to processes.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements with DRAM_SOCKET_TOTAL_ENERGY and memory metrics.
        core_df : pl.DataFrame, optional
            Core entity data for CPU socket mapping.
        dram_df : pl.DataFrame, optional
            DRAM entity data with idle_power.
        start_time : datetime, optional
            Filter start time.
        end_time : datetime, optional
            Filter end time.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) per process.
        """
        if core_df is None or dram_df is None:
            return self._empty_result()
        return self._attribute_dram_energy(measurement_df, core_df, dram_df, start_time, end_time)

    def _attribute_dram_energy(
        self,
        measurement_df: pl.DataFrame,
        core_df: pl.DataFrame,
        dram_df: pl.DataFrame,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        if measurement_df.is_empty():
            return self._empty_result()

        start_time, end_time = self._determine_start_end_times(measurement_df, start_time=start_time, end_time=end_time)

        # use lazy frames to optimise for runtime
        measurement_lf: pl.LazyFrame = measurement_df.lazy()
        dram_lf: pl.LazyFrame = dram_df.lazy()
        core_lf: pl.LazyFrame = core_df.lazy()

        energy_samples = (
            measurement_lf.filter(
                (pl.col("metric_id") == self._DRAM_METRIC)
                & (pl.col("start_ts") >= start_time)
                & (pl.col("start_ts") <= end_time)
                & (pl.col("value").is_not_null())
                & (pl.col("value").abs() < 1e307)
            )
            .join(dram_lf, left_on="physical_entity_id", right_on="id", how="inner")
            .select(
                pl.col("value"),
                pl.col("metric_id"),
                pl.col("physical_entity_id").alias("dram_id"),
                pl.col("parent_id").alias("cpu_id"),
                pl.col("logical_entity_id"),
                pl.col("start_ts"),
                pl.col("end_ts"),
            )
        )

        energy_intervals = (
            energy_samples.sort(["dram_id", "start_ts"])
            .with_columns(
                pl.col("start_ts").alias("interval_start"),
                pl.col("start_ts").shift(-1).over(partition_by="dram_id", order_by="start_ts").alias("interval_end"),
                pl.col("value").shift(-1).over(partition_by="dram_id", order_by="start_ts").alias("lead_value"),
            )
            .with_columns(
                pl.when(pl.col("lead_value").is_null())
                .then(0.0)
                .when(pl.col("lead_value") >= pl.col("value"))
                .then(pl.col("lead_value") - pl.col("value"))
                .otherwise(pl.lit(self._ENERGY_WRAP_DRAM) - pl.col("value") + pl.col("lead_value"))
                .alias("energy_j")
            )
            .filter(pl.col("interval_end").is_not_null())
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "energy_j"])
            .sort(["dram_id", "interval_start"])
        )

        dram_interval_powers = energy_intervals.join(dram_lf, left_on="dram_id", right_on="id").select(
            [
                "dram_id",
                "cpu_id",
                "interval_start",
                "interval_end",
                "energy_j",
                (
                    (
                        pl.col("interval_end").dt.timestamp(time_unit="ns")
                        - pl.col("interval_start").dt.timestamp(time_unit="ns")
                    )
                    * 1e-9
                ).alias("interval_duration_s"),
                "idle_power",
            ]
        )

        dram_interval_energy_split = dram_interval_powers.select(
            [
                "dram_id",
                "cpu_id",
                "interval_start",
                "interval_end",
                "energy_j",
                "idle_power",
                "interval_duration_s",
                (pl.col("idle_power") * pl.col("interval_duration_s")).alias("idle_energy_j"),
                pl.max_horizontal(
                    pl.lit(0.0),
                    pl.col("energy_j") - (pl.col("idle_power") * pl.col("interval_duration_s")),
                ).alias("active_energy_j"),
            ]
        )

        dram_memory_samples = measurement_lf.filter(
            (pl.col("metric_id") == self._MEMORY_METRIC) & pl.col("value").is_not_null()
        ).select(
            "logical_entity_id",
            "physical_entity_id",
            pl.col("start_ts").alias("meas_start"),
            pl.col("end_ts").alias("meas_end"),
            "value",
        )

        dram_memory_usage_by_thread = (
            self._join_overlapping(
                dram_memory_samples,
                energy_intervals,
                [pl.col("physical_entity_id") == pl.col("dram_id")],
                value_col="value",
            )
            .group_by(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id"])
            .agg(pl.col("value").sum().alias("total_memory_usage_by_thread"))
            .select(
                [
                    "dram_id",
                    "cpu_id",
                    "interval_start",
                    "interval_end",
                    "logical_entity_id",
                    "total_memory_usage_by_thread",
                ]
            )
        )

        dram_pids_present_on_dram = (
            dram_memory_usage_by_thread.filter(pl.col("total_memory_usage_by_thread") > 0)
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id"])
            .unique()
        )

        active_pids_on_dram = (
            dram_pids_present_on_dram.group_by(["dram_id", "cpu_id", "interval_start", "interval_end"])
            .agg(pl.count("logical_entity_id").alias("pid_count"))
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "pid_count"])
        )

        # core_mapping = core_lf.select(["id", "parent_id"]).rename({"id": "core_id", "parent_id": "cpu_id"})

        measurement_core = measurement_lf.join(
            core_lf,
            left_on="physical_entity_id",
            right_on="id",
        ).select(
            [
                "logical_entity_id",
                "metric_id",
                "parent_id",  # this is the cpu_id
                "start_ts",
                "end_ts",
                "value",
            ]
        )

        measurement_core_remote = measurement_core.rename({"parent_id": "source_cpu_id"})

        local_read_samples = measurement_core.filter(
            (pl.col("metric_id") == self._READ_LOCAL_METRIC) & pl.col("value").is_not_null()
        ).select(
            "logical_entity_id",
            "parent_id",
            pl.col("start_ts").alias("meas_start"),
            pl.col("end_ts").alias("meas_end"),
            "value",
        )

        dram_local_reads_by_thread = (
            self._join_overlapping(
                local_read_samples,
                energy_intervals,
                [pl.col("parent_id") == pl.col("cpu_id")],
                value_col="value",
            )
            .group_by(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id"])
            .agg(pl.col("value").sum().alias("reads"))
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id", "reads"])
        )

        remote_read_samples = measurement_core_remote.filter(pl.col("metric_id") == self._READ_REMOTE_METRIC).select(
            "logical_entity_id",
            "source_cpu_id",
            pl.col("start_ts").alias("meas_start"),
            pl.col("end_ts").alias("meas_end"),
            "value",
        )

        dram_remote_reads_by_thread = (
            self._join_overlapping(
                remote_read_samples,
                energy_intervals,
                [pl.col("source_cpu_id") != pl.col("cpu_id")],
                value_col="value",
            )
            .group_by(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id"])
            .agg(pl.col("value").sum().alias("reads"))
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id", "reads"])
        )

        dram_all_reads = pl.concat([dram_local_reads_by_thread, dram_remote_reads_by_thread])

        dram_reads_by_thread = (
            dram_all_reads.group_by(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id"]).agg(
                pl.col("reads").sum().alias("pid_total_reads")
            )
        ).select(["dram_id", "cpu_id", "interval_start", "interval_end", "logical_entity_id", "pid_total_reads"])

        dram_reads_total_on_dram = (
            dram_reads_by_thread.group_by(["dram_id", "cpu_id", "interval_start", "interval_end"])
            .agg(pl.col("pid_total_reads").sum().alias("total_reads_on_dram"))
            .select(["dram_id", "cpu_id", "interval_start", "interval_end", "total_reads_on_dram"])
        )

        dram_read_share_by_thread = (
            dram_reads_by_thread.join(
                dram_reads_total_on_dram,
                on=["dram_id", "cpu_id", "interval_start", "interval_end"],
            )
            .with_columns(
                (
                    pl.when(pl.col("total_reads_on_dram") <= 0)
                    .then(0.0)
                    .otherwise(pl.col("pid_total_reads") / pl.col("total_reads_on_dram"))
                ).alias("pid_read_share")
            )
            .select(
                [
                    "dram_id",
                    "cpu_id",
                    "interval_start",
                    "interval_end",
                    "logical_entity_id",
                    "pid_read_share",
                ]
            )
        )
        dram_active_energy_per_thread_per_dram = (
            dram_interval_energy_split.join(
                dram_read_share_by_thread,
                on=["dram_id", "cpu_id", "interval_start", "interval_end"],
                how="left",
            )
            .with_columns(
                [
                    (pl.col("pid_read_share").fill_null(0.0) * pl.col("active_energy_j")).alias("active_energy_j"),
                ]
            )
            .select(
                [
                    pl.lit(DRAM_SOCKET_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
                    pl.col("dram_id").alias("physical_entity_id"),
                    "logical_entity_id",
                    pl.col("active_energy_j").alias("value"),
                    pl.col("interval_start").alias("start_ts"),
                    pl.col("interval_end").alias("end_ts"),
                ]
            )
        )

        dram_idle_energy_per_thread_per_dram = (
            dram_interval_energy_split.join(
                active_pids_on_dram, on=["dram_id", "cpu_id", "interval_start", "interval_end"]
            )
            .join(dram_pids_present_on_dram, on=["dram_id", "cpu_id", "interval_start", "interval_end"])
            .with_columns((pl.col("idle_energy_j") / pl.col("pid_count")).alias("idle_energy_j"))
            .select(
                [
                    pl.lit(DRAM_SOCKET_PID_IDLE_ENERGY.name).alias("metric_id"),
                    pl.col("dram_id").alias("physical_entity_id"),
                    "logical_entity_id",
                    pl.col("idle_energy_j").alias("value"),
                    pl.col("interval_start").alias("start_ts"),
                    pl.col("interval_end").alias("end_ts"),
                ]
            )
        )

        total_from_active = dram_active_energy_per_thread_per_dram.select(
            [
                "logical_entity_id",
                "physical_entity_id",
                "start_ts",
                "end_ts",
                pl.col("value").alias("active_value"),
            ]
        )

        total_from_idle = dram_idle_energy_per_thread_per_dram.select(
            [
                "logical_entity_id",
                "physical_entity_id",
                "start_ts",
                "end_ts",
                pl.col("value").alias("idle_value"),
            ]
        )

        total_dram_energy_per_thread_per_dram = (
            total_from_active.join(
                total_from_idle,
                on=["logical_entity_id", "physical_entity_id", "start_ts", "end_ts"],
                how="full",
                suffix="_r",
            )
            .select(
                [
                    pl.coalesce([pl.col("logical_entity_id"), pl.col("logical_entity_id_r")]).alias(
                        "logical_entity_id"
                    ),
                    pl.coalesce([pl.col("physical_entity_id"), pl.col("physical_entity_id_r")]).alias(
                        "physical_entity_id"
                    ),
                    pl.coalesce([pl.col("start_ts"), pl.col("start_ts_r")]).alias("start_ts"),
                    pl.coalesce([pl.col("end_ts"), pl.col("end_ts_r")]).alias("end_ts"),
                    pl.col("active_value").fill_null(0.0).alias("active_energy_j"),
                    pl.col("idle_value").fill_null(0.0).alias("idle_energy_j"),
                ]
            )
            .with_columns((pl.col("active_energy_j") + pl.col("idle_energy_j")).alias("value"))
            .select(
                [
                    pl.lit(DRAM_SOCKET_PID_TOTAL_ENERGY.name).alias("metric_id"),
                    "physical_entity_id",
                    "logical_entity_id",
                    pl.col("value"),
                    pl.col("start_ts"),
                    pl.col("end_ts"),
                ]
            )
        )

        return (
            total_dram_energy_per_thread_per_dram.collect(),
            dram_idle_energy_per_thread_per_dram.collect(),
            dram_active_energy_per_thread_per_dram.collect(),
        )


class GPUEnergyAttributionModel(EnergyAttributionModelBase):
    """GPU energy attribution using energy counter and SM utilization.

    Attributes energy to processes based on their SM (Streaming Multiprocessor)
    utilization ratio. Uses the NVML energy counter as ground truth.

    Attribution Formula
    -------------------
    For each interval [t₀, t₁]:
    - E_total = GPU energy counter delta
    - E_idle = idle_power × interval_duration
    - E_dynamic = E_total - E_idle
    - PID_dynamic = E_dynamic × (U_sm[pid] / Σ U_sm[all])
    - PID_idle = E_idle × (time[pid] / Σ time[all])
    - PID_total = PID_dynamic + PID_idle
    """

    _ENERGY_METRIC = "GPU_DEVICE_TOTAL_ENERGY"
    _UTILIZATION_METRIC = "GPU_DEVICE_PID_COMPUTE_UTILIZATION"
    # Energy counter wrap-around (64-bit counter in mJ, converted to J)
    _ENERGY_WRAP_GPU = (2**64 - 1) / 1000.0

    def __init__(self) -> None:
        pass

    def _calculate_energy_intervals(
        self,
        measurement_df: pl.DataFrame,
        gpu_df: pl.DataFrame,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
    ) -> pl.DataFrame:
        """Calculate energy deltas between consecutive measurements.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements containing GPU_DEVICE_TOTAL_ENERGY samples.
        gpu_df : pl.DataFrame
            GPU entity data with idle_power field.
        start_time : datetime, optional
            Filter start time.
        end_time : datetime, optional
            Filter end time.

        Returns
        -------
        pl.DataFrame
            Energy intervals with columns: gpu_id, interval_start, interval_end,
            energy_j, idle_energy_j, dynamic_energy_j, interval_seconds.
        """
        if measurement_df.is_empty():
            return pl.DataFrame(
                schema={
                    "gpu_id": pl.String,
                    "interval_start": pl.Datetime,
                    "interval_end": pl.Datetime,
                    "energy_j": pl.Float64,
                    "idle_energy_j": pl.Float64,
                    "dynamic_energy_j": pl.Float64,
                    "interval_seconds": pl.Float64,
                }
            )

        energy_samples = measurement_df.filter(
            (pl.col("metric_id") == self._ENERGY_METRIC)
            & (pl.col("start_ts") >= start_time if start_time else True)
            & (pl.col("start_ts") <= end_time if end_time else True)
        ).select(
            pl.col("value"),
            pl.col("physical_entity_id").alias("gpu_id"),
            pl.col("start_ts"),
        )

        if energy_samples.is_empty():
            return pl.DataFrame(
                schema={
                    "gpu_id": pl.String,
                    "interval_start": pl.Datetime,
                    "interval_end": pl.Datetime,
                    "energy_j": pl.Float64,
                    "idle_energy_j": pl.Float64,
                    "dynamic_energy_j": pl.Float64,
                    "interval_seconds": pl.Float64,
                }
            )

        # Calculate energy deltas with wrap-around handling
        energy_intervals = (
            energy_samples.sort(["gpu_id", "start_ts"])
            .with_columns(
                pl.col("start_ts").alias("interval_start"),
                pl.col("start_ts").shift(-1).over(partition_by="gpu_id", order_by="start_ts").alias("interval_end"),
                pl.col("value").shift(-1).over(partition_by="gpu_id", order_by="start_ts").alias("lead_value"),
            )
            .with_columns(
                pl.when(pl.col("lead_value").is_null())
                .then(0.0)
                .when(pl.col("lead_value") >= pl.col("value"))
                .then(pl.col("lead_value") - pl.col("value"))
                .otherwise(pl.lit(self._ENERGY_WRAP_GPU) - pl.col("value") + pl.col("lead_value"))
                .alias("energy_j")
            )
            .filter(pl.col("interval_end").is_not_null())
        )

        # Join with GPU idle power and calculate idle/dynamic energy
        gpu_idle = gpu_df.select(pl.col("id").alias("gpu_id"), pl.col("idle_power").alias("idle_power_w"))

        energy_intervals = (
            energy_intervals.join(gpu_idle, on="gpu_id", how="left")
            .with_columns(
                pl.col("idle_power_w").fill_null(0.0),
                (
                    (
                        pl.col("interval_end").dt.timestamp(time_unit="ns")
                        - pl.col("interval_start").dt.timestamp(time_unit="ns")
                    )
                    / 1_000_000_000.0
                ).alias("interval_seconds"),
            )
            .with_columns((pl.col("idle_power_w") * pl.col("interval_seconds")).alias("idle_energy_j"))
            .with_columns(
                pl.max_horizontal(pl.col("energy_j") - pl.col("idle_energy_j"), pl.lit(0.0)).alias("dynamic_energy_j")
            )
            .select(
                [
                    "gpu_id",
                    "interval_start",
                    "interval_end",
                    "energy_j",
                    "idle_energy_j",
                    "dynamic_energy_j",
                    "interval_seconds",
                ]
            )
        )

        return energy_intervals

    def _get_process_utilization(
        self,
        measurement_df: pl.DataFrame,
        energy_intervals: pl.DataFrame,
    ) -> pl.DataFrame:
        """Align process utilization samples with energy intervals.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements containing GPU_DEVICE_PID_COMPUTE_UTILIZATION.
        energy_intervals : pl.DataFrame
            Energy intervals from _calculate_energy_intervals.

        Returns
        -------
        pl.DataFrame
            Utilization aligned to intervals with columns: gpu_id, interval_start,
            interval_end, logical_entity_id, sm_util.
        """
        if measurement_df.is_empty() or energy_intervals.is_empty():
            return pl.DataFrame(
                schema={
                    "gpu_id": pl.String,
                    "interval_start": pl.Datetime,
                    "interval_end": pl.Datetime,
                    "logical_entity_id": pl.String,
                    "sm_util": pl.Float64,
                }
            )

        util_samples = measurement_df.filter(pl.col("metric_id") == self._UTILIZATION_METRIC).select(
            pl.col("physical_entity_id").alias("gpu_id"),
            pl.col("logical_entity_id"),
            pl.col("start_ts").alias("util_start"),
            pl.col("end_ts").alias("util_end"),
            pl.col("value").alias("sm_util"),
        )

        if util_samples.is_empty():
            return pl.DataFrame(
                schema={
                    "gpu_id": pl.String,
                    "interval_start": pl.Datetime,
                    "interval_end": pl.Datetime,
                    "logical_entity_id": pl.String,
                    "sm_util": pl.Float64,
                }
            )

        # Utilization is a rate, so it is not split: a sample counts in full in every
        # interval it overlaps.
        aligned = self._join_overlapping(
            util_samples,
            energy_intervals,
            [pl.col("gpu_id") == pl.col("gpu_id_right")],
            meas_start="util_start",
            meas_end="util_end",
        ).select(
            pl.col("gpu_id"),
            pl.col("interval_start"),
            pl.col("interval_end"),
            pl.col("logical_entity_id"),
            pl.col("sm_util"),
        )

        return aligned

    def _attribute_by_utilization(
        self,
        energy_intervals: pl.DataFrame,
        process_utilization: pl.DataFrame,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute energy to processes based on SM utilization ratio.

        Dynamic energy: proportional to SM utilization ratio.
        Idle energy: time-weighted by process presence duration in interval.

        Parameters
        ----------
        energy_intervals : pl.DataFrame
            Energy intervals with idle/dynamic energy.
        process_utilization : pl.DataFrame
            Process utilization aligned to intervals.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) DataFrames.
        """
        if energy_intervals.is_empty() or process_utilization.is_empty():
            return self._empty_result()

        # Aggregate utilization per process per interval with sample count
        util_per_thread = process_utilization.group_by(
            ["gpu_id", "interval_start", "interval_end", "logical_entity_id"]
        ).agg(
            pl.col("sm_util").mean().alias("avg_sm_util"),
            pl.len().alias("sample_count"),  # proxy for time presence
        )

        # Total utilization and sample counts per interval
        util_per_interval = util_per_thread.group_by(["gpu_id", "interval_start", "interval_end"]).agg(
            pl.col("avg_sm_util").sum().alias("total_sm_util"),
            pl.col("sample_count").sum().alias("total_samples"),
        )

        # Join and calculate attribution
        attribution = (
            util_per_thread.join(
                energy_intervals,
                on=["gpu_id", "interval_start", "interval_end"],
                how="inner",
            )
            .join(
                util_per_interval,
                on=["gpu_id", "interval_start", "interval_end"],
                how="inner",
            )
            .with_columns(
                # Dynamic energy: proportional to SM utilization ratio
                pl.when(pl.col("total_sm_util") <= 0)
                .then(0.0)
                .otherwise((pl.col("avg_sm_util") / pl.col("total_sm_util")) * pl.col("dynamic_energy_j"))
                .alias("pid_dynamic_j"),
                # Idle energy: time-weighted by sample count (presence duration)
                pl.when(pl.col("total_samples") <= 0)
                .then(0.0)
                .otherwise((pl.col("sample_count") / pl.col("total_samples")) * pl.col("idle_energy_j"))
                .alias("pid_idle_j"),
            )
            .with_columns((pl.col("pid_dynamic_j") + pl.col("pid_idle_j")).alias("pid_total_j"))
        )

        # Format output DataFrames
        total_df = attribution.select(
            pl.lit(GPU_DEVICE_PID_TOTAL_ENERGY.name).alias("metric_id"),
            pl.col("gpu_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("pid_total_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        idle_df = attribution.select(
            pl.lit(GPU_DEVICE_PID_IDLE_ENERGY.name).alias("metric_id"),
            pl.col("gpu_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("pid_idle_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        dynamic_df = attribution.select(
            pl.lit(GPU_DEVICE_PID_DYNAMIC_ENERGY.name).alias("metric_id"),
            pl.col("gpu_id").alias("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("pid_dynamic_j").alias("value"),
            pl.col("interval_start").alias("start_ts"),
            pl.col("interval_end").alias("end_ts"),
        )

        return (total_df, idle_df, dynamic_df)

    def attribute_gpu_energy(
        self,
        measurement_df: pl.DataFrame,
        gpu_df: pl.DataFrame,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute GPU energy to processes using SM utilization.

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements with GPU_DEVICE_TOTAL_ENERGY and
            GPU_DEVICE_PID_COMPUTE_UTILIZATION.
        gpu_df : pl.DataFrame
            GPU entity data with id and idle_power columns.
        start_time : datetime, optional
            Filter measurements from this time.
        end_time : datetime, optional
            Filter measurements until this time.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) per process.

        Examples
        --------
        >>> model = GPUEnergyAttributionModel()
        >>> total, idle, dynamic = model.attribute_gpu_energy(
        ...     measurement_df=measurements,
        ...     gpu_df=gpu_entities,
        ... )
        >>> total.select(["logical_entity_id", "value"]).head()
        shape: (2, 2)
        ┌───────────────────┬───────────┐
        │ logical_entity_id ┆ value     │
        │ ---               ┆ ---       │
        │ str               ┆ f64       │
        ╞═══════════════════╪═══════════╡
        │ pid-1234          ┆ 45.2      │
        │ pid-5678          ┆ 12.8      │
        └───────────────────┴───────────┘
        """
        if measurement_df.is_empty() or gpu_df.is_empty():
            return self._empty_result()

        # Determine time bounds (inherited from base class)
        start_time, end_time = self._determine_start_end_times(measurement_df, start_time=start_time, end_time=end_time)

        # Step 1: Calculate energy intervals
        energy_intervals = self._calculate_energy_intervals(measurement_df, gpu_df, start_time, end_time)

        if energy_intervals.is_empty():
            return self._empty_result()

        # Step 2: Align process utilization
        process_utilization = self._get_process_utilization(measurement_df, energy_intervals)

        # Step 3: Attribute by utilization
        return self._attribute_by_utilization(energy_intervals, process_utilization)

    def attribute_energy(
        self,
        measurement_df: pl.DataFrame,
        gpu_df: pl.DataFrame | None = None,
        start_time: dt.datetime | None = None,
        end_time: dt.datetime | None = None,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Attribute GPU energy to processes (abstract method implementation).

        Parameters
        ----------
        measurement_df : pl.DataFrame
            Measurements with GPU energy and utilization metrics.
        gpu_df : pl.DataFrame, optional
            GPU entity data with id and idle_power columns.
        start_time : datetime, optional
            Filter start time.
        end_time : datetime, optional
            Filter end time.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            (total_energy, idle_energy, dynamic_energy) per process.
        """
        if gpu_df is None:
            return self._empty_result()
        return self.attribute_gpu_energy(measurement_df, gpu_df, start_time, end_time)


class DyncamicEnergyAttributionModel:
    """Dynamic energy attribution with disk spill and cross-batch continuity.

    Orchestrates CPU, DRAM, and GPU energy attribution with support for
    late-arriving measurements (via disk-based buffer) and ensures no
    energy is lost between consecutive attribution batches.

    Each component (CPU, DRAM, GPU) is processed independently with its
    own measurement bound, allowing faster attribution when some components
    have data ready before others.

    Parameters
    ----------
    cutoff : float
        Seconds to wait before processing measurements.
    memory_limit_rows : int
        Max rows in memory before spilling to disk.
    spill_dir : Path, optional
        Directory for spill files.
    stall_after : float, optional
        Seconds a stream may lag behind the fastest stream of its component before it
        stops being required for the window bound. Defaults to ``max(30.0, 3 * cutoff)``.

    Examples
    --------
    >>> model = DyncamicEnergyAttributionModel(cutoff=30.0)
    >>> model.update_entites(entities)
    >>> model.update_measurements(measurements_df)
    >>> cpu, dram, gpu = model.attribute_energy()
    """

    # Energy metrics per component for cross-batch continuity
    _CPU_METRIC = CPUEnergyAttributionModel._PKG_METRIC
    _DRAM_METRIC = DRAMEnergyAttributionModel._DRAM_METRIC
    _GPU_METRIC = GPUEnergyAttributionModel._ENERGY_METRIC
    _COMPONENT_ENERGY: dict[str, str] = {"cpu": _CPU_METRIC, "dram": _DRAM_METRIC, "gpu": _GPU_METRIC}

    # Every stream a component reads, not just its energy counter. The energy stream is the
    # numerator of the attribution, the per-thread streams are the denominator. Without the
    # denominator the energy is split among whichever threads happen to have arrived already:
    # the total stays right, the assignment does not.
    _COMPONENT_INPUTS: dict[str, list[str]] = {
        "cpu": [
            CPUEnergyAttributionModel._PKG_METRIC,
            CPUEnergyAttributionModel._WORK_METRIC,
            CPUEnergyAttributionModel._TIME_METRIC,
        ],
        "dram": [
            DRAMEnergyAttributionModel._DRAM_METRIC,
            DRAMEnergyAttributionModel._MEMORY_METRIC,
            DRAMEnergyAttributionModel._READ_LOCAL_METRIC,
            DRAMEnergyAttributionModel._READ_REMOTE_METRIC,
        ],
        "gpu": [
            GPUEnergyAttributionModel._ENERGY_METRIC,
            GPUEnergyAttributionModel._UTILIZATION_METRIC,
        ],
    }

    def __init__(
        self,
        cutoff: float = 30.0,
        memory_limit_rows: int = 500_000,
        spill_dir: Path | None = None,
        stall_after: float | None = None,
    ) -> None:
        self.cutoff = cutoff
        self._buffer = MeasurementBuffer(memory_limit_rows=memory_limit_rows, spill_dir=spill_dir)
        self._core_df = pl.DataFrame()
        self._cpu_df = pl.DataFrame()
        self._dram_df = pl.DataFrame()
        self._gpu_df = pl.DataFrame()

        # "Arrived too late" only means something once a window has actually been closed.
        self._last_cutoff_check: dt.datetime | None = None
        # Cross-batch continuity: cache last energy sample per (metric, entity)
        self._last_energy_samples: dict[tuple[str, str], tuple[dt.datetime, float]] = {}

        self._watermarks: dict[str, dt.datetime] = {}  # metric_id -> newest end_ts seen
        self._reported_stalls: set[str] = set()
        # Lower bound so that cutoff=0.0 does not declare every stream stalled at once.
        self._stall_after = stall_after if stall_after is not None else max(30.0, 3 * cutoff)

    @property
    def _measurements_df(self) -> pl.DataFrame:
        """Get all measurements from buffer (compatibility property)."""
        return self._buffer.get_all()

    def update_measurements(self, measurement: pl.DataFrame | Sequence[Measurement]) -> None:
        """Add new measurements to the buffer.

        Parameters
        ----------
        measurement : pl.DataFrame | Sequence[Measurement]
            New measurements to add.
        """
        if isinstance(measurement, list):
            measurement_df = pl.DataFrame()
            for m in measurement:
                if isinstance(m, Measurement):
                    m = m.to_df()
                if isinstance(m, pl.DataFrame):
                    measurement_df = pl.concat([measurement_df, m], how="vertical")
        elif isinstance(measurement, pl.DataFrame):
            measurement_df = measurement
        else:
            raise TypeError("measurement must be a Polars DataFrame or a sequence of Measurement objects.")
        if measurement_df.is_empty():
            return

        measurement_df = naive_utc_timestamps(measurement_df)
        self._buffer.add(measurement_df)
        self._advance_watermarks(measurement_df)

    def _advance_watermarks(self, measurement_df: pl.DataFrame) -> None:
        """Raise the per-stream high-water mark to the newest end_ts in this batch.

        A stream that has never appeared stays out of the dictionary and is never
        required, so a host without a GPU or without PMU counters does not block CPU
        attribution. A stream that has appeared once is required from then on.
        """
        newest = measurement_df.group_by("metric_id").agg(pl.col("end_ts").max())
        for row in newest.iter_rows(named=True):
            known = self._watermarks.get(row["metric_id"])
            if known is None or row["end_ts"] > known:
                self._watermarks[row["metric_id"]] = row["end_ts"]

    def _component_bound(
        self,
        metrics: Sequence[str],
        cutoff_time: dt.datetime,
        force: bool = False,
    ) -> dt.datetime | None:
        """Time up to which every stream of one component is complete.

        The minimum of the known streams' high-water marks, capped by the cutoff time.
        A stream lagging more than stall_after seconds behind the fastest stream of the
        same component counts as dried up and is no longer required, otherwise a finished
        workload would hold up attribution forever. With force the maximum applies
        instead: the shutdown drain is meant to empty the buffer.
        """
        marks = {metric: self._watermarks[metric] for metric in metrics if metric in self._watermarks}
        if not marks:
            return None

        newest = max(marks.values())
        stalled_before = newest - dt.timedelta(seconds=self._stall_after)

        live = []
        for metric, mark in marks.items():
            if mark < stalled_before:
                if metric not in self._reported_stalls:
                    logger.warning(
                        f"Stream {metric} lags more than {self._stall_after:.0f}s behind the other "
                        "streams of its component; it is no longer required for the window bound"
                    )
                    self._reported_stalls.add(metric)
                continue
            live.append(mark)

        if not live:
            return None
        if force:
            return newest
        return min(min(live), cutoff_time)

    def update_entites(self, entities: Sequence[Entity]) -> None:
        """Add new entities to the internal buffer."""
        for entity in entities:
            if isinstance(entity, CPU):
                self.update_cpu_df([entity])
                self.update_core_df(entity.cores)
            if isinstance(entity, DRAM):
                self.update_dram_df([entity])
            if isinstance(entity, GPU):
                self.update_gpu_df([entity])

    def update_dram_df(self, dram: Sequence[Entity]) -> None:
        """Set the DRAM dataframe for attribution."""
        dram_df = pl.concat([entity._to_df() for entity in dram], how="vertical") if dram else pl.DataFrame()
        self._dram_df = pl.concat([self._dram_df, dram_df]).unique(maintain_order=True)

    def update_cpu_df(self, cpu: Sequence[Entity]) -> None:
        """Set the CPU dataframe for attribution."""
        cpu_df = pl.concat([entity._to_df() for entity in cpu], how="vertical") if cpu else pl.DataFrame()
        self._cpu_df = pl.concat([self._cpu_df, cpu_df]).unique(maintain_order=True)

    def update_core_df(self, core: Sequence[Entity]) -> None:
        """Set the Core dataframe for attribution."""
        core_df = pl.concat([entity._to_df() for entity in core], how="vertical") if core else pl.DataFrame()
        self._core_df = pl.concat([self._core_df, core_df]).unique(maintain_order=True)

    def update_gpu_df(self, gpus: Sequence[Entity]) -> None:
        """Add GPU entities to the internal buffer."""
        gpu_df = pl.concat([entity._to_df() for entity in gpus], how="vertical") if gpus else pl.DataFrame()
        self._gpu_df = pl.concat([self._gpu_df, gpu_df]).unique(maintain_order=True)

    def _get_relevant_measurements(self, force: bool = False) -> tuple[pl.DataFrame, Mapping[str, dt.datetime | None]]:
        """Get measurements ready for attribution with per-component bounds.

        Each energy component (CPU, DRAM, GPU) gets its own measurement bound
        covering all of its input streams, allowing independent processing.

        Parameters
        ----------
        force : bool
            If True, process all measurements regardless of cutoff.

        Returns
        -------
        tuple[pl.DataFrame, dict[str, datetime | None]]
            (measurements, bounds_dict) where bounds_dict maps "cpu", "dram" and "gpu"
            to the bound of that component.
        """
        now = dt.datetime.now(UTC).replace(tzinfo=None)
        measurements_df = self._buffer.get_all()
        empty_bounds: dict[str, dt.datetime | None] = dict.fromkeys(self._COMPONENT_INPUTS)

        if measurements_df.is_empty():
            return pl.DataFrame(), empty_bounds

        cutoff_time = now if force else now - dt.timedelta(seconds=self.cutoff)

        # Check for missed data
        missed_data_count = (
            0
            if self._last_cutoff_check is None
            else measurements_df.filter(pl.col("end_ts") < self._last_cutoff_check).height
        )
        if missed_data_count > 0:
            logger.warning("Live Energy Attribution will no longer be accurate due to missing data")
            logger.debug(
                f"Last cutoff check: {self._last_cutoff_check}, new cutoff time: {cutoff_time}"
                + f" number of missed data={missed_data_count} and"
                + f" oldest missed timestamp {measurements_df.get_column('end_ts').min()}",
            )

        # Compute a measurement bound for EACH component, over all of its input streams
        bounds_dict: dict[str, dt.datetime | None] = {
            name: self._component_bound(metrics, cutoff_time, force) for name, metrics in self._COMPONENT_INPUTS.items()
        }

        # Filter out None bounds
        valid_bounds = [b for b in bounds_dict.values() if b is not None]

        if not valid_bounds:
            self._last_cutoff_check = cutoff_time
            return pl.DataFrame(), empty_bounds

        # Find the minimum bound across all components (for buffer cleanup)
        min_bound = min(valid_bounds)
        self._last_cutoff_check = cutoff_time

        # Get all measurements up to the maximum bound (each model will filter further)
        max_bound = max(valid_bounds)
        relevant_measurements = measurements_df.filter(pl.col("end_ts") <= max_bound)

        # CROSS-BATCH CONTINUITY: Cache last energy sample per entity per metric
        for component, bound in bounds_dict.items():
            metric = self._COMPONENT_ENERGY[component]
            if bound is None:
                continue
            last_samples = (
                relevant_measurements.filter((pl.col("metric_id") == metric) & (pl.col("end_ts") <= bound))
                .sort("end_ts")
                .group_by("physical_entity_id")
                .last()
            )
            for row in last_samples.iter_rows(named=True):
                key = (metric, row["physical_entity_id"])
                self._last_energy_samples[key] = (row["end_ts"], row["value"])

        # Remove processed measurements from the buffer. Never delete past the oldest anchor
        # just cached: the next batch rebuilds its first interval from that timestamp, and the
        # per-thread measurements covering that interval have to survive with it. Only energy
        # samples are re-injected, not the streams that split them. Under force the component
        # bound runs ahead of the energy stream, and without this cap it would take them along.
        anchors = [
            ts for (metric, _), (ts, _) in self._last_energy_samples.items() if metric not in self._reported_stalls
        ]
        deletion_point = min(min_bound, *anchors) if anchors else min_bound
        self._buffer.remove_before(deletion_point)

        # Put back the anchors the deletion just took. remove_before keeps end_ts > cutoff, so a
        # newer anchor is still in the buffer; re-adding it would leave two rows at the same
        # timestamp and one interval of zero length.
        cached_rows = [
            {
                "metric_id": metric,
                "physical_entity_id": entity_id,
                "logical_entity_id": None,
                "value": value,
                "start_ts": ts,
                "end_ts": ts,
            }
            for (metric, entity_id), (ts, value) in self._last_energy_samples.items()
            if ts <= deletion_point
        ]
        if cached_rows:
            cached_df = pl.DataFrame(cached_rows)
            # Align to the buffer's schema. Real measurement frames carry columns the
            # synthetic rows do not have (`id`), and a vertical concat needs both sides
            # to match exactly.
            cached_df = cached_df.select(
                [
                    pl.col(name).cast(dtype) if name in cached_df.columns else pl.lit(None, dtype).alias(name)
                    for name, dtype in relevant_measurements.schema.items()
                ]
            )
            self._buffer.add(cached_df)

        return relevant_measurements, bounds_dict

    def attribute_energy(
        self, force: bool = False
    ) -> tuple[
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    ]:
        """Attribute CPU, DRAM, and GPU energy using internal buffers.

        Each component is processed independently with its own measurement
        bound, allowing faster attribution when some data arrives before others.

        Parameters
        ----------
        force : bool
            If True, process all measurements regardless of cutoff.

        Returns
        -------
        tuple
            (pkg_result, dram_result, gpu_result) where each is
            (total_energy, idle_energy, dynamic_energy).
        """
        empty_result = EnergyAttributionModelBase._empty_result()

        if self._buffer.is_empty():
            logger.debug("No measurements available for energy attribution.")
            return empty_result, empty_result, empty_result

        relevant_measurements, bounds_dict = self._get_relevant_measurements(force=force)

        if relevant_measurements.is_empty():
            return empty_result, empty_result, empty_result

        cpu_model = CPUEnergyAttributionModel()
        dram_model = DRAMEnergyAttributionModel()
        gpu_model = GPUEnergyAttributionModel()

        # CPU attribution (with CPU-specific bound)
        cpu_bound = bounds_dict.get("cpu")
        if cpu_bound is not None:
            cpu_measurements = relevant_measurements.filter(pl.col("end_ts") <= cpu_bound)
            pkg_result = cpu_model.attribute_energy(
                measurement_df=cpu_measurements,
                cpu_df=self._cpu_df,
                core_df=self._core_df,
            )
        else:
            pkg_result = empty_result

        # DRAM attribution (with DRAM-specific bound)
        dram_bound = bounds_dict.get("dram")
        if dram_bound is not None:
            dram_measurements = relevant_measurements.filter(pl.col("end_ts") <= dram_bound)
            dram_result = dram_model.attribute_energy(
                measurement_df=dram_measurements,
                core_df=self._core_df,
                dram_df=self._dram_df,
            )
        else:
            dram_result = empty_result

        # GPU attribution (with GPU-specific bound)
        gpu_bound = bounds_dict.get("gpu")
        if gpu_bound is not None and not self._gpu_df.is_empty():
            gpu_measurements = relevant_measurements.filter(pl.col("end_ts") <= gpu_bound)
            gpu_result = gpu_model.attribute_gpu_energy(
                measurement_df=gpu_measurements,
                gpu_df=self._gpu_df,
            )
        else:
            gpu_result = empty_result

        return pkg_result, dram_result, gpu_result

    def cleanup(self) -> None:
        """Clean up disk spill files."""
        self._buffer.cleanup()
