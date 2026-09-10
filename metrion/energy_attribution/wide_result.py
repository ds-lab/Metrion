from collections.abc import Sequence

import polars as pl

from metrion.metric import (
    CPU_PKG_PID_DYNAMIC_ENERGY,
    CPU_PKG_PID_IDLE_ENERGY,
    CPU_PKG_PID_TOTAL_ENERGY,
    DRAM_SOCKET_PID_DYNAMIC_ENERGY,
    DRAM_SOCKET_PID_IDLE_ENERGY,
    DRAM_SOCKET_PID_TOTAL_ENERGY,
    GPU_DEVICE_PID_DYNAMIC_ENERGY,
    GPU_DEVICE_PID_IDLE_ENERGY,
    GPU_DEVICE_PID_TOTAL_ENERGY,
)

METRIC_TO_COLUMN: dict[str, str] = {
    CPU_PKG_PID_TOTAL_ENERGY.name: "cpu_total_j",
    CPU_PKG_PID_IDLE_ENERGY.name: "cpu_idle_j",
    CPU_PKG_PID_DYNAMIC_ENERGY.name: "cpu_dynamic_j",
    DRAM_SOCKET_PID_TOTAL_ENERGY.name: "dram_total_j",
    DRAM_SOCKET_PID_IDLE_ENERGY.name: "dram_idle_j",
    DRAM_SOCKET_PID_DYNAMIC_ENERGY.name: "dram_dynamic_j",
    GPU_DEVICE_PID_TOTAL_ENERGY.name: "gpu_total_j",
    GPU_DEVICE_PID_IDLE_ENERGY.name: "gpu_idle_j",
    GPU_DEVICE_PID_DYNAMIC_ENERGY.name: "gpu_dynamic_j",
}

ENERGY_COLUMNS: list[str] = list(METRIC_TO_COLUMN.values())

_INDEX_COLUMNS = ["window_start", "window_end", "logical_entity_id"]


def to_wide(frames: Sequence[pl.DataFrame], window_seconds: float) -> pl.DataFrame:
    """Fold the long attribution results into one row per window and thread.

    Parameters
    ----------
    frames : Sequence[pl.DataFrame]
        Result frames in the measurement schema, as attribute_energy returns them.
    window_seconds : float
        Length of the windows. Their boundaries lie on an absolute grid, so separate
        calls produce the same windows and their results can be added up afterwards.
        Must be at least the slowest energy sampling interval: below that, windows
        without a sample of that component report 0.0 instead of nothing.

    Notes
    -----
    Each result row covers one energy interval, and those intervals differ per
    component and do not line up with the grid. A row is therefore spread over the
    windows it overlaps, proportional to the overlap, which assumes the energy is
    evenly distributed inside the interval. Components that produced nothing appear
    as 0.0, which is what a sum over the missing rows would give.
    """
    populated = [frame for frame in frames if not frame.is_empty()]
    if not populated:
        return pl.DataFrame(schema=_schema())

    long = pl.concat(populated, how="vertical_relaxed")
    wide = (
        _spread_over_windows(long, window_seconds)
        .with_columns(pl.col("metric_id").replace_strict(METRIC_TO_COLUMN).alias("column"))
        .group_by([*_INDEX_COLUMNS, "column"])
        .agg(pl.col("value").sum())
        .pivot(on="column", index=_INDEX_COLUMNS, values="value")
    )

    return (
        wide.with_columns([pl.lit(0.0).alias(column) for column in ENERGY_COLUMNS if column not in wide.columns])
        .with_columns(pl.col(ENERGY_COLUMNS).fill_null(0.0))
        .select([*_INDEX_COLUMNS, *ENERGY_COLUMNS])
        .sort(_INDEX_COLUMNS)
    )


def _spread_over_windows(long: pl.DataFrame, window_seconds: float) -> pl.DataFrame:
    """Split each row across the windows it overlaps, its value by the overlap share."""
    window_us = int(window_seconds * 1_000_000)

    return (
        long.with_columns(
            start=pl.col("start_ts").dt.timestamp("us"),
            # A row of zero length belongs to one window in full. Widening it by a
            # microsecond keeps the share below free of a special case.
            end=pl.max_horizontal(pl.col("end_ts").dt.timestamp("us"), pl.col("start_ts").dt.timestamp("us") + 1),
        )
        .with_columns(window=pl.int_ranges(pl.col("start") // window_us, (pl.col("end") - 1) // window_us + 1))
        .explode("window", empty_as_null=False)
        .with_columns(window_start=pl.col("window") * window_us)
        .with_columns(window_end=pl.col("window_start") + window_us)
        .with_columns(
            overlap=pl.min_horizontal("end", "window_end") - pl.max_horizontal("start", "window_start"),
        )
        .with_columns(
            value=pl.col("value") * pl.col("overlap") / (pl.col("end") - pl.col("start")),
            window_start=pl.col("window_start").cast(pl.Datetime("us")),
            window_end=pl.col("window_end").cast(pl.Datetime("us")),
        )
    )


def _schema() -> dict[str, pl.DataType]:
    index = {"window_start": pl.Datetime("us"), "window_end": pl.Datetime("us"), "logical_entity_id": pl.String}
    return index | {column: pl.Float64 for column in ENERGY_COLUMNS}
