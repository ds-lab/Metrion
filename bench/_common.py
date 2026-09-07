"""Shared helpers for the METRION performance benchmarks.

The benchmarks are deliberately dependency-light (polars + duckdb + sqlalchemy)
so they can be executed on a bare Linux box without root privileges and without
BCC/eBPF being available.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import polars as pl

MEASUREMENT_METRICS = (
    "CPU_CORE_PID_WORK",
    "CPU_CORE_PID_TIME",
    "CPU_SOCKET_PKG_ENERGY",
    "DRAM_SOCKET_TOTAL_ENERGY",
)


@dataclass
class Result:
    name: str
    rows: int
    samples: list[float] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def rows_per_second(self) -> float:
        return self.rows / self.best if self.best > 0 else float("inf")


class Bench:
    """Collects results and prints a comparison table."""

    def __init__(self, title: str, rows: int, repeats: int = 3) -> None:
        self.title = title
        self.rows = rows
        self.repeats = repeats
        self.results: list[Result] = []

    def run(self, name: str, fn: Callable[[], object], *, setup: Callable[[], None] | None = None) -> Result:
        result = Result(name=name, rows=self.rows)
        for _ in range(self.repeats):
            if setup is not None:
                setup()
            start = time.perf_counter()
            extra = fn()
            result.samples.append(time.perf_counter() - start)
            if isinstance(extra, dict):
                result.extra.update(extra)
        self.results.append(result)
        print(f"  {name:<44} {result.best * 1000:9.1f} ms  {result.rows_per_second / 1e6:7.3f} M rows/s")
        return result

    def report(self) -> None:
        if not self.results:
            return
        baseline = self.results[0]
        print()
        print(f"=== {self.title} ({self.rows:,} rows, best of {self.repeats}) ===")
        header = f"{'variant':<44} {'best':>10} {'median':>10} {'rows/s':>12} {'vs baseline':>12}"
        print(header)
        print("-" * len(header))
        for r in self.results:
            speedup = baseline.best / r.best if r.best > 0 else float("inf")
            print(
                f"{r.name:<44} {r.best * 1000:9.1f}ms {r.median * 1000:9.1f}ms "
                f"{r.rows_per_second:12,.0f} {speedup:11.2f}x"
            )
            for key, value in r.extra.items():
                print(f"{'':<44}   {key}: {value}")
        print()


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    print(f"{label}: {(time.perf_counter() - start) * 1000:.1f} ms")


def make_measurement_df(rows: int, *, n_logical: int = 4000, n_physical: int = 128, seed: int = 7) -> pl.DataFrame:
    """Build a DataFrame with the exact shape METRION writes to the measurement table."""
    import numpy as np

    rng = np.random.default_rng(seed)
    base_ns = 1_760_000_000_000_000_000  # arbitrary fixed epoch in ns
    start_ns = base_ns + np.sort(rng.integers(0, 60_000_000_000, size=rows))
    end_ns = start_ns + rng.integers(1_000, 5_000_000, size=rows)

    return pl.DataFrame(
        {
            "id": np.arange(rows, dtype=np.int64) + 1_000_000_000,
            "value": rng.random(rows) * 1e6,
            "metric_id": pl.Series(
                rng.integers(0, len(MEASUREMENT_METRICS), size=rows), dtype=pl.UInt8
            ).cast(pl.Int64),
            "physical_entity_id": pl.Series(rng.integers(0, n_physical, size=rows)),
            "logical_entity_id": pl.Series(rng.integers(0, n_logical, size=rows)),
            "start_ts": pl.Series(start_ns).cast(pl.Datetime("ns")).dt.cast_time_unit("us"),
            "end_ts": pl.Series(end_ns).cast(pl.Datetime("ns")).dt.cast_time_unit("us"),
        }
    ).with_columns(
        pl.col("metric_id").replace_strict(
            list(range(len(MEASUREMENT_METRICS))), list(MEASUREMENT_METRICS), return_dtype=pl.String
        ),
        ("core_" + pl.col("physical_entity_id").cast(pl.String)).alias("physical_entity_id"),
        ("le_" + pl.col("logical_entity_id").cast(pl.String)).alias("logical_entity_id"),
    ).select("id", "value", "metric_id", "physical_entity_id", "logical_entity_id", "start_ts", "end_ts")


MEASUREMENT_DDL_WITH_PK = """
CREATE TABLE measurement (
    id BIGINT PRIMARY KEY,
    value DOUBLE,
    metric_id VARCHAR,
    physical_entity_id VARCHAR,
    logical_entity_id VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP
)
"""

MEASUREMENT_DDL_NO_PK = """
CREATE TABLE measurement (
    id BIGINT,
    value DOUBLE,
    metric_id VARCHAR,
    physical_entity_id VARCHAR,
    logical_entity_id VARCHAR,
    start_ts TIMESTAMP,
    end_ts TIMESTAMP
)
"""
