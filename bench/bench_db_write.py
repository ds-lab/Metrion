"""Benchmark: how much does the current measurement write path cost, and why.

Reproduces `SQLDatabaseStorage._save_measurement_df` and compares it against the
variants that the optimisation proposal suggests:

  1. baseline      -> what METRION does today (BIGINT PRIMARY KEY + strftime to
                      VARCHAR + register(polars) + INSERT ... SELECT)
  2. native_ts     -> same, but hand DuckDB real TIMESTAMP columns via Arrow
  3. no_pk         -> drop the PRIMARY KEY on measurement.id (append-only log)
  4. no_pk_native  -> both of the above
  5. tuned         -> + preserve_insertion_order=false and a bigger memory limit
  6. arrow_direct  -> register the Arrow table instead of the polars frame
  7. parquet       -> write a partitioned parquet file instead of a DB table

With --prefill N the table is pre-loaded with N rows before the measurement
starts. This is the interesting case: a BIGINT PRIMARY KEY builds an ART index
whose maintenance cost grows with the table, which is why a long METRION run
gets slower the longer it runs, while an unconstrained table stays flat.

Run:
    python bench/bench_db_write.py --rows 2000000 --batch 250000 --repeats 3
    python bench/bench_db_write.py --rows 1000000 --prefill 20000000
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import duckdb
import polars as pl

from _common import MEASUREMENT_DDL_NO_PK, MEASUREMENT_DDL_WITH_PK, Bench, make_measurement_df

INSERT_SQL = """
INSERT INTO measurement
    (id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts)
SELECT id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts
FROM tmp_df
"""


def _stringify_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Exactly what db_data_storage._save_measurement_df does today."""
    return df.with_columns(
        pl.col("start_ts")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("UTC")
        .dt.strftime("%Y-%m-%d %H:%M:%S%.6f")
        .alias("start_ts"),
        pl.col("end_ts")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("UTC")
        .dt.strftime("%Y-%m-%d %H:%M:%S%.6f")
        .alias("end_ts"),
    )


class Scenario:
    def __init__(
        self,
        workdir: Path,
        *,
        ddl: str,
        stringify: bool,
        tuned: bool = False,
        prefill: pl.DataFrame | None = None,
    ) -> None:
        self.path = workdir / f"bench_{id(self)}.duckdb"
        self.ddl = ddl
        self.stringify = stringify
        self.tuned = tuned
        self.prefill = prefill
        self.con: duckdb.DuckDBPyConnection | None = None

    def setup(self) -> None:
        self.close()
        self.path.unlink(missing_ok=True)
        Path(str(self.path) + ".wal").unlink(missing_ok=True)
        self.con = duckdb.connect(str(self.path))
        if self.tuned:
            self.con.execute("SET preserve_insertion_order=false")
            self.con.execute("SET memory_limit='4GB'")
            self.con.execute("SET threads=4")
        self.con.execute(self.ddl)
        if self.prefill is not None:
            self.con.register("prefill_df", self.prefill)
            try:
                self.con.execute(INSERT_SQL.replace("tmp_df", "prefill_df"))
            finally:
                self.con.unregister("prefill_df")
            self.con.execute("CHECKPOINT")

    def insert(self, batches: list[pl.DataFrame]) -> dict[str, object]:
        assert self.con is not None
        for batch in batches:
            payload = _stringify_timestamps(batch) if self.stringify else batch
            self.con.register("tmp_df", payload)
            try:
                self.con.execute(INSERT_SQL)
            finally:
                self.con.unregister("tmp_df")
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"db size": f"{size / 1e6:.1f} MB"}

    def insert_arrow(self, arrow_batches: list[object]) -> dict[str, object]:
        """Hand DuckDB an Arrow table directly (no polars replacement scan)."""
        assert self.con is not None
        for tbl in arrow_batches:
            self.con.from_arrow(tbl).insert_into("measurement")
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"db size": f"{size / 1e6:.1f} MB"}

    def close(self) -> None:
        if self.con is not None:
            self.con.close()
            self.con = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch", type=int, default=250_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--prefill",
        type=int,
        default=0,
        help="rows to load into the table before measuring (shows index-growth cost)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the generated database files")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="metrion_bench_db_"))
    print(f"workdir: {workdir}")
    print(f"generating {args.rows:,} measurement rows ...")
    df = make_measurement_df(args.rows)
    batches = [df.slice(i, args.batch) for i in range(0, df.height, args.batch)]
    arrow_batches = [b.to_arrow() for b in batches]
    print(f"{len(batches)} batches of up to {args.batch:,} rows")

    prefill_df = None
    if args.prefill:
        print(f"generating {args.prefill:,} prefill rows ...")
        prefill_df = make_measurement_df(args.prefill, seed=99).with_columns(pl.col("id") + 10_000_000_000)
    print()

    bench = Bench("DuckDB measurement write path", rows=args.rows, repeats=args.repeats)
    scenarios: list[Scenario] = []

    def add(name: str, *, ddl: str, stringify: bool, tuned: bool = False, arrow: bool = False) -> None:
        sc = Scenario(workdir, ddl=ddl, stringify=stringify, tuned=tuned, prefill=prefill_df)
        scenarios.append(sc)
        fn = (lambda: sc.insert_arrow(arrow_batches)) if arrow else (lambda: sc.insert(batches))
        bench.run(name, fn, setup=sc.setup)

    add("1 baseline (PK + varchar ts)", ddl=MEASUREMENT_DDL_WITH_PK, stringify=True)
    add("2 PK + native timestamps", ddl=MEASUREMENT_DDL_WITH_PK, stringify=False)
    add("3 no PK + varchar ts", ddl=MEASUREMENT_DDL_NO_PK, stringify=True)
    add("4 no PK + native timestamps", ddl=MEASUREMENT_DDL_NO_PK, stringify=False)
    add("5 no PK + native + tuned pragmas", ddl=MEASUREMENT_DDL_NO_PK, stringify=False, tuned=True)
    add("6 no PK + native + arrow direct", ddl=MEASUREMENT_DDL_NO_PK, stringify=False, tuned=True, arrow=True)

    # 7: parquet sink (what a "write raw, aggregate later" architecture would cost)
    parquet_dir = workdir / "parquet"

    def parquet_setup() -> None:
        shutil.rmtree(parquet_dir, ignore_errors=True)
        parquet_dir.mkdir(parents=True, exist_ok=True)

    def parquet_write() -> dict[str, object]:
        for i, batch in enumerate(batches):
            batch.write_parquet(parquet_dir / f"part_{i:05d}.parquet", compression="zstd")
        size = sum(p.stat().st_size for p in parquet_dir.glob("*.parquet"))
        return {"parquet size": f"{size / 1e6:.1f} MB"}

    bench.run("7 parquet sink (zstd)", parquet_write, setup=parquet_setup)

    for sc in scenarios:
        sc.close()
    bench.report()

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"artifacts kept in {workdir}")


if __name__ == "__main__":
    main()
