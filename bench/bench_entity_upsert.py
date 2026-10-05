"""Benchmark: the logical-entity upsert path in SQLDatabaseStorage.save_entities.

METRION re-upserts every new *and* every metadata-enriched logical entity on
every collection cycle. Today that happens as

    session.execute(insert(logical_entity).on_conflict_do_update(...), rows)

i.e. an executemany of an ON CONFLICT statement with a coalesce() expression per
column, through SQLAlchemy + duckdb-engine, in its own transaction. DuckDB is a
columnar engine; row-at-a-time conflict resolution is its worst case.

Variants compared:
  1. sqlalchemy_upsert  -> what METRION does today
  2. staging_upsert     -> bulk-load the batch into a temp table via Arrow, then
                           run ONE set-based INSERT ... ON CONFLICT DO UPDATE
  3. staging_merge      -> same, but resolve duplicates with an explicit
                           anti-join + UPDATE (no ON CONFLICT / no index needed)
  4. append_only        -> append every version, deduplicate at read time

Run:
    python bench/bench_entity_upsert.py --entities 20000 --cycles 10
"""

from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

import duckdb
import polars as pl
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, create_engine, func
from sqlalchemy.orm import sessionmaker

from _common import Bench

LOGICAL_ENTITY_COLUMNS = [
    "native_thread_id",
    "native_pid",
    "pid_id",
    "native_parent_tid",
    "parent_tid_id",
    "native_parent_pid",
    "parent_pid_id",
    "executable_name",
    "cmdline",
    "resource_group_id",
    "resource_group_name",
    "user_id",
    "group_id",
]

DDL = """
CREATE TABLE logical_entity (
    id VARCHAR PRIMARY KEY,
    native_thread_id BIGINT,
    native_pid BIGINT,
    pid_id VARCHAR,
    native_parent_tid BIGINT,
    parent_tid_id VARCHAR,
    native_parent_pid BIGINT,
    parent_pid_id VARCHAR,
    executable_name VARCHAR,
    cmdline VARCHAR,
    resource_group_id BIGINT,
    resource_group_name VARCHAR,
    user_id BIGINT,
    group_id BIGINT
)
"""


def make_rows(n: int, *, seed: int, overlap_ids: list[str] | None = None, overlap: float = 0.7) -> list[dict]:
    """Build entity rows; `overlap` of them reuse ids that already exist (the update path)."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        if overlap_ids and rng.random() < overlap:
            entity_id = rng.choice(overlap_ids)
        else:
            entity_id = f"le_{seed}_{i}"
        pid = rng.randint(1, 400_000)
        rows.append(
            {
                "id": entity_id,
                "native_thread_id": pid,
                "native_pid": pid,
                "pid_id": f"le_{seed}_{pid}",
                "native_parent_tid": rng.randint(1, 400_000),
                "parent_tid_id": f"le_{seed}_p",
                "native_parent_pid": rng.randint(1, 400_000),
                "parent_pid_id": f"le_{seed}_pp",
                "executable_name": rng.choice(["python3", "kworker", "systemd", "nginx", "postgres"]),
                "cmdline": "/usr/bin/python3 -m something --with --a --few --flags",
                "resource_group_id": rng.randint(1, 5000),
                "resource_group_name": "/system.slice/whatever.service",
                "user_id": rng.randint(0, 2000),
                "group_id": rng.randint(0, 2000),
            }
        )
    return rows


def build_sa_table(metadata: MetaData) -> Table:
    return Table(
        "logical_entity",
        metadata,
        Column("id", String, primary_key=True),
        *[
            Column(name, BigInteger if name.startswith(("native_", "user_", "group_", "resource_group_i")) else String)
            for name in LOGICAL_ENTITY_COLUMNS
        ],
        Column("_unused", DateTime, nullable=True),
        extend_existing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=int, default=10_000, help="entities per collection cycle")
    parser.add_argument("--cycles", type=int, default=10, help="collection cycles to simulate")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="metrion_bench_entity_"))
    print(f"workdir: {workdir}")

    seed_rows = make_rows(args.entities, seed=0)
    existing_ids = [r["id"] for r in seed_rows]
    cycles = [seed_rows] + [
        make_rows(args.entities, seed=c, overlap_ids=existing_ids, overlap=0.7) for c in range(1, args.cycles)
    ]
    total_rows = sum(len(c) for c in cycles)
    print(f"{args.cycles} cycles x {args.entities:,} entities = {total_rows:,} upserts (70% updates)")
    print()

    bench = Bench("logical_entity upsert path", rows=total_rows, repeats=args.repeats)

    # ---------------- variant 1: what METRION does today -------------------
    state: dict[str, object] = {}

    def sa_setup() -> None:
        path = workdir / "sa.duckdb"
        path.unlink(missing_ok=True)
        Path(str(path) + ".wal").unlink(missing_ok=True)
        engine = create_engine(f"duckdb:///{path}")
        metadata = MetaData()
        table = build_sa_table(metadata)
        with engine.begin() as con:
            con.exec_driver_sql(DDL)
        state["engine"] = engine
        state["table"] = table
        state["Session"] = sessionmaker(bind=engine)

    def sa_upsert() -> None:
        from duckdb_engine import insert as ddb_insert

        table = state["table"]
        make_session = state["Session"]
        for cycle in cycles:
            stmt = ddb_insert(table)
            stmt = stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    name: func.coalesce(getattr(stmt.excluded, name), table.c[name]) for name in LOGICAL_ENTITY_COLUMNS
                },
            )
            session = make_session()
            try:
                session.execute(stmt, cycle)
                session.commit()
            finally:
                session.close()
        state["engine"].dispose()

    bench.run("1 sqlalchemy on_conflict executemany", sa_upsert, setup=sa_setup)

    def sa_plain_insert() -> None:
        """Isolates ON CONFLICT from plain SQLAlchemy/duckdb-engine executemany overhead."""
        table = state["table"]
        make_session = state["Session"]
        seen: set[str] = set()
        for cycle in cycles:
            fresh = [r for r in cycle if r["id"] not in seen]
            seen.update(r["id"] for r in fresh)
            if not fresh:
                continue
            session = make_session()
            try:
                session.execute(table.insert(), fresh)
                session.commit()
            finally:
                session.close()
        state["engine"].dispose()

    bench.run("1b sqlalchemy plain insert executemany", sa_plain_insert, setup=sa_setup)

    # ---------------- variants 2-4: native duckdb --------------------------
    def native_setup(name: str):
        def _setup() -> None:
            path = workdir / f"{name}.duckdb"
            path.unlink(missing_ok=True)
            Path(str(path) + ".wal").unlink(missing_ok=True)
            con = duckdb.connect(str(path))
            con.execute(DDL if name != "append" else DDL.replace(" PRIMARY KEY", ""))
            state[name] = con

        return _setup

    set_clause = ", ".join(f"{c} = coalesce(excluded.{c}, logical_entity.{c})" for c in LOGICAL_ENTITY_COLUMNS)
    all_cols = ", ".join(["id", *LOGICAL_ENTITY_COLUMNS])

    def staging_upsert() -> None:
        con: duckdb.DuckDBPyConnection = state["staging"]
        for cycle in cycles:
            df = pl.DataFrame(cycle)
            con.register("stage", df.to_arrow())
            try:
                con.execute(
                    f"INSERT INTO logical_entity ({all_cols}) "
                    f"SELECT {all_cols} FROM (SELECT *, row_number() OVER (PARTITION BY id) rn FROM stage) "
                    f"WHERE rn = 1 "
                    f"ON CONFLICT (id) DO UPDATE SET {set_clause}"
                )
            finally:
                con.unregister("stage")
        con.close()

    bench.run("2 arrow staging + set-based ON CONFLICT", staging_upsert, setup=native_setup("staging"))

    def staging_merge() -> None:
        con: duckdb.DuckDBPyConnection = state["merge"]
        for cycle in cycles:
            df = pl.DataFrame(cycle)
            con.register("stage", df.to_arrow())
            try:
                con.execute(
                    "CREATE OR REPLACE TEMP TABLE stage_dedup AS "
                    "SELECT * EXCLUDE (rn) FROM "
                    "(SELECT *, row_number() OVER (PARTITION BY id) rn FROM stage) WHERE rn = 1"
                )
                con.execute(
                    "UPDATE logical_entity SET "
                    + ", ".join(f"{c} = coalesce(s.{c}, logical_entity.{c})" for c in LOGICAL_ENTITY_COLUMNS)
                    + " FROM stage_dedup s WHERE s.id = logical_entity.id"
                )
                con.execute(
                    f"INSERT INTO logical_entity ({all_cols}) SELECT {all_cols} FROM stage_dedup s "
                    "WHERE NOT EXISTS (SELECT 1 FROM logical_entity t WHERE t.id = s.id)"
                )
            finally:
                con.unregister("stage")
        con.close()

    bench.run("3 arrow staging + UPDATE/anti-join merge", staging_merge, setup=native_setup("merge"))

    def append_only() -> None:
        con: duckdb.DuckDBPyConnection = state["append"]
        for cycle in cycles:
            df = pl.DataFrame(cycle)
            con.register("stage", df.to_arrow())
            try:
                con.execute(f"INSERT INTO logical_entity ({all_cols}) SELECT {all_cols} FROM stage")
            finally:
                con.unregister("stage")
        con.close()

    bench.run("4 append-only (dedup at read time)", append_only, setup=native_setup("append"))

    bench.report()
    shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
