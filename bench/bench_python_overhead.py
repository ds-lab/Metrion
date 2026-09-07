"""Benchmark: the per-batch Python overhead in the sched_switch -> storage pipeline.

Each section isolates one thing the pipeline does today and compares it with the
proposed alternative. Nothing here needs root, BCC or a GPU, so it runs on any
Linux box (and on macOS for a rough relative picture).

Sections:
  A  snowflake id generation      util._generate_id  (Python loop, one id/row)
  B  duplicate .unique() passes   _get_raw_monitoring_data + _calculate_and_cache_metrics
  C  metric fan-out               _calculate_cpu_metrics builds all 6 metric frames
  D  cgroup map rebuild           util.build_cgroup_id_map (os.walk + stat per dir)
  E  per-row registry loop        _register_entities_from_inline_metadata / iter_rows
  F  IPC round trip               write_temp_ipc + read_temp_ipc per batch

Run:
    python bench/bench_python_overhead.py --rows 500000
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import polars as pl

from _common import Bench

RAW_SCHEMA_NUMERIC = [
    "pid_start", "time_start", "time_end", "aperf_start", "aperf_end", "mperf_start", "mperf_end",
    "unhalted_core_cycles_start", "unhalted_core_cycles_end", "loads_local_start", "loads_local_end",
    "loads_remote_start", "loads_remote_end", "stores_start", "stores_end", "tgid_start",
    "parent_tid_start", "parent_tgid_start", "cgroup_id",
]


def make_raw_df(rows: int, *, n_pids: int = 5000, seed: int = 3) -> pl.DataFrame:
    """A ring-buffer batch as it arrives in _get_raw_monitoring_data (30 columns)."""
    rng = np.random.default_rng(seed)
    pids = rng.integers(1, n_pids, size=rows).astype(np.uint32)
    data = {c: rng.integers(1, 2**40, size=rows).astype(np.uint64) for c in RAW_SCHEMA_NUMERIC}
    data |= {
        "pid": pids,
        "tgid": pids,
        "cpu_core": rng.integers(0, 128, size=rows).astype(np.uint32),
        "parent_tid": rng.integers(1, n_pids, size=rows).astype(np.uint32),
        "parent_tgid": rng.integers(1, n_pids, size=rows).astype(np.uint32),
        "uid": rng.integers(0, 2000, size=rows).astype(np.uint32),
        "gid": rng.integers(0, 2000, size=rows).astype(np.uint32),
        "event_type": (rng.random(rows) < 0.02).astype(np.uint8),
    }
    df = pl.DataFrame(data)
    return df.with_columns(
        pl.lit("some-process").alias("comm"),
        pl.lit("/usr/bin/some-process --a --b --c").alias("cmdline"),
        # pid_start must be stable per pid, otherwise the joins in METRION degenerate
        (pl.col("pid").cast(pl.UInt64) * 1_000_003).alias("pid_start"),
    )


def section_a(rows: int, repeats: int) -> None:
    from snowflake import SnowflakeGenerator

    bench = Bench("A: snowflake id generation for one batch", rows=rows, repeats=repeats)
    df = pl.DataFrame({"value": np.arange(rows, dtype=np.float64)})

    def current() -> None:
        gen = SnowflakeGenerator(1)
        ids = []
        for _ in range(rows):
            ids.append(next(gen))
        df.with_columns(pl.Series("id", ids))

    def vectorised() -> None:
        # monotonic 64-bit ids: 41-bit ms timestamp | 12-bit machine | 11-bit sequence,
        # generated as a single numpy range instead of one Python call per row
        now_ms = int(time.time() * 1000) & ((1 << 41) - 1)
        base = (now_ms << 22) | (1 << 12)
        ids = base + np.arange(rows, dtype=np.int64)
        df.with_columns(pl.Series("id", ids))

    bench.run("current: Python loop over next(gen)", current)
    bench.run("proposed: vectorised numpy id block", vectorised)
    bench.report()


def section_b(df: pl.DataFrame, repeats: int) -> None:
    bench = Bench("B: duplicate .unique() passes over the raw batch", rows=df.height, repeats=repeats)

    meta_cols = ["pid", "pid_start", "tgid", "tgid_start", "parent_tid", "parent_tgid",
                 "parent_tid_start", "parent_tgid_start", "cgroup_id", "uid", "gid",
                 "comm", "cmdline", "event_type"]
    data_cols = ["pid", "pid_start", "cpu_core", "time_start", "time_end", "aperf_start", "aperf_end",
                 "mperf_start", "mperf_end", "unhalted_core_cycles_start", "unhalted_core_cycles_end",
                 "loads_local_start", "loads_local_end", "loads_remote_start", "loads_remote_end",
                 "stores_start", "stores_end"]

    def current() -> None:
        # 1) full-frame unique in _get_raw_monitoring_data
        full = df.unique()
        # 2) unique over metadata projection
        full.select(meta_cols).unique()
        # 3) unique over measurement projection
        full.select(data_cols).unique()

    def proposed() -> None:
        # ring-buffer events are unique by construction; only the metadata
        # projection genuinely needs deduplication, and only for metadata events
        df.filter(pl.col("event_type") == 1).select(meta_cols).unique(subset=["pid", "pid_start"])
        df.select(data_cols)

    bench.run("current: 3 hash-dedup passes", current)
    bench.run("proposed: 1 dedup on metadata events only", proposed)
    bench.report()


def section_c(df: pl.DataFrame, repeats: int, enabled: int = 2) -> None:
    """The fan-out itself is a cheap projection; the cost is downstream volume.

    _calculate_cpu_metrics materialises one measurement frame per CPU metric —
    six copies of the same rows, each with a metric_id string column — and only
    afterwards drops the ones the config disabled. Every kept frame is then
    serialised to IPC, shipped to the saver process and inserted into DuckDB, so
    the multiplier lands on the two things the user reports as slow. This section
    measures the *serialise + hand over* cost, which is what actually scales.
    """
    metrics = ["CPU_CORE_PID_TIME", "CPU_CORE_PID_APERF", "CPU_CORE_PID_MPERF",
               "CPU_CORE_PID_UNHALTED_CYCLES", "CPU_CORE_PID_EXECUTION_TYPE", "CPU_CORE_PID_WORK"]
    base = df.select(
        pl.col("pid").cast(pl.String).alias("physical_entity_id"),
        pl.col("tgid").cast(pl.String).alias("logical_entity_id"),
        pl.col("time_start").alias("start_ts"),
        pl.col("time_end").alias("end_ts"),
        (pl.col("aperf_end") - pl.col("aperf_start")).cast(pl.Float64).alias("v"),
    )

    def build_and_ship(names: list[str]) -> dict[str, object]:
        import io

        total_rows = 0
        total_bytes = 0
        for name in names:
            frame = base.select(
                pl.lit(name).alias("metric_id"),
                pl.col("physical_entity_id"), pl.col("logical_entity_id"),
                pl.col("start_ts"), pl.col("end_ts"),
                pl.col("v").alias("value"),
            )
            buf = io.BytesIO()
            frame.write_ipc(buf)
            total_rows += frame.height
            total_bytes += buf.tell()
        return {"rows shipped": f"{total_rows:,}", "bytes shipped": f"{total_bytes / 1e6:.1f} MB"}

    bench = Bench("C: metric fan-out -> IPC + DB volume", rows=df.height, repeats=repeats)
    bench.run("current: materialise + ship all 6 metrics", lambda: build_and_ship(metrics))
    bench.run(f"proposed: only the {enabled} enabled metrics", lambda: build_and_ship(metrics[:enabled]))
    bench.report()


def section_d(repeats: int) -> None:
    root = "/sys/fs/cgroup"
    if not os.path.isdir(root):
        print("=== D: cgroup map rebuild — skipped, /sys/fs/cgroup not present (not Linux) ===\n")
        return

    def build_cgroup_id_map(root: str = root) -> dict[int, str]:
        id2path = {}
        for dirpath, _, _ in os.walk(root):
            rel = os.path.relpath(dirpath, root)
            path = "/" if rel == "." else "/" + rel
            try:
                st = os.stat(dirpath)
            except FileNotFoundError:
                continue
            id2path[st.st_ino] = path
        return id2path

    cached: dict[int, str] = {}

    def current() -> None:
        build_cgroup_id_map()

    def proposed() -> None:
        # incremental: only stat directories we have not seen before
        for dirpath, _, _ in os.walk(root):
            try:
                ino = os.stat(dirpath).st_ino
            except FileNotFoundError:
                continue
            if ino not in cached:
                rel = os.path.relpath(dirpath, root)
                cached[ino] = "/" if rel == "." else "/" + rel

    n = len(build_cgroup_id_map())
    bench = Bench(f"D: cgroup map rebuild ({n} cgroups)", rows=n, repeats=repeats)
    bench.run("current: full os.walk + stat per call", current)
    proposed()  # warm
    bench.run("proposed: cached, incremental", proposed)
    bench.report()


def section_e(df: pl.DataFrame, repeats: int) -> None:
    bench = Bench("E: per-row registry loop", rows=df.height, repeats=repeats)
    meta = df.filter(pl.col("event_type") == 1).select(
        ["pid", "pid_start", "tgid", "comm", "cmdline", "uid", "gid"]
    ).unique(subset=["pid", "pid_start"])
    registry: dict[tuple[int, int], dict] = {}

    def current() -> None:
        registry.clear()
        for row in meta.iter_rows(named=True):
            key = (row["pid"], row["pid_start"])
            if key in registry:
                continue
            registry[key] = {
                "tgid": row["tgid"], "comm": row["comm"], "cmdline": row["cmdline"],
                "uid": row["uid"], "gid": row["gid"],
            }

    def proposed() -> None:
        registry.clear()
        known = set(registry)
        # anti-join in polars, then only touch genuinely new rows in Python
        if known:
            known_df = pl.DataFrame({"pid": [k[0] for k in known], "pid_start": [k[1] for k in known]})
            new = meta.join(known_df, on=["pid", "pid_start"], how="anti")
        else:
            new = meta
        for key, tgid, comm, cmdline, uid, gid in zip(
            zip(new["pid"].to_list(), new["pid_start"].to_list()),
            new["tgid"].to_list(), new["comm"].to_list(), new["cmdline"].to_list(),
            new["uid"].to_list(), new["gid"].to_list(),
            strict=True,
        ):
            registry[key] = {"tgid": tgid, "comm": comm, "cmdline": cmdline, "uid": uid, "gid": gid}

    print(f"  ({meta.height:,} metadata rows out of {df.height:,} events)")
    print("  NOTE: the real loop also calls get_logical_entity() and reads /proc/<pid>/cmdline")
    print("        per new entity, so the production gap is larger than shown here.")
    bench.run("current: iter_rows(named=True) + dict build", current)
    bench.run("proposed: anti-join + columnar to_list", proposed)
    bench.report()


def section_f(df: pl.DataFrame, repeats: int) -> None:
    bench = Bench("F: IPC round trip per batch", rows=df.height, repeats=repeats)
    tmpdir = Path(tempfile.gettempdir())

    def uncompressed() -> dict[str, object]:
        path = tmpdir / f"bench_{uuid.uuid4().hex}.ipc"
        df.write_ipc(path, compression="uncompressed")
        size = path.stat().st_size
        pl.read_ipc(path)
        path.unlink(missing_ok=True)
        return {"file size": f"{size / 1e6:.1f} MB"}

    def lz4() -> dict[str, object]:
        path = tmpdir / f"bench_{uuid.uuid4().hex}.ipc"
        df.write_ipc(path, compression="lz4")
        size = path.stat().st_size
        pl.read_ipc(path)
        path.unlink(missing_ok=True)
        return {"file size": f"{size / 1e6:.1f} MB"}

    def in_memory() -> dict[str, object]:
        import io

        buf = io.BytesIO()
        df.write_ipc(buf)
        size = buf.tell()
        buf.seek(0)
        pl.read_ipc(buf)
        return {"buffer size": f"{size / 1e6:.1f} MB"}

    bench.run("current: temp file, uncompressed", uncompressed)
    bench.run("alt: temp file, lz4", lz4)
    bench.run("alt: shared-memory buffer (no disk)", in_memory)
    bench.report()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000, help="ring-buffer events per batch")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", default="abcdef", help="sections to run, e.g. --only bc")
    args = parser.parse_args()

    print(f"building a synthetic ring-buffer batch of {args.rows:,} events ...")
    df = make_raw_df(args.rows)
    print(f"batch in memory: {df.estimated_size('mb'):.1f} MB, {len(df.columns)} columns")
    print()

    if "a" in args.only:
        section_a(args.rows, args.repeats)
    if "b" in args.only:
        section_b(df, args.repeats)
    if "c" in args.only:
        section_c(df, args.repeats)
    if "d" in args.only:
        section_d(args.repeats)
    if "e" in args.only:
        section_e(df, args.repeats)
    if "f" in args.only:
        section_f(df, args.repeats)


if __name__ == "__main__":
    main()
