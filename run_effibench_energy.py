#!/usr/bin/env python3
"""
Run EffiBench entries under metrion monitoring and collect per-entry energy data.

Flow:
  1. Start `metrion monitor` in the background (writes to DB via --persist).
  2. Wait --warmup seconds for metrion to initialise.
  3. For each input file and each entry within it, build a runnable script and
     execute it as a subprocess, recording the PID and timing.
  4. Send SIGTERM to metrion and wait until it exits gracefully (DB flush).
  5. Query the database for energy, CPU cycles, kernel time, and memory per PID
     (including child processes).
  6. Write one JSON results file per input file, named {stem}_energy_results.json.

Usage:
    python run_effibench_energy.py --input-files EffiBench_vllm_qwen3.5-27b_1.json
    python run_effibench_energy.py --input-files file1.json file2.json file3.json
    python run_effibench_energy.py --input-files file1.json --limit 5
    python run_effibench_energy.py --input-files file1.json file2.json \\
        --config /path/to/config.yaml
    python run_effibench_energy.py --analyze-only \\
        --input-files file1.json file2.json --db-path /path/to/monitor.db
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers copied from test.py (build_runnable_script and dependencies).
# test.py is not a proper module, so we inline what we need.
# ---------------------------------------------------------------------------

LIST_NODE_TEXT = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
"""

TREE_NODE_TEXT = """
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

IMPORT_PKG = """
from typing import *
from bisect import *
from collections import *
from copy import *
from datetime import *
from heapq import *
from math import *
from re import *
from string import *
from random import *
from itertools import *
from functools import *
from operator import *

import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import itertools
import functools
import operator
"""


def _strip_markdown_fences(code: str) -> str:
    if "```python" in code:
        start = code.find("```python") + len("```python")
        rest = code[start:]
        if "```" in rest:
            rest = rest[: rest.find("```")]
        return rest
    if "```" in code:
        start = code.find("```") + 3
        rest = code[start:]
        if "```" in rest:
            rest = rest[: rest.find("```")]
        return rest
    return code


def _find_class_name(code: str) -> str | None:
    """Return the first class name defined in *code*, or None."""
    m = re.search(r"^class\s+(\w+)", code, re.MULTILINE)
    return m.group(1) if m else None


def build_runnable_script(entry: dict) -> str | None:
    """
    Reproduce EffiLearner's add_string_to_py_file() for the EffiBench dataset.
    Returns None if the entry can't be turned into a runnable script.
    """
    completion = entry.get("completion") or ""
    test_case = entry.get("test_case") or ""
    if not completion:
        return None

    completion = _strip_markdown_fences(completion)

    cls_name = _find_class_name(completion)
    if cls_name is None:
        return None

    # Cap test cases at 100 lines, matching add_string_to_py_file().
    test_case = "\n".join(test_case.split("\n")[:100])

    # If the class isn't called Solution but the test harness expects one,
    # add an alias so `solution = Solution()` still works.
    alias = ""
    if cls_name != "Solution" and "Solution()" in test_case:
        alias = f"\nSolution = {cls_name}\n"

    return (
        IMPORT_PKG
        + "\n"
        + TREE_NODE_TEXT
        + "\n"
        + LIST_NODE_TEXT
        + "\n"
        + completion
        + alias
        + "\nsolution=Solution()\n"
        + test_case
    )


# ---------------------------------------------------------------------------
# Metrion lifecycle helpers
# ---------------------------------------------------------------------------


def start_metrion(config_path: Path, log_dir: Path) -> tuple[subprocess.Popen, Path]:
    venv_metrion = Path(__file__).resolve().parent / ".venv" / "bin" / "metrion"
    cmd = ["sudo", str(venv_metrion), "monitor", "--config", str(config_path)]
    timestamp = datetime.now(UTC).strftime("%Y_%m_%d_%H_%M_%S")
    log_path = log_dir / f"metrion_{timestamp}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", buffering=1)  # line-buffered
    logger.info(f"[metrion] Starting: {' '.join(cmd)}")
    logger.info(f"[metrion] Logging to: {log_path}")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    proc._log_file = log_file  # keep reference so it stays open
    return proc, log_path


def stop_metrion(proc: subprocess.Popen) -> None:
    logger.info("[metrion] Sending SIGINT (equivalent to Ctrl+C) …")
    proc.send_signal(signal.SIGINT)
    t0 = time.monotonic()
    while True:
        rc = proc.poll()
        if rc is not None:
            elapsed = time.monotonic() - t0
            logger.info(f"[metrion] Exited (rc={rc}) after {elapsed:.0f}s")
            break
        elapsed = time.monotonic() - t0
        logger.info(f"[metrion] Waiting for graceful shutdown … {elapsed:.0f}s elapsed")
        time.sleep(5)
    log_file = getattr(proc, "_log_file", None)
    if log_file is not None:
        log_file.close()


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def run_script(
    script_text: str,
    timeout_seconds: float | None = None,
) -> tuple[int, datetime, datetime, int | None, bool]:
    """Write script to a temp file, execute it, return (pid, start, end, returncode, timed_out)."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", prefix="metrion_effibench_", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(script_text)
        script_path = f.name

    try:
        start_time = datetime.now(UTC)
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        pid = proc.pid
        timed_out = False
        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                proc.wait(timeout=timeout_seconds)
            else:
                proc.wait()
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning(
                f"[runner] Script PID={pid} exceeded timeout of {timeout_seconds}s; terminating process group."
            )
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        end_time = datetime.now(UTC)
        return pid, start_time, end_time, proc.returncode, timed_out
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _db_config_from_yaml(config_path: Path) -> tuple[Path, str, bool]:
    """Return (base_db_path, db_type, debug) from a metrion config.yaml."""
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    general = cfg.get("general", {})
    db_cfg = general.get("database", {})
    db_type = db_cfg.get("type", "duckdb")
    raw_path = db_cfg.get("path", "./monitor.db")
    debug = bool(general.get("debug", False))
    # Metrion resolves relative DB paths against its working directory (the
    # repo root, one level above the config/ folder), not against the config
    # file itself.  Mirror that behaviour here.
    db_path = (config_path.parent.parent / raw_path).resolve()
    return db_path, db_type, debug


def find_debug_db(base_db_path: Path, started_after: float) -> Path | None:
    """
    When debug=true, metrion appends a timestamp to the DB stem
    (e.g. monitor.db → monitor_2026_04_09_10_30_00.db).

    Find the newest file matching ``{stem}_*.{suffix}`` in the same directory
    that was *created/modified* after ``started_after`` (a monotonic-clock value
    converted to wall-clock mtime via a small bias correction isn't needed —
    we use actual file mtime compared to the wall-clock time we recorded).
    """
    stem = base_db_path.stem
    suffix = base_db_path.suffix
    parent = base_db_path.parent

    candidates = sorted(
        parent.glob(f"{stem}_*{suffix}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # started_after is a monotonic value; convert to an approximate wall-clock
    # reference by computing offset from time.monotonic() at call time.
    wall_ref = time.time() - (time.monotonic() - started_after)

    for candidate in candidates:
        if candidate.stat().st_mtime >= wall_ref:
            return candidate

    # Fallback: return the most recent match even if mtime is older.
    return candidates[0] if candidates else None


def _aggregate_energy(df) -> dict:
    """Sum energy columns across all PID/interval rows."""
    if df is None or df.is_empty():
        return {
            "cpu_total_energy_j": None,
            "cpu_active_energy_j": None,
            "cpu_idle_energy_j": None,
            "dram_total_energy_j": None,
            "dram_active_energy_j": None,
            "dram_idle_energy_j": None,
            "gpu_total_energy_j": None,
            "gpu_active_energy_j": None,
            "gpu_idle_energy_j": None,
        }

    def _col_sum(col: str):
        if col in df.columns:
            v = df[col].sum()
            return float(v) if v is not None else None
        return None

    return {
        "cpu_total_energy_j": _col_sum("cpu_total_energy_j"),
        "cpu_active_energy_j": _col_sum("cpu_active_energy_j"),
        "cpu_idle_energy_j": _col_sum("cpu_idle_energy_j"),
        "dram_total_energy_j": _col_sum("dram_total_energy_j"),
        "dram_active_energy_j": _col_sum("dram_active_energy_j"),
        "dram_idle_energy_j": _col_sum("dram_idle_energy_j"),
        "gpu_total_energy_j": _col_sum("gpu_total_energy_j"),
        "gpu_active_energy_j": _col_sum("gpu_active_energy_j"),
        "gpu_idle_energy_j": _col_sum("gpu_idle_energy_j"),
    }


def _aggregate_perf(df) -> dict:
    """Sum/max performance columns across all PID rows."""
    if df is None or df.is_empty():
        return {
            "unhalted_cycles": None,
            "kernel_time_ms": None,
            "dram_memory_mb": None,
            "gpu_memory_mb": None,
        }

    def _col_sum(col):
        if col in df.columns:
            v = df[col].sum()
            return float(v) if v is not None else None
        return None

    def _col_max(col):
        if col in df.columns:
            v = df[col].max()
            return float(v) if v is not None else None
        return None

    return {
        "unhalted_cycles": _col_sum("unhalted_cycles"),
        "kernel_time_ms": _col_sum("kernel_time_ms"),
        "dram_memory_mb": _col_max("dram_memory_mb"),
        "gpu_memory_mb": _col_max("gpu_memory_mb"),
    }


def _resolve_db_path(args: argparse.Namespace, metrion_start_mono: float) -> tuple[Path, str]:
    """Return (db_path, db_type), handling debug timestamped filenames."""
    if args.db_path is not None:
        _, db_type, _ = _db_config_from_yaml(args.config)
        return args.db_path, db_type

    base_db_path, db_type, debug_mode = _db_config_from_yaml(args.config)
    if debug_mode:
        db_path = find_debug_db(base_db_path, metrion_start_mono)
        if db_path is None:
            logger.warning(
                f"[db] debug=true but no timestamped DB found matching "
                f"{base_db_path.stem}_*{base_db_path.suffix} in {base_db_path.parent}"
            )
            db_path = base_db_path
        else:
            logger.info(f"[db] debug=true — using timestamped DB: {db_path.name}")
    else:
        db_path = base_db_path
    return db_path, db_type


def _run_analysis(records: list[dict], db_path: Path, db_type: str) -> None:
    """Phase D: bulk-query DB for energy/perf and annotate records in place.

    Three queries total: one recursive SQL to resolve all descendant threads,
    one bulk energy query, one bulk perf query.  Per-record attribution is then
    done in polars in memory using top_parent_native_pid.
    """
    import polars as pl

    from metrion.energy_attribution.energy_attribution_cli import (
        close_db_connection,
        connect_to_db,
        get_descendant_threads,
        get_energy_for_pids,
        get_performance_metrics_for_pids,
    )

    engine = connect_to_db(db_path, db_type)
    if engine is None:
        logger.error("[db] Failed to connect to database.")
        return

    active_records = [rec for rec in records if rec["pid"] is not None]
    if not active_records:
        close_db_connection(engine)
        return

    # --- Step 1: resolve all descendant threads in one recursive SQL query ---
    seed_pids = [rec["pid"] for rec in active_records]
    logger.info(f"[db] Resolving descendant threads for {len(seed_pids)} seed PIDs …")
    threads_df = get_descendant_threads(engine, seed_pids)

    # Build per-record all_pids from top_parent_native_pid grouping
    if not threads_df.is_empty():
        pid_map: dict[int, list[int]] = {
            row["top_parent_native_pid"]: sorted(row["native_pid"])
            for row in threads_df.group_by("top_parent_native_pid")
            .agg(pl.col("native_pid").unique())
            .iter_rows(named=True)
        }
        all_pids: list[int] = threads_df["native_pid"].unique().to_list()
    else:
        pid_map = {}
        all_pids = seed_pids

    for rec in active_records:
        rec["all_pids"] = pid_map.get(rec["pid"], [rec["pid"]])

    # --- Step 2: two bulk queries for all PIDs across all records ---
    time_bound = timedelta(days=1)
    global_start = min(datetime.fromisoformat(rec["start_time"]) for rec in active_records) - time_bound
    global_end = max(datetime.fromisoformat(rec["end_time"]) for rec in active_records) + time_bound

    logger.info(f"[db] Bulk energy query for {len(all_pids)} PIDs …")
    try:
        energy_df: pl.DataFrame | None = get_energy_for_pids(engine, all_pids, global_start, global_end)
    except Exception as e:
        logger.warning(f"[db] get_energy_for_pids (bulk) failed: {e}")
        energy_df = None

    logger.info(f"[db] Bulk perf query for {len(all_pids)} PIDs …")
    try:
        perf_df: pl.DataFrame | None = get_performance_metrics_for_pids(engine, all_pids, global_start, global_end)
    except Exception as e:
        logger.warning(f"[db] get_performance_metrics_for_pids (bulk) failed: {e}")
        perf_df = None

    close_db_connection(engine)

    # --- Step 3: filter by top_parent_native_pid, aggregate per record in polars ---
    for rec in active_records:
        pid_set = rec["all_pids"]

        rec_energy = (
            energy_df.filter(pl.col("native_pid").is_in(pid_set))
            if energy_df is not None and not energy_df.is_empty()
            else None
        )
        rec["energy"] = _aggregate_energy(rec_energy)

        rec_perf = (
            perf_df.filter(pl.col("native_pid").is_in(pid_set))
            if perf_df is not None and not perf_df.is_empty()
            else None
        )
        rec["performance"] = _aggregate_perf(rec_perf)

        logger.info(
            f"[db] problem={rec['problem_idx']} pids={rec['all_pids']} "
            f"cpu_energy={rec['energy'].get('cpu_total_energy_j')} J "
            f"cycles={rec['performance'].get('unhalted_cycles')}"
        )


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def _out_path(input_file: Path) -> Path:
    """Derive the output JSON path: {stem}_energy_results.json in same directory."""
    return input_file.with_stem(input_file.stem + "_energy_results")


def _pids_path(input_file: Path) -> Path:
    """Derive the intermediate pids JSON path: {stem}_energy_results.pids.json."""
    return _out_path(input_file).with_suffix(".pids.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        required=True,
        metavar="INPUT_FILE",
        help=(
            "One or more EffiBench JSON files. Each produces its own "
            "{stem}_energy_results.json output in the same directory."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "config.yaml",
        help="Path to metrion config.yaml.",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip metrion and script execution; load per-file pids files "
        "(auto-derived from --input-files) and re-run only the database "
        "analysis (Phase D+E). Requires --db-path.",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Explicit path to the metrion DuckDB file. Required with "
        "--analyze-only; otherwise auto-detected from config.",
    )
    p.add_argument("--limit", type=int, default=None, help="Only process the first N entries (debug).")
    p.add_argument(
        "--entry-timeout",
        type=float,
        default=120.0,
        help="Max runtime in seconds per entry script. Use 0 or negative to disable (default: 120).",
    )
    p.add_argument(
        "--warmup", type=int, default=20, help="Seconds to wait for metrion to initialise before running scripts."
    )
    p.add_argument(
        "--log-dir",
        type=Path,
        default=Path(__file__).parent / "logs",
        help="Directory for metrion log files (default: container/logs/).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # --analyze-only: skip straight to Phase D using saved pids files
    # -----------------------------------------------------------------------
    if args.analyze_only:
        if args.db_path is None:
            logger.error("[analyze] --db-path is required with --analyze-only")
            sys.exit(1)

        db_path, db_type = _resolve_db_path(args, 0.0)
        logger.info(f"[db] Connecting to {db_type}:///{db_path}")
        if not db_path.exists():
            logger.error(f"[db] Database not found at {db_path}")
            sys.exit(1)

        for input_file in args.input_files:
            pids_file = _pids_path(input_file)
            out_file = _out_path(input_file)
            if not pids_file.exists():
                logger.error(f"[analyze] Pids file not found: {pids_file}")
                continue
            records: list[dict] = json.loads(pids_file.read_text())
            logger.info(f"[analyze] Loaded {len(records)} records from {pids_file.name}")
            _run_analysis(records, db_path, db_type)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(records, indent=2, default=str))
            logger.info(f"Results written to {out_file} ({len(records)} entries)")
        return

    # -----------------------------------------------------------------------
    # Normal run: Phases A → B → C → D → E
    # -----------------------------------------------------------------------
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    for input_file in args.input_files:
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)

    # Load all entries grouped by input file.
    file_entries: list[tuple[Path, list[dict]]] = []
    for input_file in args.input_files:
        entries: list[dict] = json.loads(input_file.read_text())
        if args.limit is not None:
            entries = entries[: args.limit]
        logger.info(f"Loaded {len(entries)} entries from {input_file.name}")
        file_entries.append((input_file, entries))

    # --- Phase A: start metrion ---
    metrion_start_mono = time.monotonic()
    metrion_proc, metrion_log = start_metrion(args.config, args.log_dir)
    logger.info(f"[metrion] PID={metrion_proc.pid}. Waiting {args.warmup}s for initialisation …")
    time.sleep(args.warmup)
    if metrion_proc.poll() is not None:
        logger.error("[metrion] Process died during warmup!")
        sys.exit(1)
    logger.info("[metrion] Ready.")

    # --- Phase B: run scripts (all input files inside a single metrion session) ---
    file_records: list[tuple[Path, list[dict]]] = []
    for input_file, entries in tqdm(
        file_entries,
        desc="Processing input files",
        unit="file",
        file=sys.stdout,
    ):
        records: list[dict] = []
        total = len(entries)
        for i, entry in tqdm(
            enumerate(entries),
            total=total,
            desc=f"Entries in {input_file.name}",
            unit="entry",
            file=sys.stdout,
            leave=False,
        ):
            problem_idx = entry.get("problem_idx")
            task_name = entry.get("task_name", "")
            prefix = f"[{i + 1}/{total}|{input_file.name}] problem={problem_idx} task={task_name!r}"

            script_text = build_runnable_script(entry)
            if script_text is None:
                logger.info(f"{prefix} … SKIP (no runnable completion)")
                records.append(
                    {
                        "source_entry_index": i,
                        "problem_idx": problem_idx,
                        "task_name": task_name,
                        "pid": None,
                        "all_pids": [],
                        "returncode": None,
                        "start_time": None,
                        "end_time": None,
                        "energy": {},
                        "performance": {},
                        "error": "no_runnable_completion",
                    }
                )
                continue

            try:
                pid, start_time, end_time, returncode, timed_out = run_script(script_text, args.entry_timeout)
                if timed_out:
                    logger.warning(f"{prefix} … TIMEOUT after {args.entry_timeout}s (rc={returncode})")
                records.append(
                    {
                        "source_entry_index": i,
                        "problem_idx": problem_idx,
                        "task_name": task_name,
                        "pid": pid,
                        "all_pids": [pid],
                        "returncode": returncode,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "energy": {},
                        "performance": {},
                        "error": f"execution_timeout: exceeded {args.entry_timeout}s" if timed_out else "",
                    }
                )
            except Exception as e:
                logger.error(f"{prefix} … ERROR: {e}")
                records.append(
                    {
                        "source_entry_index": i,
                        "problem_idx": problem_idx,
                        "task_name": task_name,
                        "pid": None,
                        "all_pids": [],
                        "returncode": None,
                        "start_time": None,
                        "end_time": None,
                        "energy": {},
                        "performance": {},
                        "error": f"execution_failed: {e}",
                    }
                )

        # Save intermediate pids file for this input before moving to the next.
        pids_file = _pids_path(input_file)
        pids_file.parent.mkdir(parents=True, exist_ok=True)
        pids_file.write_text(json.dumps(records, indent=2, default=str))
        logger.info(f"[pids] Intermediate records saved to {pids_file}")
        file_records.append((input_file, records))

    # --- Phase C: stop metrion ---
    stop_metrion(metrion_proc)

    # --- Phase D: query database (resolve DB and chown once) ---
    db_path, db_type = _resolve_db_path(args, metrion_start_mono)

    # The DB is written by root (sudo metrion); transfer ownership to the current user.
    if db_path.exists():
        result = subprocess.run(
            ["sudo", "chown", f"{os.getuid()}:{os.getgid()}", str(db_path)],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info(f"[db] Ownership of {db_path.name} transferred to current user.")
        else:
            logger.warning(f"[db] chown failed: {result.stderr.decode().strip()}")

    logger.info(f"[db] Connecting to {db_type}:///{db_path}")
    db_available = db_path.exists()
    if not db_available:
        logger.error(f"[db] Database not found at {db_path}. Skipping energy queries.")

    # --- Phase E: write one output file per input ---
    for input_file, records in file_records:
        if db_available:
            _run_analysis(records, db_path, db_type)
        out_file = _out_path(input_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(records, indent=2, default=str))
        logger.info(f"Results written to {out_file} ({len(records)} entries)")


if __name__ == "__main__":
    main()
