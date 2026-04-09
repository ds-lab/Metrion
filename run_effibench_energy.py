#!/usr/bin/env python3
"""
Run EffiBench entries under metrion monitoring and collect per-entry energy data.

Flow:
  1. Start `metrion monitor` in the background (writes to DB via --persist).
  2. Wait --warmup seconds for metrion to initialise.
  3. For each entry in the input JSON, build a runnable script and execute it
     as a subprocess, recording the PID and timing.
  4. Send SIGTERM to metrion and wait until it exits gracefully (DB flush).
  5. Query the database for energy, CPU cycles, kernel time, and memory per PID
     (including child processes).
  6. Write a JSON results file.

Usage:
    python run_effibench_energy.py --input-file EffiBench_vllm_qwen3.5-27b_1.json
    python run_effibench_energy.py --input-file EffiBench_vllm_qwen3.5-27b_1.json --limit 5
    python run_effibench_energy.py --input-file EffiBench_vllm_qwen3.5-27b_1.json \\
        --config /path/to/config.yaml --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from loguru import logger

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


def build_runnable_script(entry: dict) -> str | None:
    completion = entry.get("completion") or ""
    test_case = entry.get("test_case") or ""
    if not completion or "class Solution" not in completion:
        return None
    completion = _strip_markdown_fences(completion)
    test_case = "\n".join(test_case.split("\n")[:100])
    return (
        IMPORT_PKG
        + "\n"
        + TREE_NODE_TEXT
        + "\n"
        + LIST_NODE_TEXT
        + "\n"
        + completion
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
        logger.info(f"[metrion] Waiting for graceful shutdown … {elapsed:.0f}s elapsed \n")
        time.sleep(5)
    log_file = getattr(proc, "_log_file", None)
    if log_file is not None:
        log_file.close()


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def run_script(script_text: str) -> tuple[int, datetime, datetime, int | None]:
    """Write script to a temp file, execute it, return (pid, start, end, returncode)."""
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
        )
        pid = proc.pid
        proc.wait()
        end_time = datetime.now(UTC)
        return pid, start_time, end_time, proc.returncode
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
    """Phase D: query DB for energy/perf and annotate records in place."""
    from metrion.energy_attribution.energy_attribution_cli import (
        close_db_connection,
        connect_to_db,
        get_all_descendant_pids,
        get_energy_for_pids,
        get_performance_metrics_for_pids,
    )

    engine = connect_to_db(db_path, db_type)
    if engine is None:
        logger.error("[db] Failed to connect to database.")
        return

    for rec in records:
        if rec["pid"] is None:
            continue
        pid = rec["pid"]
        time_bound = timedelta(days=1)  # sanity check to avoid querying with absurd time ranges
        start_time = datetime.fromisoformat(rec["start_time"]) - time_bound
        end_time = datetime.fromisoformat(rec["end_time"]) + time_bound

        try:
            descendants = get_all_descendant_pids(engine, {pid})
        except Exception as e:
            logger.warning(f"[db] get_all_descendant_pids failed for pid={pid}: {e}")
            descendants = set()

        all_pids = {pid} | descendants
        rec["all_pids"] = sorted(all_pids)

        try:
            energy_df = get_energy_for_pids(engine, list(all_pids), start_time, end_time)
            rec["energy"] = _aggregate_energy(energy_df)
        except Exception as e:
            logger.warning(f"[db] get_energy_for_pids failed for pid={pid}: {e}")
            rec["energy"] = _aggregate_energy(None)

        try:
            perf_df = get_performance_metrics_for_pids(engine, list(all_pids), start_time, end_time)
            rec["performance"] = _aggregate_perf(perf_df)
        except Exception as e:
            logger.warning(f"[db] get_performance_metrics_for_pids failed for pid={pid}: {e}")
            rec["performance"] = _aggregate_perf(None)

        logger.info(
            f"[db] problem={rec['problem_idx']} pids={rec['all_pids']} "
            f"cpu_energy={rec['energy'].get('cpu_total_energy_j')} J "
            f"cycles={rec['performance'].get('unhalted_cycles')} \n"
        )

    close_db_connection(engine)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to the EffiBench JSON file (required unless --analyze-only).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "config.yaml",
        help="Path to metrion config.yaml.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "effibench_energy_results.json",
        help="Output JSON file path.",
    )
    p.add_argument(
        "--pids-file",
        type=Path,
        default=None,
        help="Intermediate JSON with PID/timing records saved after execution "
        "and loaded by --analyze-only. Default: <out>.pids.json.",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip metrion and script execution; load --pids-file and re-run "
        "only the database analysis (Phase D+E). Requires --db-path.",
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
        "--warmup", type=int, default=45, help="Seconds to wait for metrion to initialise before running scripts."
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
    pids_file = args.pids_file or args.out.with_suffix(".pids.json")

    # -----------------------------------------------------------------------
    # --analyze-only: skip straight to Phase D using a saved pids file
    # -----------------------------------------------------------------------
    if args.analyze_only:
        if not pids_file.exists():
            logger.error(f"[analyze] Pids file not found: {pids_file}")
            sys.exit(1)
        if args.db_path is None:
            logger.error("[analyze] --db-path is required with --analyze-only")
            sys.exit(1)
        records: list[dict] = json.loads(pids_file.read_text())
        logger.info(f"[analyze] Loaded {len(records)} records from {pids_file.name}")

        db_path, db_type = _resolve_db_path(args, 0.0)
        logger.info(f"[db] Connecting to {db_type}:///{db_path}")
        if not db_path.exists():
            logger.error(f"[db] Database not found at {db_path}")
            sys.exit(1)

        _run_analysis(records, db_path, db_type)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(records, indent=2, default=str))
        logger.info(f"Results written to {args.out} ({len(records)} entries)")
        return

    # -----------------------------------------------------------------------
    # Normal run: Phases A → B → C → D → E
    # -----------------------------------------------------------------------
    if args.input_file is None:
        logger.error("--input-file is required (or use --analyze-only to skip execution)")
        sys.exit(1)
    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    entries: list[dict] = json.loads(args.input_file.read_text())
    if args.limit is not None:
        entries = entries[: args.limit]
    logger.info(f"Loaded {len(entries)} entries from {args.input_file.name}")

    # --- Phase A: start metrion ---
    metrion_start_mono = time.monotonic()
    metrion_proc, metrion_log = start_metrion(args.config, args.log_dir)
    logger.info(f"[metrion] PID={metrion_proc.pid}. Waiting {args.warmup}s for initialisation …")
    time.sleep(args.warmup)
    if metrion_proc.poll() is not None:
        logger.error("[metrion] Process died during warmup!")
        sys.exit(1)
    logger.info("[metrion] Ready.")

    # --- Phase B: run scripts ---
    records = []
    for i, entry in enumerate(entries):
        problem_idx = entry.get("problem_idx")
        task_name = entry.get("task_name", "")
        prefix = f"[{i + 1}/{len(entries)}] problem={problem_idx} task={task_name!r}"

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
            pid, start_time, end_time, returncode = run_script(script_text)
            duration = (end_time - start_time).total_seconds()
            logger.info(f"{prefix} … pid={pid} rc={returncode} {duration:.2f}s")
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
                    "error": "",
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

    # Save intermediate records so analysis can be re-run without re-executing scripts.
    pids_file.parent.mkdir(parents=True, exist_ok=True)
    pids_file.write_text(json.dumps(records, indent=2, default=str))
    logger.info(f"[pids] Intermediate records saved to {pids_file}")

    # --- Phase C: stop metrion ---
    stop_metrion(metrion_proc)

    # --- Phase D: query database ---
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
    if not db_path.exists():
        logger.error(f"[db] Database not found at {db_path}. Skipping energy queries.")
    else:
        _run_analysis(records, db_path, db_type)

    # --- Phase E: write output ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, default=str))
    logger.info(f"Results written to {args.out} ({len(records)} entries)")


if __name__ == "__main__":
    main()
