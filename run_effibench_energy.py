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
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol, TextIO

import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

_METRION_LOG_FILES: dict[int, TextIO] = {}


class _LockLike(Protocol):
    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool: ...

    def release(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers copied from test.py (build_runnable_script and dependencies).
# test.py is not a proper module, so we inline what we need.
# ---------------------------------------------------------------------------

LIST_NODE_TEXT = """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    def __eq__(self, other):
        if self is other: return True
        a, b = self, other
        seen = set()
        while a is not None and b is not None:
            if a is b: return True
            if id(a) in seen: return id(a) == id(b)
            seen.add(id(a))
            if not hasattr(b, 'val') or a.val != b.val: return False
            a, b = a.next, b.next
        return a is None and b is None
    def __hash__(self): return id(self)
"""

TREE_NODE_TEXT = """
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
    def __eq__(self, other):
        if self is other: return True
        if other is None: return False
        stack = [(self, other)]
        seen = set()
        while stack:
            a, b = stack.pop()
            if a is b: continue
            if a is None or b is None: return False
            if not hasattr(b, 'val') or a.val != b.val: return False
            key = (id(a), id(b))
            if key in seen: continue
            seen.add(key)
            stack.append((getattr(a, 'left', None), getattr(b, 'left', None)))
            stack.append((getattr(a, 'right', None), getattr(b, 'right', None)))
        return True
    def __hash__(self): return id(self)

class BigArray:
    def __init__(self, arr):
        self._arr = list(arr)
    def at(self, i):
        return self._arr[i]
    def size(self):
        return len(self._arr)

class Node:
    def __init__(self, val=0, left=None, right=None, next=None, children=None, random=None, neighbors=None, parent=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
        self.children = children if children is not None else []
        self.random = random
        self.neighbors = neighbors if neighbors is not None else []
        self.parent = parent
    def __eq__(self, other):
        if self is other: return True
        if other is None or not hasattr(other, 'val'): return False
        if self.val != other.val: return False
        if (self.left or None) != (other.left or None): return False
        if (self.right or None) != (other.right or None): return False
        return list(self.children or []) == list(getattr(other, 'children', None) or [])
    def __hash__(self): return id(self)
"""  # noqa: E501

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
pow = __builtins__['pow'] if isinstance(__builtins__, dict) else __builtins__.pow

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


_HELPER_CLASSES = {
    "TreeNode",
    "ListNode",
    "Node",
    "SegmentTree",
    "FenwickTree",
    "BIT",
    "BinaryIndexedTree",
    "Trie",
    "TrieNode",
    "UnionFind",
    "DSU",
    "DisjointSet",
    "SegTree",
    "LazySegTree",
    "Heap",
    "LinkedList",
    "DoublyLinkedList",
    "DLNode",
    "DLinkedNode",
    "DLL",
    "MultiSet",
    "SortedList",
}


def _find_class_name(code: str) -> str | None:
    """Return the main user class name — prefers Solution, then the last non-helper."""
    names = re.findall(r"^\s*class\s+(\w+)", code, re.MULTILINE)
    if "Solution" in names:
        return "Solution"
    candidates = [n for n in names if n not in _HELPER_CLASSES]
    return candidates[-1] if candidates else (names[-1] if names else None)


def build_runnable_script(entry: dict) -> str | None:
    """
    Reproduce EffiLearner's add_string_to_py_file() for the EffiBench dataset.
    Returns None if the entry can't be turned into a runnable script.
    """
    completion = entry.get("completion") or ""
    test_case = entry.get("test_case") or ""
    if not completion:
        return None

    completion = _strip_markdown_fences(completion).strip()

    # Cap test cases at 100 lines, matching add_string_to_py_file().
    test_lines = test_case.split("\n")[:100]
    test_case = "\n".join(test_lines)

    cls_name = _find_class_name(completion)

    # Check if the test harness creates its own solution variable.
    first_test_line = test_case.strip().split("\n")[0].strip()
    tests_create_solution = first_test_line.startswith("solution =") or first_test_line.startswith("solution=")

    glue = ""
    if cls_name == "Solution":
        if not tests_create_solution:
            glue = "\nsolution = Solution()\n"
    elif cls_name is not None:
        # Non-Solution class (e.g. MedianFinder, TreeAncestor, etc.)
        if not tests_create_solution:
            glue = f"\nSolution = {cls_name}\nsolution = Solution()\n"
        else:
            glue = f"\nSolution = {cls_name}\n"
    else:
        # No class found — function-only code; just concatenate.
        if not tests_create_solution:
            glue = "\n"

    return IMPORT_PKG + "\n" + TREE_NODE_TEXT + "\n" + LIST_NODE_TEXT + "\n" + completion + glue + test_case


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
    _METRION_LOG_FILES[proc.pid] = log_file
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
    log_file = _METRION_LOG_FILES.pop(proc.pid, None)
    if log_file is not None:
        log_file.close()


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def run_script(
    script_text: str,
    timeout_seconds: float | None = None,
    output_path: Path | None = None,
) -> tuple[int, datetime, datetime, int | None, bool]:
    """Write script to a temp file, execute it, return (pid, start, end, returncode, timed_out)."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", prefix="metrion_effibench_", delete=False, mode="w", encoding="utf-8"
    ) as f:
        f.write(script_text)
        script_path = f.name

    output_stream: TextIO | None = None
    try:
        if output_path is not None:
            output_stream = output_path.open("a", encoding="utf-8")
        start_time = datetime.now(UTC)
        stdout_target = output_stream if output_stream is not None else subprocess.DEVNULL
        stderr_target = output_stream if output_stream is not None else subprocess.DEVNULL
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=stdout_target,
            stderr=stderr_target,
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
            _terminate_process_group(proc)
        end_time = datetime.now(UTC)
        return pid, start_time, end_time, proc.returncode, timed_out
    finally:
        if output_stream is not None:
            output_stream.close()
        os.unlink(script_path)


def _terminate_process_group(proc: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate a process group, escalating from SIGTERM to SIGKILL if needed."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return

    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


def _write_execution_log_marker(output_stream: TextIO | None, marker: str) -> None:
    """Write a timestamped marker line to the execution output log."""
    if output_stream is None:
        return

    ts = datetime.now(UTC).isoformat()
    output_stream.write(f"\n===== [{ts}] {marker} =====\n")
    output_stream.flush()


def _write_execution_log_marker_threadsafe(
    output_path: Path | None,
    marker: str,
    marker_lock: _LockLike | None,
) -> None:
    """Write execution markers under a lock so marker boundaries remain readable."""
    if output_path is None:
        return

    if marker_lock is None:
        with output_path.open("a", encoding="utf-8") as output_stream:
            _write_execution_log_marker(output_stream, marker)
        return

    marker_lock.acquire()
    try:
        with output_path.open("a", encoding="utf-8") as output_stream:
            _write_execution_log_marker(output_stream, marker)
    finally:
        marker_lock.release()


def _process_log_path(base_log_path: Path | None) -> Path | None:
    """Return a per-process log file path derived from the shared base path."""
    if base_log_path is None:
        return None
    return base_log_path.with_name(f"{base_log_path.stem}.pid{os.getpid()}{base_log_path.suffix}")


def _execute_entry(
    input_file: Path,
    entry_idx: int,
    total_entries: int,
    entry: dict,
    num_runs: int,
    same_pid_mode: bool,
    entry_timeout: float,
    execution_log_path: Path | None,
    marker_lock: _LockLike | None,
) -> tuple[int, dict, list[dict]]:
    """Execute one entry and return (entry_idx, metadata, flat run records)."""
    # Process-pool workers don't share a lock with the parent process; route
    # them to per-process logs to avoid cross-process interleaving.
    local_log_path = _process_log_path(execution_log_path) if marker_lock is None else execution_log_path

    problem_idx = entry.get("problem_idx")
    task_name = entry.get("task_name", "")
    difficulty = entry.get("difficulty", "")
    meta = {
        "source_entry_index": entry_idx,
        "problem_idx": problem_idx,
        "task_name": task_name,
        "difficulty": difficulty,
    }

    prefix = f"[{entry_idx + 1}/{total_entries}|{input_file.name}] problem={problem_idx}"
    script_text = build_runnable_script(entry)
    if script_text is None:
        logger.info(f"{prefix} … SKIP (no runnable completion)")
        return entry_idx, meta, []

    run_records: list[dict] = []

    if same_pid_mode:
        try:
            _write_execution_log_marker_threadsafe(
                local_log_path,
                f"{prefix} same-pid session start (runs={num_runs}, timeout={entry_timeout})",
                marker_lock,
            )
            session_pid, session_summary = run_script_same_pid(
                script_text,
                num_runs,
                entry_timeout,
                output_path=local_log_path,
            )
            runs_timed_out = session_summary.get("runs_timed_out", 0)
            if runs_timed_out > 0:
                logger.warning(f"{prefix} … {runs_timed_out}/{num_runs} runs timed out")
            _write_execution_log_marker_threadsafe(
                local_log_path,
                (
                    f"{prefix} same-pid session end (rc={session_summary.get('returncode')}, "
                    f"runs_ok={session_summary.get('runs_ok', 0)}, runs_timed_out={runs_timed_out})"
                ),
                marker_lock,
            )

            run_records.append(
                {
                    "input_file": str(input_file),
                    "entry_idx": entry_idx,
                    "problem_idx": problem_idx,
                    "pid": session_pid,
                    "all_pids": [session_pid] if session_pid is not None else [],
                    "returncode": session_summary.get("returncode"),
                    "runs_total": session_summary.get("runs_total", num_runs),
                    "runs_ok": session_summary.get("runs_ok", 0),
                    "runs_timed_out": runs_timed_out,
                    "start_time": session_summary.get("start_time"),
                    "end_time": session_summary.get("end_time"),
                    "duration_s": session_summary.get("duration_s"),
                    "energy": {},
                    "performance": {},
                    "timed_out": bool(session_summary.get("timed_out", False)),
                    "error": session_summary.get("error", ""),
                }
            )
        except Exception as exc:
            logger.error(f"{prefix} ERROR in same-PID session: {exc}")
            run_records.append(
                {
                    "input_file": str(input_file),
                    "entry_idx": entry_idx,
                    "problem_idx": problem_idx,
                    "pid": None,
                    "all_pids": [],
                    "returncode": None,
                    "runs_total": num_runs,
                    "runs_ok": 0,
                    "runs_timed_out": 0,
                    "start_time": None,
                    "end_time": None,
                    "duration_s": None,
                    "energy": {},
                    "performance": {},
                    "timed_out": False,
                    "error": str(exc),
                }
            )
    else:
        for run_idx in range(num_runs):
            run_label = f"run {run_idx + 1}/{num_runs}" if num_runs > 1 else ""
            label_for_markers = run_label or "run 1/1"
            try:
                _write_execution_log_marker_threadsafe(
                    local_log_path,
                    f"{prefix} {label_for_markers} start (timeout={entry_timeout})",
                    marker_lock,
                )
                pid, start_time, end_time, returncode, timed_out = run_script(
                    script_text,
                    entry_timeout,
                    output_path=local_log_path,
                )
                _write_execution_log_marker_threadsafe(
                    local_log_path,
                    (f"{prefix} {label_for_markers} end (pid={pid}, rc={returncode}, timed_out={timed_out})"),
                    marker_lock,
                )
                if timed_out:
                    logger.warning(f"{prefix} {run_label} TIMEOUT (rc={returncode})")
                run_records.append(
                    {
                        "input_file": str(input_file),
                        "entry_idx": entry_idx,
                        "problem_idx": problem_idx,
                        "run_idx": run_idx,
                        "pid": pid,
                        "all_pids": [pid],
                        "returncode": returncode,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "energy": {},
                        "performance": {},
                        "timed_out": timed_out,
                        "error": f"execution_timeout: exceeded {entry_timeout}s" if timed_out else "",
                    }
                )
            except Exception as exc:
                logger.error(f"{prefix} {run_label} ERROR: {exc}")
                run_records.append(
                    {
                        "input_file": str(input_file),
                        "entry_idx": entry_idx,
                        "problem_idx": problem_idx,
                        "run_idx": run_idx,
                        "pid": None,
                        "all_pids": [],
                        "returncode": None,
                        "start_time": None,
                        "end_time": None,
                        "energy": {},
                        "performance": {},
                        "timed_out": False,
                        "error": str(exc),
                    }
                )

    return entry_idx, meta, run_records


_SAME_PID_SESSION_RUNNER = """#!/usr/bin/env python3
import argparse
import json
import signal
from datetime import UTC, datetime
from pathlib import Path
from loguru import logger


class IterationTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise IterationTimeout()


def _write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary), encoding="utf-8")


def main() -> int:
    session_start: datetime.now(UTC)
    p = argparse.ArgumentParser()
    p.add_argument("--script-path", required=True)
    p.add_argument("--meta-path", required=True)
    p.add_argument("--runs", type=int, required=True)
    p.add_argument("--timeout-seconds", type=float, default=0.0)
    args = p.parse_args()

    script_path = Path(args.script_path)
    meta_path = Path(args.meta_path)
    script_code = script_path.read_text(encoding="utf-8")
    compiled = compile(script_code, str(script_path), "exec")

    # Keep a single globals dict so state can persist across iterations.
    session_globals = {"__name__": "__main__", "__file__": str(script_path)}
    timeout = args.timeout_seconds if args.timeout_seconds > 0 else None

    runs_ok = 0
    runs_timed_out = 0
    error_messages: list[str] = []
    session_end: datetime | None = None

    for run_idx in range(args.runs):
        # logger.debug(f"[runner] Starting iteration {run_idx + 1}/{args.runs} …")
        start = datetime.now(UTC)
        if session_start is None:
            session_start = start

        timed_out = False
        returncode: int | None = 0
        error = ""
        old_handler = None

        try:
            if timeout is not None:
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, _on_alarm)
                signal.setitimer(signal.ITIMER_REAL, timeout)
            exec(compiled, session_globals, session_globals)
        except IterationTimeout:
            timed_out = True
            returncode = 124
            error = f"execution_timeout: exceeded {timeout}s"
            logger.error(f"[runner] Iteration {run_idx + 1}/{args.runs} timed out: {error}")
        except SystemExit as exc:
            if isinstance(exc.code, int):
                returncode = exc.code
            elif exc.code is None:
                returncode = 0
            else:
                returncode = 1
        except BaseException as exc:
            returncode = 1
            error = f"{exc.__class__.__name__}: {exc}"
            logger.error(f"[runner] Iteration {run_idx + 1}/{args.runs} failed: {error}")
        finally:
            if timeout is not None:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        end = datetime.now(UTC)
        session_end = end
        if returncode == 0:
            runs_ok += 1
        if timed_out:
            runs_timed_out += 1
        if error:
            error_messages.append(error)

        if timed_out:
            # Abort the remaining iterations after the first timeout to avoid
            # continuing in potentially corrupted in-process state.
            pending_count = args.runs - (run_idx + 1)
            if pending_count > 0:
                runs_timed_out += pending_count
                error_messages.append("session_aborted_after_timeout")
            break

    if session_start is not None and session_end is not None:
        start_time = session_start.isoformat()
        end_time = session_end.isoformat()
        duration_s = max(0.0, (session_end - session_start).total_seconds())
    else:
        start_time = None
        end_time = None
        duration_s = None

    timed_out_any = runs_timed_out > 0
    session_returncode = 0 if runs_ok == args.runs else (124 if timed_out_any else 1)
    error_summary = "; ".join(dict.fromkeys(error_messages))

    summary = {
        "runs_total": args.runs,
        "runs_ok": runs_ok,
        "runs_timed_out": runs_timed_out,
        "start_time": start_time,
        "end_time": end_time,
        "duration_s": duration_s,
        "timed_out": timed_out_any,
        "returncode": session_returncode,
        "error": error_summary,
    }
    _write_summary(meta_path, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _load_session_summary(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}

    try:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[runner] Failed to parse session sidecar {meta_path}: {exc}")
        return {}

    if not isinstance(loaded, dict):
        return {}
    return loaded


def run_script_same_pid(
    script_text: str,
    runs: int,
    timeout_seconds: float | None = None,
    output_path: Path | None = None,
) -> tuple[int | None, dict]:
    """Run one entry multiple times inside the same Python process PID."""
    if runs <= 0:
        return None, {
            "runs_total": 0,
            "runs_ok": 0,
            "runs_timed_out": 0,
            "start_time": None,
            "end_time": None,
            "duration_s": None,
            "timed_out": False,
            "returncode": None,
            "error": "",
        }

    with tempfile.NamedTemporaryFile(
        suffix=".py", prefix="metrion_effibench_entry_", delete=False, mode="w", encoding="utf-8"
    ) as script_file:
        script_file.write(script_text)
        script_path = Path(script_file.name)

    with tempfile.NamedTemporaryFile(
        suffix=".py", prefix="metrion_effibench_session_runner_", delete=False, mode="w", encoding="utf-8"
    ) as runner_file:
        runner_file.write(_SAME_PID_SESSION_RUNNER)
        runner_path = Path(runner_file.name)

    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="metrion_effibench_session_meta_", delete=False, mode="w", encoding="utf-8"
    ) as meta_file:
        meta_file.write("{}")
        meta_path = Path(meta_file.name)

    session_pid: int | None = None
    session_timed_out = False
    output_stream: TextIO | None = None
    try:
        cmd = [
            sys.executable,
            str(runner_path),
            "--script-path",
            str(script_path),
            "--meta-path",
            str(meta_path),
            "--runs",
            str(runs),
            "--timeout-seconds",
            str(timeout_seconds if timeout_seconds is not None else 0),
        ]
        if output_path is not None:
            output_stream = output_path.open("a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            stdout=output_stream if output_stream is not None else subprocess.DEVNULL,
            stderr=output_stream if output_stream is not None else subprocess.DEVNULL,
            start_new_session=True,
        )
        session_pid = proc.pid

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                # Give the outer session some headroom over the per-iteration timeout.
                total_timeout = timeout_seconds * runs + 10.0
                proc.wait(timeout=total_timeout)
            else:
                proc.wait()
        except subprocess.TimeoutExpired:
            session_timed_out = True
            logger.warning(
                f"[runner] Same-PID session PID={session_pid} exceeded aggregate timeout; terminating process group."
            )
            _terminate_process_group(proc)

        raw_summary = _load_session_summary(meta_path)

        runs_ok_raw = raw_summary.get("runs_ok")
        if isinstance(runs_ok_raw, int):
            runs_ok = max(0, min(runs_ok_raw, runs))
        else:
            runs_ok = 0

        runs_timed_out_raw = raw_summary.get("runs_timed_out")
        if isinstance(runs_timed_out_raw, int):
            runs_timed_out = max(0, min(runs_timed_out_raw, runs))
        else:
            runs_timed_out = 0

        start_time = raw_summary.get("start_time")
        if not isinstance(start_time, str):
            start_time = None

        end_time = raw_summary.get("end_time")
        if not isinstance(end_time, str):
            end_time = None

        duration_s_raw = raw_summary.get("duration_s")
        if isinstance(duration_s_raw, (int | float)):
            duration_s = max(0.0, float(duration_s_raw))
        else:
            duration_s = None

        timed_out = bool(raw_summary.get("timed_out", False) or session_timed_out)
        if timed_out and runs_timed_out == 0:
            runs_timed_out = max(1, runs - runs_ok)

        error = raw_summary.get("error") if isinstance(raw_summary.get("error"), str) else ""
        if session_timed_out and not error:
            error = "session_terminated_before_completion"

        returncode_raw = raw_summary.get("returncode")
        if isinstance(returncode_raw, int):
            returncode = returncode_raw
        else:
            returncode = 0 if runs_ok == runs else (124 if timed_out else 1)

        summary = {
            "runs_total": runs,
            "runs_ok": runs_ok,
            "runs_timed_out": runs_timed_out,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "timed_out": timed_out,
            "returncode": returncode,
            "error": error,
        }

        return session_pid, summary
    finally:
        if output_stream is not None:
            output_stream.close()
        for path in (script_path, runner_path, meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


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
            "cpu_core_pid_work": None,
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
        "cpu_core_pid_work": _col_sum("cpu_core_pid_work"),
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

        # logger.info(
        #     f"[db] problem={rec['problem_idx']} pids={rec['all_pids']} "
        #     f"cpu_energy={rec['energy'].get('cpu_total_energy_j')} J "
        #     f"cycles={rec['performance'].get('unhalted_cycles')}"
        # )


def _record_time_bounds(rec: dict) -> tuple[datetime, datetime] | None:
    start_raw = rec.get("start_time")
    end_raw = rec.get("end_time")
    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        return None

    try:
        start_dt = datetime.fromisoformat(start_raw)
        end_dt = datetime.fromisoformat(end_raw)
    except ValueError:
        return None

    if end_dt < start_dt:
        end_dt = start_dt
    return start_dt, end_dt


def _run_analysis_same_pid(records: list[dict], db_path: Path, db_type: str) -> None:
    """Phase D for same-PID mode: compute parent-session totals per entry."""
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

    try:
        active_records = [rec for rec in records if rec.get("pid") is not None and _record_time_bounds(rec) is not None]
        if not active_records:
            return

        # Resolve descendant threads once per unique session PID.
        seed_pids = sorted({int(rec["pid"]) for rec in active_records})
        logger.info(f"[db] [same-pid] Resolving descendant threads for {len(seed_pids)} unique seed PIDs …")
        threads_df = get_descendant_threads(engine, seed_pids)

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

        record_bounds = [bounds for rec in active_records if (bounds := _record_time_bounds(rec)) is not None]
        if not record_bounds:
            return

        time_pad = timedelta(days=1)
        global_start = min(start for start, _ in record_bounds) - time_pad
        global_end = max(end for _, end in record_bounds) + time_pad

        logger.info(f"[db] [same-pid] Bulk energy query for {len(all_pids)} PIDs …")
        try:
            energy_df: pl.DataFrame | None = get_energy_for_pids(engine, all_pids, global_start, global_end)
        except Exception as exc:
            logger.warning(f"[db] [same-pid] get_energy_for_pids (bulk) failed: {exc}")
            energy_df = None

        logger.info(f"[db] [same-pid] Bulk perf query for {len(all_pids)} PIDs …")
        try:
            perf_df: pl.DataFrame | None = get_performance_metrics_for_pids(engine, all_pids, global_start, global_end)
        except Exception as exc:
            logger.warning(f"[db] [same-pid] get_performance_metrics_for_pids (bulk) failed: {exc}")
            perf_df = None

        for rec in active_records:
            pid_set = rec.get("all_pids", [])

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
                f"[db] [same-pid] problem={rec.get('problem_idx')} pid={rec.get('pid')} "
                f"cpu_energy={rec['energy'].get('cpu_total_energy_j')} "
                f"cycles={rec['performance'].get('unhalted_cycles')}"
            )
    finally:
        close_db_connection(engine)


# ---------------------------------------------------------------------------
# Multi-run aggregation helpers
# ---------------------------------------------------------------------------

ENERGY_KEYS = [
    "cpu_total_energy_j",
    "cpu_active_energy_j",
    "cpu_idle_energy_j",
    "dram_total_energy_j",
    "dram_active_energy_j",
    "dram_idle_energy_j",
    "gpu_total_energy_j",
    "gpu_active_energy_j",
    "gpu_idle_energy_j",
]

PERFORMANCE_KEYS = [
    "cpu_core_pid_work",
    "unhalted_cycles",
    "kernel_time_ms",
    "dram_memory_mb",
    "gpu_memory_mb",
]


def _aggregate_metric_values(values: list[object]) -> dict:
    """Return sum/mean/stdev/n for numeric values, or Nones when unavailable."""
    numeric_values = np.fromiter(
        (float(v) for v in values if isinstance(v, (int | float | np.number))),
        dtype=np.float64,
    )
    if numeric_values.size == 0:
        return {
            "sum": None,
            "mean": None,
            "stdev": None,
            "n": 0,
        }

    if numeric_values.size >= 2:
        numeric_values = numeric_values[numeric_values > 0.0]

    if numeric_values.size == 0:
        return {
            "sum": None,
            "mean": None,
            "stdev": None,
            "n": 0,
        }

    if numeric_values.size > 2:
        # Keep sample stdev semantics (ddof=1) while avoiding unstable tiny samples.
        stdev = float(np.std(numeric_values, ddof=1))
    else:
        stdev = 0.0

    return {
        "sum": float(np.sum(numeric_values, dtype=np.float64)),
        "mean": float(np.mean(numeric_values, dtype=np.float64)),
        "stdev": stdev,
        "n": int(numeric_values.size),
    }


def _aggregate_runs(run_records: list[dict]) -> dict:
    """Compute totals and summary stats across run-level energy/performance."""
    per_run_energy = [r.get("energy", {}) for r in run_records]
    per_run_performance = [r.get("performance", {}) for r in run_records]

    energy_totals: dict[str, float | None] = {}
    energy_stats: dict[str, dict[str, float | int | None]] = {}
    for key in ENERGY_KEYS:
        summary = _aggregate_metric_values([energy.get(key) for energy in per_run_energy])
        energy_totals[key] = summary["sum"]
        energy_stats[key] = {
            "mean": summary["mean"],
            "stdev": summary["stdev"],
            "n": summary["n"],
        }

    performance_totals: dict[str, float | None] = {}
    performance_stats: dict[str, dict[str, float | int | None]] = {}
    for key in PERFORMANCE_KEYS:
        summary = _aggregate_metric_values([perf.get(key) for perf in per_run_performance])
        performance_totals[key] = summary["sum"]
        performance_stats[key] = {
            "mean": summary["mean"],
            "stdev": summary["stdev"],
            "n": summary["n"],
        }

    return {
        "energy_totals": energy_totals,
        "energy_stats": energy_stats,
        "performance_totals": performance_totals,
        "performance_stats": performance_stats,
        "per_run_energy": per_run_energy,
    }


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def _out_path(input_file: Path) -> Path:
    """Derive the output JSON path: {stem}_energy_results.json in same directory."""
    return input_file.with_stem(input_file.stem + "_energy_results")


def _pids_path(input_file: Path) -> Path:
    """Derive the intermediate pids JSON path: {stem}_energy_results.pids.json."""
    return _out_path(input_file).with_suffix(".pids.json")


def _entry_metadata(entries: list[dict]) -> list[dict]:
    """Build per-entry metadata for final output records."""
    return [
        {
            "source_entry_index": i,
            "problem_idx": entry.get("problem_idx"),
            "task_name": entry.get("task_name", ""),
            "difficulty": entry.get("difficulty", ""),
        }
        for i, entry in enumerate(entries)
    ]


def _entry_run_map_from_flat_records(
    flat_records: list[dict],
    total_entries: int,
    input_file: Path,
) -> dict[int, list[int]]:
    """Group flat run records by entry index for one input file."""
    entry_run_map: dict[int, list[int]] = {}
    for ri, rec in enumerate(flat_records):
        entry_idx_raw = rec.get("entry_idx")
        if not isinstance(entry_idx_raw, int):
            logger.warning(
                f"[analyze] {input_file.name}: skipping record {ri} with non-integer entry_idx={entry_idx_raw!r}"
            )
            continue
        if entry_idx_raw < 0 or entry_idx_raw >= total_entries:
            logger.warning(
                f"[analyze] {input_file.name}: skipping record {ri} with out-of-range entry_idx={entry_idx_raw}"
            )
            continue
        entry_run_map.setdefault(entry_idx_raw, []).append(ri)
    return entry_run_map


def _records_look_like_same_pid_mode(records: list[dict]) -> bool:
    """Best-effort detection of same-PID pids files when CLI flags are omitted."""
    if not records:
        return False

    same_pid_markers = 0
    run_idx_markers = 0
    for rec in records:
        if any(k in rec for k in ("runs_total", "runs_ok", "runs_timed_out", "duration_s")):
            same_pid_markers += 1
        if "run_idx" in rec:
            run_idx_markers += 1

    return same_pid_markers > 0 and run_idx_markers == 0


def _infer_num_runs_from_flat_records(records: list[dict], same_pid_mode: bool) -> int:
    """Infer the configured run count from pids records."""
    if not records:
        return 1

    if same_pid_mode:
        runs_total_values = [rec.get("runs_total") for rec in records if isinstance(rec.get("runs_total"), int)]
        if runs_total_values:
            return max(1, max(runs_total_values))
        return 1

    per_entry_counts: dict[int, int] = {}
    for rec in records:
        entry_idx = rec.get("entry_idx")
        if isinstance(entry_idx, int) and entry_idx >= 0:
            per_entry_counts[entry_idx] = per_entry_counts.get(entry_idx, 0) + 1
    if per_entry_counts:
        return max(1, max(per_entry_counts.values()))

    run_idx_values = [rec.get("run_idx") for rec in records if isinstance(rec.get("run_idx"), int)]
    if run_idx_values:
        return max(1, max(run_idx_values) + 1)

    return 1


def _build_output_records(
    entry_meta_list: list[dict],
    entry_run_map: dict[int, list[int]],
    run_records_flat: list[dict],
    same_pid_mode: bool,
    num_runs: int,
    entry_timeout: float,
) -> list[dict]:
    """Build final per-entry output records from analyzed flat run records."""
    output_records: list[dict] = []
    for i, meta in enumerate(entry_meta_list):
        run_indices = entry_run_map.get(i, [])

        if not run_indices:
            # Entry was skipped (no runnable completion)
            output_records.append(
                {
                    **meta,
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

        run_recs = [run_records_flat[ri] for ri in run_indices]
        all_pids_across_runs = sorted(
            {
                pid
                for run_rec in run_recs
                for pid in [run_rec.get("pid"), *(run_rec.get("all_pids") or [])]
                if isinstance(pid, int)
            }
        )

        if same_pid_mode:
            parent = run_recs[0]
            output_records.append(
                {
                    **meta,
                    "pid": None,
                    "all_pids": all_pids_across_runs,
                    "returncode": parent.get("returncode"),
                    "runs_total": parent.get("runs_total", num_runs),
                    "runs_ok": parent.get("runs_ok", 0),
                    "runs_timed_out": parent.get("runs_timed_out", 0),
                    "start_time": parent.get("start_time"),
                    "end_time": parent.get("end_time"),
                    "duration_s": parent.get("duration_s"),
                    "energy": parent.get("energy", {}),
                    "performance": parent.get("performance", {}),
                    "error": parent.get("error", ""),
                }
            )
            continue

        if num_runs == 1:
            # Single-run mode: backward-compatible output format.
            r = run_recs[0]
            output_records.append(
                {
                    **meta,
                    "pid": None,
                    "all_pids": all_pids_across_runs,
                    "returncode": r["returncode"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "energy": r["energy"],
                    "performance": r["performance"],
                    "error": r.get("error")
                    or (f"execution_timeout: exceeded {entry_timeout}s" if r.get("timed_out") else ""),
                }
            )
        else:
            # Multi-run mode: aggregate energy/performance across runs.
            ok_runs = [r for r in run_recs if r.get("returncode") == 0]
            agg = _aggregate_runs(run_recs)
            error_msgs = [r.get("error", "") for r in run_recs if isinstance(r.get("error"), str) and r.get("error")]
            error_summary = "; ".join(dict.fromkeys(error_msgs))

            # Use last successful run's returncode/times as representative.
            rep = ok_runs[-1] if ok_runs else run_recs[-1]
            output_records.append(
                {
                    **meta,
                    "pid": None,
                    "all_pids": all_pids_across_runs,
                    "returncode": rep["returncode"],
                    "start_time": rep["start_time"],
                    "end_time": rep["end_time"],
                    "runs_total": len(run_recs),
                    "runs_ok": len(ok_runs),
                    "energy": agg["energy_totals"],
                    "energy_stats": agg["energy_stats"],
                    "per_run_energy": agg["per_run_energy"],
                    "performance": agg["performance_totals"],
                    "performance_stats": agg["performance_stats"],
                    "error": error_summary,
                }
            )

    return output_records


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
        "--runs",
        type=int,
        default=1,
        help="Number of independent runs per entry. Each run is a separate "
        "subprocess; energy mean and stdev are computed across runs (default: 1).",
    )
    p.add_argument(
        "--same-pid-runs",
        action="store_true",
        help="Run all iterations of each entry in a single long-lived Python process (same PID). "
        "Shared interpreter state is preserved and output reports parent-session totals.",
    )
    p.add_argument(
        "--entry-timeout",
        type=float,
        default=120.0,
        help="Max runtime in seconds per entry script. Use 0 or negative to disable (default: 120).",
    )
    p.add_argument(
        "--max-parallel-entries",
        type=int,
        default=1,
        help="Maximum number of entries to execute concurrently within each input file (default: 1).",
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

    if not args.analyze_only and args.same_pid_runs and args.runs <= 1:
        logger.warning("[args] --same-pid-runs requires --runs > 1; falling back to default per-run subprocess mode.")
        args.same_pid_runs = False
    if args.max_parallel_entries < 1:
        logger.warning("[args] --max-parallel-entries must be >= 1; falling back to 1.")
        args.max_parallel_entries = 1

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
            if not input_file.exists():
                logger.error(f"[analyze] Input file not found: {input_file}")
                continue

            entries: list[dict] = json.loads(input_file.read_text())
            if args.limit is not None:
                entries = entries[: args.limit]
            entry_meta_list = _entry_metadata(entries)

            records: list[dict] = json.loads(pids_file.read_text())
            logger.info(f"[analyze] Loaded {len(records)} records from {pids_file.name}")
            inferred_same_pid_mode = _records_look_like_same_pid_mode(records)
            same_pid_mode = args.same_pid_runs or inferred_same_pid_mode
            if inferred_same_pid_mode and not args.same_pid_runs:
                logger.info(
                    "[analyze] Inferred same-PID records from pids file; applying same-PID analysis mode "
                    "(override with --same-pid-runs to make this explicit)."
                )

            inferred_num_runs = _infer_num_runs_from_flat_records(records, same_pid_mode)
            num_runs = inferred_num_runs if args.runs == 1 else args.runs
            if args.runs != 1 and args.runs != inferred_num_runs:
                logger.warning(
                    f"[analyze] --runs={args.runs} differs from inferred runs={inferred_num_runs} from pids file; "
                    f"using --runs={args.runs}."
                )

            if same_pid_mode:
                _run_analysis_same_pid(records, db_path, db_type)
            else:
                _run_analysis(records, db_path, db_type)

            entry_run_map = _entry_run_map_from_flat_records(records, len(entry_meta_list), input_file)
            output_records = _build_output_records(
                entry_meta_list,
                entry_run_map,
                records,
                same_pid_mode,
                num_runs,
                args.entry_timeout,
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(output_records, indent=2, default=str))
            logger.info(
                f"Results written to {out_file} "
                f"({len(output_records)} entries, "
                f"inferred runs/entry={inferred_num_runs}, output runs/entry={num_runs})"
            )
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
    num_runs = args.runs
    same_pid_mode = bool(args.same_pid_runs and num_runs > 1)
    if same_pid_mode:
        logger.info("[runner] Same-PID mode enabled: each entry will run all iterations inside one process PID.")

    execution_timestamp = datetime.now(UTC).strftime("%Y_%m_%d_%H_%M_%S")
    execution_log_path = args.log_dir / f"execution_{execution_timestamp}.log"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    execution_log_marker_lock = threading.Lock()
    logger.info(f"[runner] Logging all script stdout/stderr to: {execution_log_path}")

    # For default mode, run_records_flat has one record per subprocess run.
    # For same-PID mode, run_records_flat has one parent-session record per
    # entry (covering all in-process sub-runs).
    # run_records_flat: every individual execution across all files.
    # file_entry_map: per input file, per entry index → list of flat-record indices.
    run_records_flat: list[dict] = []
    file_entry_map: dict[str, dict[int, list[int]]] = {}
    # We also keep per-file entry metadata for the final output.
    file_entry_meta: dict[str, list[dict]] = {}

    for input_file, entries in tqdm(
        file_entries,
        desc="Processing input files",
        unit="file",
        file=sys.stdout,
    ):
        fkey = str(input_file)
        file_entry_map[fkey] = {}
        total = len(entries)
        file_entry_meta[fkey] = [{} for _ in range(total)]
        max_workers = min(args.max_parallel_entries, total) if total > 0 else 1

        if max_workers > 1:
            logger.info(
                f"[runner] {input_file.name}: executing up to {max_workers} entries in parallel "
                f"(--max-parallel-entries={args.max_parallel_entries})."
            )
            logger.info(
                f"[runner] {input_file.name}: worker logs use per-process files named "
                f"{execution_log_path.stem}.pid<process_id>{execution_log_path.suffix}"
            )
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(
                        _execute_entry,
                        input_file,
                        i,
                        total,
                        entry,
                        num_runs,
                        same_pid_mode,
                        args.entry_timeout,
                        execution_log_path,
                        None,
                    ): i
                    for i, entry in enumerate(entries)
                }

                for future in tqdm(
                    as_completed(future_to_idx),
                    total=total,
                    desc=f"Entries in {input_file.name}",
                    unit="entry",
                    file=sys.stdout,
                    leave=False,
                ):
                    i = future_to_idx[future]
                    try:
                        entry_idx, entry_meta, entry_run_records = future.result()
                    except Exception as exc:
                        entry = entries[i]
                        problem_idx = entry.get("problem_idx")
                        logger.error(
                            f"[{i + 1}/{total}|{input_file.name}] problem={problem_idx} ERROR in entry worker: {exc}"
                        )
                        entry_idx = i
                        entry_meta = {
                            "source_entry_index": i,
                            "problem_idx": problem_idx,
                            "task_name": entry.get("task_name", ""),
                            "difficulty": entry.get("difficulty", ""),
                        }
                        entry_run_records = [
                            {
                                "input_file": str(input_file),
                                "entry_idx": i,
                                "problem_idx": problem_idx,
                                "pid": None,
                                "all_pids": [],
                                "returncode": None,
                                "start_time": None,
                                "end_time": None,
                                "energy": {},
                                "performance": {},
                                "timed_out": False,
                                "error": str(exc),
                            }
                        ]

                    file_entry_meta[fkey][entry_idx] = entry_meta
                    run_indices: list[int] = []
                    for rec in entry_run_records:
                        flat_idx = len(run_records_flat)
                        run_records_flat.append(rec)
                        run_indices.append(flat_idx)
                    file_entry_map[fkey][entry_idx] = run_indices
        else:
            for i, entry in tqdm(
                enumerate(entries),
                total=total,
                desc=f"Entries in {input_file.name}",
                unit="entry",
                file=sys.stdout,
                leave=False,
            ):
                entry_idx, entry_meta, entry_run_records = _execute_entry(
                    input_file,
                    i,
                    total,
                    entry,
                    num_runs,
                    same_pid_mode,
                    args.entry_timeout,
                    execution_log_path,
                    execution_log_marker_lock,
                )
                file_entry_meta[fkey][entry_idx] = entry_meta

                run_indices: list[int] = []
                for rec in entry_run_records:
                    flat_idx = len(run_records_flat)
                    run_records_flat.append(rec)
                    run_indices.append(flat_idx)
                file_entry_map[fkey][entry_idx] = run_indices

        # Save intermediate pids file for this input before moving to the next.
        # In multi-run mode, the pids file contains all flat run records for this file.
        pids_file = _pids_path(input_file)
        pids_file.parent.mkdir(parents=True, exist_ok=True)
        file_flat_indices = [ri for entry_idx in range(total) for ri in file_entry_map[fkey].get(entry_idx, [])]
        file_flat_records = [run_records_flat[ri] for ri in file_flat_indices]
        pids_file.write_text(json.dumps(file_flat_records, indent=2, default=str))
        logger.info(f"[pids] Intermediate records saved to {pids_file} ({len(file_flat_records)} run records)")

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

    # Run DB analysis on all flat run records at once.
    if db_available:
        if same_pid_mode:
            _run_analysis_same_pid(run_records_flat, db_path, db_type)
        else:
            _run_analysis(run_records_flat, db_path, db_type)

    # --- Phase E: aggregate runs and write one output file per input ---
    for input_file, _entries in file_entries:
        fkey = str(input_file)
        entry_meta_list = file_entry_meta[fkey]
        entry_run_map = file_entry_map[fkey]

        output_records = _build_output_records(
            entry_meta_list,
            entry_run_map,
            run_records_flat,
            same_pid_mode,
            num_runs,
            args.entry_timeout,
        )

        out_file = _out_path(input_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(output_records, indent=2, default=str))
        logger.info(f"Results written to {out_file} ({len(output_records)} entries, {num_runs} runs/entry)")


if __name__ == "__main__":
    main()
