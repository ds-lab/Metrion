"""Diagnose *where* METRION hangs on shutdown.

Runs the monitor, sends SIGINT, and if the process tree has not exited within a
deadline it makes every Python process in the tree dump the stacks of all its
threads. That covers the parts of METRION that are hard to reason about from the
outside: the DataSaver process, the eBPF ring-buffer processes, and the
DataSourceRunner threads.

It works by injecting a `sitecustomize.py` that registers
`faulthandler.register(SIGUSR1, all_threads=True)` in every interpreter that
starts, including the multiprocessing children (METRION uses the "spawn" start
method, so children re-import site and pick this up too).

Usage (as root, because METRION needs eBPF/RAPL):

    sudo $(which python) bench/shutdown_watchdog.py \
        --cmd "$(which metrion) monitor" \
        --run-seconds 60 --deadline 20

    # or against a config with everything enabled
    sudo $(which python) bench/shutdown_watchdog.py \
        --cmd "$(which metrion) monitor -c metrion/config.yaml" \
        --run-seconds 300 --deadline 30 --dump-dir ./shutdown-dumps

The output is a timeline plus, on timeout, one stack dump file per process. The
frame at the top of each dump is the thing that is hanging.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SITECUSTOMIZE = """\
import faulthandler, os, signal, sys

_dump_dir = os.environ.get("METRION_WATCHDOG_DUMP_DIR")
if _dump_dir:
    try:
        os.makedirs(_dump_dir, exist_ok=True)
        _path = os.path.join(_dump_dir, "stacks_pid%d.txt" % os.getpid())
        _fh = open(_path, "w")
        faulthandler.register(signal.SIGUSR1, file=_fh, all_threads=True, chain=False)
        # keep a reference so the file object is not garbage collected
        sys.__metrion_watchdog_file = _fh
    except Exception:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
else:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
"""


def descendants(pid: int) -> list[int]:
    """All descendant pids, via /proc only (no psutil dependency)."""
    children: dict[int, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # ppid is field 4, but comm (field 2) may contain spaces/parens
        tail = stat[stat.rindex(")") + 2 :].split()
        ppid = int(tail[1])
        children.setdefault(ppid, []).append(int(entry.name))

    out: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            out.append(child)
            stack.append(child)
    return out


def describe(pid: int) -> str:
    try:
        cmdline = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        state = (Path("/proc") / str(pid) / "stat").read_text()
        state_char = state[state.rindex(")") + 2]
        return f"pid {pid} [{state_char}] {cmdline.strip()[:90]}"
    except OSError:
        return f"pid {pid} [gone]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="the METRION command to run, quoted")
    parser.add_argument("--run-seconds", type=float, default=60.0, help="how long to monitor before stopping")
    parser.add_argument("--deadline", type=float, default=20.0, help="seconds to allow for graceful shutdown")
    parser.add_argument("--dump-dir", default=None, help="where to write stack dumps (default: temp dir)")
    parser.add_argument("--signal", default="INT", choices=["INT", "TERM"], help="stop signal to send")
    args = parser.parse_args()

    if sys.platform != "linux":
        print("This watchdog reads /proc and needs Linux.", file=sys.stderr)
        return 2

    inject_dir = Path(tempfile.mkdtemp(prefix="metrion_watchdog_"))
    (inject_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)
    dump_dir = Path(args.dump_dir).resolve() if args.dump_dir else inject_dir / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(inject_dir) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["METRION_WATCHDOG_DUMP_DIR"] = str(dump_dir)

    print(f"injecting sitecustomize from {inject_dir}")
    print(f"stack dumps will go to      {dump_dir}")
    print(f"running: {args.cmd}")
    print()

    proc = subprocess.Popen(shlex.split(args.cmd), env=env, start_new_session=True)
    pgid = os.getpgid(proc.pid)

    t0 = time.monotonic()
    print(f"[{0.0:7.2f}s] started, pid {proc.pid} (pgid {pgid})")

    deadline_run = t0 + args.run_seconds
    while time.monotonic() < deadline_run:
        if proc.poll() is not None:
            print(f"[{time.monotonic() - t0:7.2f}s] process exited early with {proc.returncode}")
            return proc.returncode or 0
        time.sleep(0.5)

    tree_before = [proc.pid, *descendants(proc.pid)]
    print(f"[{time.monotonic() - t0:7.2f}s] process tree before shutdown:")
    for pid in tree_before:
        print(f"            {describe(pid)}")

    stop_signal = signal.SIGINT if args.signal == "INT" else signal.SIGTERM
    t_stop = time.monotonic()
    print(f"[{t_stop - t0:7.2f}s] sending SIG{args.signal} to pgid {pgid}")
    os.killpg(pgid, stop_signal)

    hung = False
    while True:
        elapsed = time.monotonic() - t_stop
        if proc.poll() is not None:
            print(f"[{time.monotonic() - t0:7.2f}s] exited {elapsed:.2f}s after SIG{args.signal}, rc={proc.returncode}")
            break
        if elapsed > args.deadline:
            hung = True
            break
        time.sleep(0.25)

    if hung:
        alive = [pid for pid in (proc.pid, *descendants(proc.pid)) if Path(f"/proc/{pid}").exists()]
        print()
        print(f"!!! still alive {args.deadline:.0f}s after SIG{args.signal} — dumping stacks")
        for pid in alive:
            print(f"    {describe(pid)}")
            try:
                os.kill(pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
        time.sleep(2.0)

        print()
        for dump in sorted(dump_dir.glob("stacks_pid*.txt")):
            content = dump.read_text().strip()
            if not content:
                continue
            print("=" * 78)
            print(f"{dump.name}")
            print("=" * 78)
            print(content)
            print()

        print(f"escalating: SIGKILL to pgid {pgid}")
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)

    print()
    print(f"shutdown took {'>' if hung else ''}{min(time.monotonic() - t_stop, args.deadline):.2f}s")
    leftovers = list(Path(tempfile.gettempdir()).glob("batch_*.ipc"))
    if leftovers:
        size = sum(p.stat().st_size for p in leftovers)
        print(
            f"WARNING: {len(leftovers)} orphaned IPC batch files left in {tempfile.gettempdir()} "
            f"({size / 1e6:.1f} MB) — these are measurements that never reached the database"
        )
    return 1 if hung else 0


if __name__ == "__main__":
    raise SystemExit(main())
