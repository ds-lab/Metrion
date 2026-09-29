import argparse
import ctypes as ct
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

from metrion.entity.logical_entity_registry import TaskEvent
from metrion.metrion_rust import BpfProgram
from metrion.monitoring.data_source.cpu.collection_plan import SchedSwitchCollectionPlan
from metrion.monitoring.data_source.cpu.sched_switch import SchedSwitchBpfMonitor
from metrion.monitoring.data_source.cpu.utils import count_cpu_sockets_from_sysfs
from metrion.monitoring.data_source.network.tcp_monitor import TcpEvent

REPO = Path(__file__).resolve().parents[1]
# The last commit that loaded the probes with BCC.
DEFAULT_REF = "a75b6e2"
REFERENCE_SOURCES = [
    "metrion/entity/logical_entity_registry.c",
    "metrion/monitoring/data_source/cpu/sched_switch.c",
    "metrion/monitoring/data_source/network/tcp_monitor.c",
    "metrion/monitoring/data_source/helpers/common_bpf.h",
]
PERF_ARRAYS = ["cycles_core", "aperf_core", "mperf_core", "load_local_perf", "load_remote_perf", "store_perf"]
RING_MIB = 128


def extract_reference(ref: str, folder: Path) -> None:
    """Write the BCC sources of commit ``ref`` into ``folder``."""
    for path in REFERENCE_SOURCES:
        source = subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], check=True, capture_output=True).stdout
        (folder / Path(path).name).write_bytes(source)


def boot_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def check_registry(reference: Path) -> bool:
    """Fork, exec, rename and exit a known workload and compare every field of every event."""
    from bcc import BPF

    fields = [name for name, _ in TaskEvent._fields_ if name != "__pad"]
    # Each program stamps these with its own clock read, so they differ by the gap between the two.
    time_fields = {"tid_start_ns", "tgid_start_ns", "parent_tid_start_ns", "parent_tgid_start_ns"}
    record = lambda event: {name: getattr(event, name) for name in fields}  # noqa: E731

    bcc = BPF(src_file=str(reference / "logical_entity_registry.c"), cflags=[f"-I{reference}"])
    core = BpfProgram("logical_entity_registry")
    events_bcc, events_core = [], []
    bcc["events"].open_ring_buffer(lambda ctx, data, size: events_bcc.append(record(bcc["events"].event(data))) or 0)
    core.open_ring_buffer("events", lambda data: events_core.append(record(TaskEvent.from_buffer_copy(data))) or 0)
    time.sleep(0.2)
    bcc.ring_buffer_consume()
    core.ring_buffer_consume()
    events_bcc.clear()
    events_core.clear()

    for i in range(20):
        subprocess.run(["/bin/sh", "-c", f"sleep 0.01; /bin/true ab_marker_{i}"], check=True)
    threads = [threading.Thread(target=time.sleep, args=(0.05,)) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rename = "import ctypes, time; ctypes.CDLL(None).prctl(15, b'ab_renamed'); time.sleep(0.02)"
    subprocess.run([sys.executable, "-c", rename], check=True)
    time.sleep(0.3)
    bcc.ring_buffer_consume()
    core.ring_buffer_consume()
    bcc.cleanup()

    key = lambda event: (event["event_type"], event["tid"], event["tgid"], event["old_tid"])  # noqa: E731
    print(f"  events: bcc={len(events_bcc)} core={len(events_core)}")
    if Counter(map(key, events_bcc)) != Counter(map(key, events_core)):
        print("  the two programs saw different events")
        return False

    # A task can send several events of one type (a rename at exec and one by prctl), so they are
    # matched in order.
    by_key = {}
    for event in events_core:
        by_key.setdefault(key(event), []).append(event)
    mismatches, max_dt = [], 0
    for event in events_bcc:
        other = by_key[key(event)].pop(0)
        for name in fields:
            if name in time_fields and (event[name] == 0) == (other[name] == 0):
                max_dt = max(max_dt, abs(event[name] - other[name]))
            elif event[name] != other[name]:
                mismatches.append((key(event), name, event[name], other[name]))
    print(f"  largest timestamp difference: {max_dt} ns")
    for mismatch in mismatches[:10]:
        print("  mismatch:", mismatch)
    return not mismatches


def check_sched_switch(reference: Path, all_perf: bool, control: bool) -> bool:
    """Compare both programs' event sequence per CPU, fed by the same perf counters.

    Parameters
    ----------
    reference : Path
        Folder with the BCC sources.
    all_perf : bool
        Open every perf group, not just cycles, APERF and MPERF.
    control : bool
        Run two CO-RE programs against each other instead, to see what differs from running two
        programs on one tracepoint alone.
    """
    num_cpus = psutil.cpu_count(logical=True)
    max_entries = {name: num_cpus for name in PERF_ARRAYS} | {"events": RING_MIB * 1024 * 1024}
    if control:
        first = BpfProgram("sched_switch", dict(max_entries))
    else:
        from bcc import BPF

        first = BPF(
            src_file=str(reference / "sched_switch.c"),
            cflags=[
                f"-DNUM_CPUS={num_cpus}",
                f"-DNUM_SOCKETS={count_cpu_sockets_from_sysfs()}",
                f"-DRINGBUF_PAGES={RING_MIB * 1024 * 1024 // 4096}",
                f"-I{reference}",
            ],
        )
    core = BpfProgram("sched_switch", max_entries)
    loaded = boot_ns()

    # Metrion's own perf setup opens the counters; every fd goes into both programs.
    class Fanout:
        def set_perf_fd(self, name, index, fd):
            core.set_perf_fd(name, index, fd)
            if control:
                first.set_perf_fd(name, index, fd)
            else:
                first[name][first[name].Key(index)] = first[name].Leaf(fd)

    monitor = object.__new__(SchedSwitchBpfMonitor)
    monitor._collection_plan = (
        SchedSwitchCollectionPlan.all_enabled() if all_perf else SchedSwitchCollectionPlan(True, False, False, False)
    )
    monitor._bpf = Fanout()
    monitor._setup_perf_events()

    raw_first, raw_core = [], []
    if control:
        first.open_ring_buffer("events", lambda data: raw_first.append(data) or 0)
    else:
        first["events"].open_ring_buffer(lambda ctx, data, size: raw_first.append(ct.string_at(data, size)) or 0)
    core.open_ring_buffer("events", lambda data: raw_core.append(data) or 0)
    time.sleep(0.5)
    first.ring_buffer_consume()
    core.ring_buffer_consume()

    t0 = boot_ns()
    spin = "x = 0\nfor i in range(300_000): x += i * i"
    workers = [threading.Thread(target=exec, args=(spin,)) for _ in range(6)]
    for worker in workers:
        worker.start()
    for _ in range(15):
        subprocess.run(["/bin/sh", "-c", "head -c 200000 /dev/urandom | sha256sum > /dev/null"])
    for worker in workers:
        worker.join()
    t1 = boot_ns()
    time.sleep(0.3)
    first.ring_buffer_consume()
    core.ring_buffer_consume()

    dtype = SchedSwitchBpfMonitor.SCHED_EVENT_DTYPE
    drops_first = first.percpu_sum("rb_drops", 0) if control else sum(first["rb_drops"][first["rb_drops"].Key(0)])
    drops_core = core.percpu_sum("rb_drops", 0)
    if not control:
        first.cleanup()
    a = np.frombuffer(b"".join(r[: dtype.itemsize] for r in raw_first), dtype)
    b = np.frombuffer(b"".join(r[: dtype.itemsize] for r in raw_core), dtype)
    print(f"  events: first={len(a)} core={len(b)}, drops: first={drops_first} core={drops_core}")

    identity = ["pid", "tgid", "pid_start", "cpu_core", "event_type"]
    metadata = ["tgid_start", "parent_tid", "parent_tgid", "parent_tid_start", "parent_tgid_start"]
    metadata += ["cgroup_id", "uid", "gid", "comm", "cmdline"]
    counters = ["aperf", "mperf", "unhalted_core_cycles", "loads_local", "loads_remote", "stores"]
    compared, first_seen_gaps, max_dt, mismatches = 0, 0, 0, []
    diffs = {name: [] for name in counters}
    sizes = {name: [] for name in counters}
    for cpu in range(num_cpus):
        ea = a[(a["cpu_core"] == cpu) & (a["time_end"] >= t0) & (a["time_end"] <= t1)]
        eb = b[b["cpu_core"] == cpu]
        if len(ea) == 0:
            continue
        # Anchor on the event nearest in time to the window's first one.
        dt = np.abs(eb["time_end"].astype(np.int64) - int(ea[0]["time_end"]))
        dt[(eb["pid"] != ea[0]["pid"]) | (eb["pid_start"] != ea[0]["pid_start"])] = np.iinfo(np.int64).max
        j = int(np.argmin(dt))
        if dt[j] > 1_000_000:
            mismatches.append((cpu, "no common event"))
            continue
        eb = eb[j : j + len(ea)]
        if len(eb) != len(ea):
            mismatches.append((cpu, "length", len(ea), len(eb)))
            continue
        # A task the first program saw before the second was loaded gets its metadata event from
        # the second program only: the load order, not a porting difference.
        gap = (ea["event_type"] != eb["event_type"]) & (ea["pid_start"] < loaded)
        for name in ["pid", "tgid", "pid_start", "cpu_core"]:
            gap &= ea[name] == eb[name]
        first_seen_gaps += int(gap.sum())
        ea, eb = ea[~gap], eb[~gap]
        for name in identity + metadata:
            bad = np.nonzero(ea[name] != eb[name])[0]
            if len(bad):
                mismatches.append((cpu, name, len(bad), ea[bad[0]][name], eb[bad[0]][name]))
        for name in ["time_start", "time_end"]:
            max_dt = max(max_dt, int(np.abs(ea[name].astype(np.int64) - eb[name].astype(np.int64)).max()))
        for name in counters:
            da = ea[f"{name}_end"].astype(np.int64) - ea[f"{name}_start"].astype(np.int64)
            db = eb[f"{name}_end"].astype(np.int64) - eb[f"{name}_start"].astype(np.int64)
            diffs[name].append(np.abs(da - db))
            sizes[name].append(np.abs(da))
        compared += len(ea)

    print(f"  compared {compared} events; largest time difference {max_dt} ns; first-seen gaps {first_seen_gaps}")
    for name in counters:
        diff, size = np.concatenate(diffs[name] or [[0]]), np.concatenate(sizes[name] or [[0]])
        print(f"  {name:22s} sum(|diff|)/sum(|delta|) = {diff.sum() / max(size.sum(), 1):.2e}")
    for mismatch in mismatches[:10]:
        print("  mismatch:", mismatch)
    return not mismatches and drops_first == 0 and drops_core == 0


def check_tcp(reference: Path) -> bool:
    """Send known amounts over TCP and UDP, IPv4 and IPv6, and compare every event of this process."""
    from bcc import BPF

    fields = [name for name, _ in TcpEvent._fields_ if name != "__pad"]
    metadata = {"parent_tid_start", "parent_tgid_start", "parent_tid", "parent_tgid", "uid", "gid", "comm", "cmdline"}

    def record(event):
        values = {name: getattr(event, name) for name in fields}
        return {name: bytes(v) if isinstance(v, ct.Array) else v for name, v in values.items()}

    bcc = BPF(src_file=str(reference / "tcp_monitor.c"))
    for kernel_function, probe in [
        ("tcp_cleanup_rbuf", "trace_tcp_cleanup_rbuf"),
        ("tcp_sendmsg", "trace_tcp_sendmsg"),
        ("udp_sendmsg", "trace_udp_sendmsg"),
        ("udp_recvmsg", "trace_udp_recvmsg"),
    ]:
        bcc.attach_kprobe(event=kernel_function, fn_name=probe)
    core = BpfProgram("tcp_monitor")
    events_bcc, events_core = [], []
    bcc["events"].open_ring_buffer(lambda ctx, data, size: events_bcc.append(record(bcc["events"].event(data))) or 0)
    core.open_ring_buffer("events", lambda data: events_core.append(record(TcpEvent.from_buffer_copy(data))) or 0)
    time.sleep(0.2)
    bcc.ring_buffer_consume()
    core.ring_buffer_consume()
    events_bcc.clear()
    events_core.clear()

    def tcp_roundtrip(family, host, sizes):
        server = socket.socket(family, socket.SOCK_STREAM)
        server.bind((host, 0))
        server.listen(1)

        def serve():
            conn, _ = server.accept()
            received = 0
            while received < sum(sizes):
                received += len(conn.recv(65536))
            conn.sendall(b"k" * 1234)
            conn.close()

        thread = threading.Thread(target=serve)
        thread.start()
        client = socket.socket(family, socket.SOCK_STREAM)
        client.connect(server.getsockname()[:2])
        for size in sizes:
            client.sendall(b"x" * size)
        received = 0
        while received < 1234:
            received += len(client.recv(65536))
        client.close()
        thread.join()
        server.close()

    def udp_roundtrip(family, host, sizes):
        receiver = socket.socket(family, socket.SOCK_DGRAM)
        receiver.bind((host, 0))
        sender = socket.socket(family, socket.SOCK_DGRAM)
        for size in sizes:
            sender.sendto(b"u" * size, receiver.getsockname()[:2])
            receiver.recv(65536)
        receiver.close()
        sender.close()

    tcp_roundtrip(socket.AF_INET, "127.0.0.1", [100, 5000, 70000])
    tcp_roundtrip(socket.AF_INET6, "::1", [42, 9000])
    udp_roundtrip(socket.AF_INET, "127.0.0.1", [10, 1400, 20000])
    udp_roundtrip(socket.AF_INET6, "::1", [77, 3000])
    time.sleep(0.3)
    bcc.ring_buffer_consume()
    core.ring_buffer_consume()
    bcc.cleanup()

    me = os.getpid()
    a = [event for event in events_bcc if event["tgid"] == me]
    b = [event for event in events_core if event["tgid"] == me]
    sent = Counter()
    for event in b:
        sent[("tcp", "udp")[event["transport_type"]], ("recv", "send")[event["direction"]]] += event["bytes"]
    print(f"  events of this process: bcc={len(a)} core={len(b)}; bytes: {dict(sent)}")
    mismatches, max_dt = [], 0
    if len(a) != len(b):
        mismatches.append(("count", len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        for name in fields:
            if name == "time_ns":
                max_dt = max(max_dt, abs(x[name] - y[name]))
            elif name in metadata and x["event_type"] == 0:
                continue  # left unwritten on data-only events
            elif x[name] != y[name]:
                mismatches.append((i, name, x[name], y[name]))
    print(f"  largest timestamp difference: {max_dt} ns")
    for mismatch in mismatches[:10]:
        print("  mismatch:", mismatch)
    return not mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BCC probes of an older commit and their libbpf CO-RE ports side by side on the "
        "same tracepoints and compare the events. Needs root and BCC."
    )
    parser.add_argument("program", choices=["registry", "sched_switch", "tcp"])
    parser.add_argument("--ref", default=DEFAULT_REF, help=f"commit with the BCC sources (default {DEFAULT_REF})")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--all-perf", action="store_true", help="sched_switch: open every perf group")
    parser.add_argument("--control", action="store_true", help="sched_switch: CO-RE against CO-RE")
    args = parser.parse_args()

    failed = 0
    with tempfile.TemporaryDirectory() as folder:
        reference = Path(folder)
        extract_reference(args.ref, reference)
        for run in range(1, args.runs + 1):
            print(f"{args.program}, run {run}/{args.runs}")
            if args.program == "registry":
                identical = check_registry(reference)
            elif args.program == "sched_switch":
                identical = check_sched_switch(reference, args.all_perf, args.control)
            else:
                identical = check_tcp(reference)
            print("  IDENTICAL" if identical else "  DIFFERENT")
            failed += not identical
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
