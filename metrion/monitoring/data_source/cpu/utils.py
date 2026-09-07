import ctypes as ct
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import psutil
from bcc import Perf
from bcc.utils import get_online_cpus

boot_time = datetime.fromtimestamp(psutil.boot_time())
_ppid_start_cache: dict[int, float] = {}
_CLK_TCK = os.sysconf(os.sysconf_names["SC_CLK_TCK"])


def grouping_threshold_ms(interval_ms: int) -> int:
    """
    Guard band that keeps grouped rows away from the boundaries of an energy interval.

    Ten percent of the interval, clamped to 5-100 ms. A row is only grouped when it stays that
    far from both ends, so the usable window is ``interval_ms - 2 * grouping_threshold_ms(...)``
    and stays at 80 % of the interval down to 50 ms.
    """
    return max(5, min(100, interval_ms // 10))


def get_pid_starttime(pid: int) -> datetime:
    """
    Get the start time of a process given its PID.

    Parameters
    ----------
    pid : int
        Process ID to get the start time for.

    Returns
    -------
    datetime
        The start time of the process.
    """
    try:
        start_ns = get_pid_starttime_ns(pid)
    except (FileNotFoundError, ValueError):
        return boot_time
    return boot_time + timedelta(microseconds=start_ns // 1000)


def read_starttime_from_stat_file_in_ns(stat_path: Path) -> int:
    try:
        content = stat_path.read_text()
    except FileNotFoundError:
        raise

    # /proc/<pid>/stat format:
    # pid (comm) state ppid ... starttime
    #  skip the "(comm)" field
    rparen = content.rfind(")")
    if rparen == -1:
        raise ValueError(f"Malformed stat file for path {stat_path}")

    fields = content[rparen + 2 :].split()

    # Field 22 (1-based) => index 19 *after* the state field
    # fields[0] == state (field 3)
    try:
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot parse starttime for path {stat_path}") from error

    start_ns = (start_ticks * 1_000_000_000) // _CLK_TCK
    return start_ns


def get_pid_starttime_ns(pid: int) -> int:
    """
    Get the start time of a process given its PID in nanoseconds since boot.

    Parameters
    ----------
    pid : int
        Process ID.

    Returns
    -------
    int
        Start time in nanoseconds since boot.

    Raises
    ------
    FileNotFoundError
        If the process no longer exists.
    ValueError
        If /proc/<pid>/stat cannot be parsed.
    """
    if pid in _ppid_start_cache:
        return _ppid_start_cache[pid]

    stat_path = Path(f"/proc/{pid}/stat")
    try:
        start_ns = read_starttime_from_stat_file_in_ns(stat_path)
    except (FileNotFoundError, ValueError):
        raise

    _ppid_start_cache[pid] = start_ns
    return start_ns


def get_pid_cmdline(pid: int, short_cmdline: str) -> str | None:
    """
    Get the full command line for a given process ID.

    Parameters
    ----------
    pid : int
        Process ID to get the command line for.

    Returns
    -------
    str or None
        The command line of the process, or None if not found.
    """
    try:
        cmdline_file = Path(f"/proc/{pid}/cmdline")
        with cmdline_file.open() as f:
            content = f.read()
            # cmdline is null-separated
            res = content.replace("\0", " ").strip()
            if res == "":
                return short_cmdline
            return res
    except Exception:
        return short_cmdline


def get_ppid(pid: int) -> int | None:
    """
    Fast PPID lookup using /proc/<pid>/stat.
    Reads minimal data and avoids full splitting.
    """
    stat_path = f"/proc/{pid}/stat"
    try:
        with open(stat_path) as f:
            data = f.read().split()
            if len(data) >= 4:
                return int(data[3])
    except FileNotFoundError:
        return None

    return None


def read_perf_type(path_type: str) -> int:
    path = Path(path_type)
    if not path.exists():
        raise FileNotFoundError(f"Path {path_type} does not exist.")
    with path.open() as f:
        type = f.read().strip()
        return int(type, 16) if type.startswith("0x") else int(type)


def read_perf_event(path_event: str) -> int:
    path = Path(path_event)
    if not path.exists():
        raise FileNotFoundError(f"Path {path_event} does not exist.")
    with path.open() as f:
        event = f.read().strip()
        # 'event=0x2e,umask=0x41' -> first part is the event number second part is the umask
        # combine to 0x412e
        if "," in event:
            event, umask = event.split(",")
            event = f"0x{umask[8:]}{event[8:]}"  # remove '0x' and 'umask='
        elif event.startswith("event="):
            event = event[6:]
        return int(event, 16) if event.startswith("0x") else int(event)


def get_cgroup_v2_path(pid: int) -> str:
    try:
        cgroup_file = Path(f"/proc/{pid}/cgroup")
        with cgroup_file.open() as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 3 and parts[0] == "0":  # v2 has only one line, hierarchy ID is 0
                    return parts[2]  # The cgroup path
    except Exception:
        return "Cgroup v2 path not found"
    return "Cgroup v2 path not found"


def get_process_comm(pid: int) -> str:
    """
    Get the command name (comm) for a given process ID.

    Parameters
    ----------
    pid : int
        Process ID to get the command name for.

    Returns
    -------
    str
        The command name of the process, or "other" if not found.
    """
    try:
        comm_file = Path(f"/proc/{pid}/comm")
        with comm_file.open() as f:
            return f.read().strip()
    except Exception:
        return "other"


def find_cgroup_path_by_inode(target_inode: int, base_path: Path = Path("/sys/fs/cgroup")) -> Path | None:
    """
    Recursively search for a directory in the cgroup filesystem (using pathlib)
    with a matching inode.

    Parameters
    ----------
    target_inode : int
        The inode number (cgroup id) to search for.
    base_path : Path, optional
        The starting point for the search (default is Path("/sys/fs/cgroup")).

    Returns
    -------
    Path or None
        The full path to the matching directory, or None if not found.
    """
    try:
        # Iterate through items in the current directory
        for entry in base_path.iterdir():
            try:
                # Avoid following symlinks for safety and limit to directories only
                if entry.is_dir() and not entry.is_symlink():
                    # Use lstat to get the inode without following symlinks
                    if entry.lstat().st_ino == target_inode:
                        return entry
                    # Recursively search in the subdirectory
                    result = find_cgroup_path_by_inode(target_inode, entry)
                    return result
            except PermissionError:
                # Skip directories for which you don't have permission
                continue
    except PermissionError:
        # Skip base_path if permission is denied
        pass

    return None


def count_cpu_sockets_from_sysfs():
    ids = set()
    for topo_file in glob("/sys/devices/system/cpu/cpu*/topology/physical_package_id"):
        with open(topo_file) as f:
            ids.add(int(f.read().strip()))
    return len(ids)


def get_cpu_topology():
    topo = {}
    for cpu_path in glob("/sys/devices/system/cpu/cpu[0-9]*"):
        cpu = int(cpu_path.split("cpu")[-1])
        topo_dir = f"{cpu_path}/topology"
        # read socket and core IDs
        with open(f"{topo_dir}/physical_package_id") as f:
            socket = int(f.read().strip())
        with open(f"{topo_dir}/core_id") as f:
            core = int(f.read().strip())
        # insert into nested dict
        topo.setdefault(socket, {}).setdefault(core, []).append(cpu)

    return topo


# https://github.com/iovisor/bcc/blob/v0.35.0/src/python/bcc/perf.py#L155
class CustomPerf(Perf):
    @staticmethod
    def _open_for_cpu(cpu, attr, pid=-1, group_fd=-1):
        pfd = Perf.syscall(
            Perf.NR_PERF_EVENT_OPEN,
            ct.byref(attr),
            pid,
            cpu,
            group_fd,
            Perf.PERF_FLAG_FD_CLOEXEC,
        )
        if pfd < 0:
            errno_ = ct.get_errno()
            raise OSError(errno_, os.strerror(errno_), f"for cpu:{cpu}")

        if attr.type == Perf.PERF_TYPE_TRACEPOINT:
            if Perf.ioctl(pfd, Perf.PERF_EVENT_IOC_SET_FILTER, "common_pid == -17") < 0:
                errno_ = ct.get_errno()
                raise OSError(errno_, os.strerror(errno_), f"for cpu:{cpu}")

        # we don't setup the perf ring buffers, as we won't read them

        if Perf.ioctl(pfd, Perf.PERF_EVENT_IOC_ENABLE, 0) < 0:
            errno_ = ct.get_errno()
            raise OSError(errno_, os.strerror(errno_), f"for cpu:{cpu}")

        return pfd

    @staticmethod
    def perf_custom_event_open_with_results(attr, pid=-1, cpus: list[int] = [], group_fds: list[int] = []) -> list[int]:  # noqa: B006
        fds = []
        cpus = cpus if cpus else get_online_cpus()
        group_fds = group_fds if len(group_fds) == len(cpus) else [-1] * len(cpus)
        for cpu, group_fd in zip(cpus, group_fds, strict=True):
            fd = CustomPerf._open_for_cpu(cpu, attr, pid, group_fd)
            fds.append(fd)
        return fds


def run_strace(event, cpu=None, perf="perf"):
    # sudo strace -e perf_event_open -v perf stat -e cpu_atom/r003c/ -C 15 sleep 1 2>&1 | grep perf_event_open
    cmd = ["strace", "-e", "perf_event_open", "-v", perf, "stat", "-e", event]
    if cpu is not None:
        cmd += ["-C", str(cpu)]
    cmd += ["--", "true"]
    p = subprocess.run(cmd, text=True, capture_output=True)
    text = p.stdout + p.stderr
    if p.returncode != 0 and "perf_event_open(" not in text:
        raise RuntimeError(text.strip() or "strace failed")
    # Grab the first perf_event_open call
    # match ([^}]*) match everything except }
    m = re.search(r"perf_event_open\(\{([^}]*)\}", text)
    if not m:
        raise RuntimeError("Event not supported (no perf_event_open captured)")
    return m.group(1)  # inner "key=val, key=val, ..."


PERF_TYPE_MAP = {
    "PERF_TYPE_HARDWARE": 0,
    "PERF_TYPE_SOFTWARE": 1,
    "PERF_TYPE_TRACEPOINT": 2,
    "PERF_TYPE_HW_CACHE": 3,
    "PERF_TYPE_RAW": 4,
    "PERF_TYPE_BREAKPOINT": 5,
}


def translate_perf_type(type_str: str) -> int:
    """
    Translate a perf type string like "PERF_TYPE_HARDWARE" to its enum value.
    """
    if type_str not in PERF_TYPE_MAP:
        raise ValueError(f"Unknown perf type: {type_str}")
    return PERF_TYPE_MAP[type_str]


def parse_attr_kvlist(kvblob):
    # Split on ', ' at top level (no nested braces present in perf_event_attr)
    pairs = [p.strip() for p in kvblob.split(",")]
    out = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        v = v.strip()
        # remove comments from value /**/
        v = re.sub(r"\/\*.*?\*\/", "", v).strip()
        # Convert perf type string to int
        if k == "type" and v.startswith("PERF_TYPE_"):
            out[k] = translate_perf_type(v)
            continue
        # Convert hex numbers like 0x1b7
        if v.startswith("0x"):
            out[k] = int(v, 16)
            continue
        # Convert plain ints
        if re.fullmatch(r"-?\d+", v):
            out[k] = int(v)
            continue
        # Leave other strings as-is (rare here)
        out[k] = v
    return out


def resolve_with_strace(event, cpu=None, perf="perf"):
    print(f"Resolving event {event} with CPU {cpu} and perf {perf}")
    kvblob = run_strace(event, cpu=cpu, perf=perf)
    d = parse_attr_kvlist(kvblob)
    # Validate presence
    need = ["type", "config"]
    if not all(k in d for k in need):
        raise RuntimeError(f"Missing required fields in strace output: have {list(d)}")
    # config1/config2 might be absent depending on event/uarch
    return {
        "event": event,
        "type": d.get("type"),
        "config": d.get("config"),
        "config1": d.get("config1", 0),
        "config2": d.get("config2", 0),
    }


if __name__ == "__main__":
    event = sys.argv[1] if len(sys.argv) > 1 else "offcore_response.demand_rfo.l3_miss_remote_dram.snoop_miss_or_no_fwd"
    print(resolve_with_strace(event, cpu=15))
