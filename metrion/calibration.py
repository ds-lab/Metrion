import fnmatch
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

RAPL = Path("/sys/devices/virtual/powercap/intel-rapl")
# The worker's own unit. The fixed name also keeps two calibrations from running at once.
UNIT = "metrion-calibrate"
# Container runtimes and their containers live in system.slice, which is never frozen as a whole.
RUNTIMES = ["containerd.service", "docker.service", "docker-*.scope", "k3s*.service", "crio.service", "libpod-*.scope"]
# Frozen only on request: a guest may not survive a minute without time passing.
OPT_IN = ["machine.slice"]
# Freezing these would stop the machine itself, or the measurement.
NEVER = ["-.slice", "system.slice", "init.scope", "dbus*.service", "systemd-*.service", f"{UNIT}.service"]
# Time for the frozen machine to settle into idle before the first reading.
SETTLE_S = 60


class CalibrationError(RuntimeError):
    pass


def cache_dir() -> Path:
    """Where measurements that hold per machine are kept; under sudo that is root's cache."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "metrion"


def cache_file() -> Path:
    return cache_dir() / "calibration.json"


def load() -> dict | None:
    try:
        return json.loads(cache_file().read_text())
    except FileNotFoundError:
        return None


def idle_power(component: str, devices: list[str]) -> dict[str, float | None]:
    """Idle power in watts per device, None where the machine has no counter for it.

    Parameters
    ----------
    component : str
        cpu, dram or gpu.
    devices : list[str]
        RAPL package numbers, or GPU UUIDs.

    Raises
    ------
    CalibrationError
        If there is no calibration, or it does not know one of the devices.
    """
    calibration = load()
    if calibration is None:
        raise CalibrationError("The idle power is not measured yet; run: sudo metrion calibrate")
    measured = calibration[component]
    missing = [device for device in devices if device not in measured]
    if missing:
        raise CalibrationError(
            f"The calibration from {calibration['measured_at']} does not know {component} {', '.join(missing)}; "
            "run: sudo metrion calibrate"
        )
    return {device: measured[device] for device in devices}


def select_units(active: list[str], freeze: list[str], skip: list[str]) -> list[str]:
    """The units to freeze: top-level slices and container runtimes, changed by freeze and skip patterns.

    Raises
    ------
    CalibrationError
        If a pattern matches no active unit, which is most likely a typo.
    """
    for pattern in [*freeze, *skip]:
        if not fnmatch.filter(active, pattern):
            raise CalibrationError(f"No active unit matches {pattern}")

    def matches(unit: str, patterns: list[str]) -> bool:
        return any(fnmatch.fnmatchcase(unit, pattern) for pattern in patterns)

    # A slice without a dash sits directly below the root slice; its children freeze with it.
    top_level = [unit for unit in active if unit.endswith(".slice") and "-" not in unit.removesuffix(".slice")]
    default = [unit for unit in top_level if not matches(unit, OPT_IN)] + [
        unit for unit in active if matches(unit, RUNTIMES)
    ]
    chosen = default + [unit for unit in active if matches(unit, freeze) and unit not in default]
    return [unit for unit in chosen if not matches(unit, skip) and not matches(unit, NEVER)]


def active_units() -> list[str]:
    lines = subprocess.run(
        ["systemctl", "list-units", "--state=active", "--type=slice,service,scope", "--plain", "--no-legend"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return [line.split()[0] for line in lines if line.strip()]


def describe(units: list[str]) -> dict[str, dict[str, str]]:
    """CanFreeze and TasksCurrent of each unit, from systemctl show."""
    output = subprocess.run(
        ["systemctl", "show", "-p", "Id", "-p", "CanFreeze", "-p", "TasksCurrent", "--", *units],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    blocks = [dict(line.split("=", 1) for line in block.splitlines()) for block in output.strip().split("\n\n")]
    return {block["Id"]: block for block in blocks}


def gpu_processes() -> list[str]:
    """The processes that hold a GPU; they keep their context while frozen."""
    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return []
    try:
        found = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            for process in [
                *pynvml.nvmlDeviceGetComputeRunningProcesses(handle),
                *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle),
            ]:
                found.append(f"{process.pid} {_comm(process.pid)} on GPU {index}")
        return found
    finally:
        pynvml.nvmlShutdown()


def run(units: list[str], seconds: int) -> dict:
    """Measure in a transient system service, which stays outside every unit it freezes."""
    systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
    # systemd reads backslashes and percent signs in a command line as escapes.
    thaw = " ".join(unit.replace("\\", "\\\\").replace("%", "%%") for unit in units)
    result = subprocess.run(
        [
            "systemd-run",
            f"--unit={UNIT}",
            "--collect",
            "--wait",
            "--pipe",
            "--quiet",
            "-p",
            f"RuntimeMaxSec={seconds + SETTLE_S + 60}",
            # Thaws even if the worker is killed; the leading dash ignores units already thawed.
            "-p",
            f"ExecStopPost=-{systemctl} thaw {thaw}",
            sys.executable,
            "-m",
            "metrion.calibration",
            str(seconds),
            *units,
        ],
        # Nothing may reach the terminal: sudo's terminal lives in user.slice, which is frozen.
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # With --pipe the worker's traceback comes back here; the journal only knows how the service ended.
        reason = (result.stderr.strip().splitlines() or [f"killed or timed out, see journalctl -u {UNIT}"])[-1]
        raise CalibrationError(
            f"The measurement failed with exit code {result.returncode}, the units are thawed again: {reason}"
        )
    calibration = json.loads(result.stdout)
    cache_dir().mkdir(parents=True, exist_ok=True)
    cache_file().write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def measure(units: list[str], seconds: int) -> dict:
    """Freeze the units, read every energy counter twice, thaw. Runs inside the transient service."""
    frozen, not_frozen = [], {}
    try:
        for unit in units:
            # One at a time, to know which units did not freeze. Returns once the unit is frozen.
            result = subprocess.run(["systemctl", "freeze", unit], capture_output=True, text=True)
            if result.returncode == 0:
                frozen.append(unit)
            else:
                not_frozen[unit] = result.stderr.strip()
        time.sleep(SETTLE_S)
        before, started = read_counters(), time.monotonic_ns()
        time.sleep(seconds)
        after, ended = read_counters(), time.monotonic_ns()
    finally:
        # All at once, like ExecStopPost: systemctl thaws the others even if one of them fails.
        if frozen:
            subprocess.run(["systemctl", "thaw", *frozen], capture_output=True)

    elapsed_s = (ended - started) / 1e9
    power = {component: {} for component in ("cpu", "dram", "gpu")}
    for (component, device), (value, wraps_at, per_joule) in before.items():
        if value is None:
            power[component][device] = None
            continue
        delta = after[component, device][0] - value
        power[component][device] = (delta % wraps_at if wraps_at else delta) / per_joule / elapsed_s
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seconds": round(elapsed_s, 3),
        "frozen": frozen,
        "not_frozen": not_frozen,
        **power,
    }


def read_counters() -> dict[tuple[str, str], tuple[int | None, int | None, float]]:
    """Every energy counter: (component, device) to (value, value it wraps at, units per joule).

    CPU and DRAM are keyed by RAPL package number, a package without a DRAM domain has value None.
    GPUs are keyed by UUID.
    """
    counters = {}
    for package in sorted(RAPL.glob("intel-rapl:[0-9]*")):
        name = _read(package / "name")
        if not name or not name.startswith("package-"):
            continue
        number = name.removeprefix("package-")
        counters["cpu", number] = _rapl(package)
        dram = [domain for domain in package.glob("intel-rapl:*:*") if _read(domain / "name") == "dram"]
        counters["dram", number] = _rapl(dram[0]) if dram else (None, None, 1e6)

    import pynvml

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return counters
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            # Millijoules since the driver was loaded; 64 bits do not wrap.
            counters["gpu", pynvml.nvmlDeviceGetUUID(handle)] = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
                None,
                1e3,
            )
    finally:
        pynvml.nvmlShutdown()
    return counters


def _rapl(domain: Path) -> tuple[int, int, float]:
    return int(_read(domain / "energy_uj")), int(_read(domain / "max_energy_range_uj")), 1e6


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _comm(pid: int) -> str:
    return _read(Path(f"/proc/{pid}/comm")) or "?"


if __name__ == "__main__":
    # systemd stops the service with SIGTERM; as an exception it runs the thaw in measure.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    print(json.dumps(measure(sys.argv[2:], int(sys.argv[1]))))
