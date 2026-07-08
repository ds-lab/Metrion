from __future__ import annotations

import csv
import os
import pwd
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

LOG_DEFAULT = "/var/log"
LOG_NAME = "gpu-dram-max-power.log"
BASE_RAPL = Path("/sys/devices/virtual/powercap/intel-rapl")


@dataclass
class GpuCaps:
    index: int
    name: str
    bus: str
    min_cap_w: str
    max_cap_w: str
    enforced_w: str


@dataclass
class RaplDomainCaps:
    package: str
    path: Path
    power_limit_w: str
    max_cap_w: str


@dataclass
class RaplPackageCaps:
    package: str
    path: Path
    power_limit_w: str
    max_cap_w: str
    dram: list[RaplDomainCaps] = field(default_factory=list)


def now_iso() -> str:
    """Return current timestamp in ISO format (seconds precision)."""
    return datetime.now(UTC).isoformat(
        timespec="seconds",
    )


def ensure_log_dir(path: Path) -> None:
    """Ensure the log directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def chown_to_sudo_user(path: Path) -> None:
    """Chown path (and children) to the sudo user, if present."""
    sudo_user = os.environ.get("SUDO_USER")
    if not sudo_user:
        return
    try:
        info = pwd.getpwnam(sudo_user)
    except KeyError:
        return

    def _safe_chown(p: Path) -> None:
        try:
            os.chown(p, info.pw_uid, info.pw_gid)
        except PermissionError:
            pass

    _safe_chown(path)
    if path.is_dir():
        for child in path.rglob("*"):
            _safe_chown(child)


def log_lines(log_file: TextIO, lines: Iterable[str]) -> None:
    """Write lines to stdout and the log file."""
    for line in lines:
        print(line)
        log_file.write(line + "\n")
    log_file.flush()


def run_nvidia_smi() -> list[GpuCaps]:
    """Collect GPU power caps via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,pci.bus_id,power.min_limit,power.max_limit,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    reader = csv.reader(line.strip() for line in out.splitlines() if line.strip())
    caps: list[GpuCaps] = []
    for idx, row in enumerate(reader):
        row += [""] * (5 - len(row))
        name, bus, min_cap, max_cap, enforced = (col.strip() for col in row[:5])
        caps.append(
            GpuCaps(
                index=idx,
                name=name,
                bus=bus,
                min_cap_w=min_cap,
                max_cap_w=max_cap,
                enforced_w=enforced,
            )
        )
    return caps


def read_power_uw(file_path: Path) -> str:
    """Read a power value in microwatts and return watts as a string."""
    try:
        micro_watts = float(file_path.read_text().strip())
    except (OSError, ValueError):
        return "N/A"
    return f"{micro_watts / 1e6:.4f}"


def scan_rapl_packages() -> list[Path]:
    """Find RAPL package (CPU socket) domains with energy counters."""
    if not BASE_RAPL.is_dir():
        return []
    pkgs = []
    for pkg in BASE_RAPL.glob("intel-rapl:[0-9]*"):
        if not (pkg / "energy_uj").is_file():
            continue
        name_file = pkg / "name"
        if not name_file.is_file() or not name_file.read_text().strip().startswith("package-"):
            continue
        pkgs.append(pkg)
    return pkgs


def rapl_package_caps(pkg: Path) -> RaplPackageCaps:
    """Build package-level RAPL caps for a package path."""
    pkg_num = (pkg / "name").read_text().strip().removeprefix("package-")
    return RaplPackageCaps(
        package=pkg_num,
        path=pkg,
        power_limit_w=read_power_uw(pkg / "constraint_0_power_limit_uw"),
        max_cap_w=read_power_uw(pkg / "constraint_0_max_power_uw"),
    )


def rapl_dram_caps(pkg: Path) -> list[RaplDomainCaps]:
    """Collect DRAM domain caps under a package."""
    pkg_num = pkg.name.split(":", 1)[-1]
    caps: list[RaplDomainCaps] = []
    for sub in pkg.glob("intel-rapl:*:*"):
        name_file = sub / "name"
        if not name_file.is_file():
            continue
        if name_file.read_text().strip() != "dram":
            continue
        caps.append(
            RaplDomainCaps(
                package=pkg_num,
                path=sub,
                power_limit_w=read_power_uw(sub / "constraint_0_power_limit_uw"),
                max_cap_w=read_power_uw(sub / "constraint_0_max_power_uw"),
            )
        )
    return caps


def collect_gpu_caps() -> tuple[bool, list[GpuCaps]]:
    """Return (has_nvidia_smi, gpu_caps)."""
    if not shutil.which("nvidia-smi"):
        return False, []
    return True, run_nvidia_smi()


def collect_rapl_caps() -> list[RaplPackageCaps]:
    """Collect package and DRAM RAPL caps."""
    packages = []
    for pkg in scan_rapl_packages():
        pkg_caps = rapl_package_caps(pkg)
        pkg_caps.dram = rapl_dram_caps(pkg)
        packages.append(pkg_caps)
    return packages


def write_gpu_caps(log_file: TextIO, has_nvidia_smi: bool, caps: list[GpuCaps]) -> None:
    """Render GPU cap data to the log."""
    log_lines(log_file, ["NVIDIA : checking nvidia-smi for power caps"])
    if not has_nvidia_smi:
        log_lines(log_file, ["NVIDIA : nvidia-smi not found"])
        return

    if not caps:
        log_lines(log_file, ["NVIDIA : no GPUs reported by nvidia-smi"])
        return

    log_lines(log_file, [f"Found  : {len(caps)} NVIDIA GPU(s)"])
    for cap in caps:
        line = (
            f"GPU[{cap.index}] {cap.name:<50} | bus={cap.bus:<14} | "
            f"min_cap={cap.min_cap_w:>8} W | max_cap={cap.max_cap_w:>8} W | "
            f"enforced={cap.enforced_w:>8} W"
        )
        log_lines(log_file, [line])


def write_rapl_caps(log_file: TextIO, packages: list[RaplPackageCaps]) -> None:
    """Render RAPL package and DRAM cap data to the log."""
    log_lines(log_file, [f"RAPL   : scanning package (CPU) and DRAM domains under {BASE_RAPL}"])
    if not packages:
        log_lines(log_file, ["RAPL   : no package domains found"])
        return

    log_lines(log_file, [f"Found  : {len(packages)} package domain(s)"])
    for pkg_caps in packages:
        log_lines(log_file, [f"PKG    : package={pkg_caps.package} | path={pkg_caps.path}"])
        log_lines(log_file, [f"         power_limit={pkg_caps.power_limit_w} W | max_cap={pkg_caps.max_cap_w} W"])

        if not pkg_caps.dram:
            log_lines(log_file, [f"DRAM   : package={pkg_caps.package} | no DRAM domain detected"])
            continue

        for cap in pkg_caps.dram:
            log_lines(log_file, [f"DRAM   : package={cap.package} | path={cap.path}"])
            log_lines(log_file, [f"         power_limit={cap.power_limit_w} W | max_cap={cap.max_cap_w} W"])


def main() -> int:
    """Entry point: collect and log GPU/RAPL caps."""
    log_folder = Path(LOG_DEFAULT)
    ensure_log_dir(log_folder)

    log_path = log_folder / LOG_NAME
    with log_path.open("a", encoding="utf-8") as log_file:
        chown_to_sudo_user(log_folder)
        log_lines(log_file, [f"LOG    : {log_path}", f"Begin  : {now_iso()}"])

        has_nvidia_smi, gpu_caps = collect_gpu_caps()
        rapl_caps = collect_rapl_caps()

        write_gpu_caps(log_file, has_nvidia_smi, gpu_caps)
        write_rapl_caps(log_file, rapl_caps)

        log_lines(log_file, [f"Done   : {now_iso()}"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
