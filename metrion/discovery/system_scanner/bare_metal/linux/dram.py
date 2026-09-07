import pathlib
import re

from loguru import logger

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import collect_rapl_caps
from metrion.discovery.system_scanner.bare_metal.linux.util import get_cpu_idle_power_data, get_nodes_per_package


def _resolve_power_data(power_config: dict | list | str) -> tuple[str, dict[int, dict[str, float]]]:
    """Return (mode, per-socket power data) using manual or auto sources."""
    if isinstance(power_config, str):
        power_config = power_config.lower()

    if power_config == "auto":
        return "auto", _combine_power_data(_auto_idle_power_data(), _auto_max_power_data())

    manual_entries: dict[int, dict[str, float | str | None]] = {}

    if isinstance(power_config, list):
        for entry in power_config:
            package_id = int(entry.get("ID", -1))
            manual_entries[package_id] = {
                "idle": entry.get("IDLE_POWER", "auto"),
                "max": entry.get("MAX_POWER", "auto"),
            }
    else:
        logger.warning(f"Unknown DRAM power config '{power_config}', falling back to auto")
        return "auto", _combine_power_data(_auto_idle_power_data(), _auto_max_power_data())

    if not manual_entries:
        return "auto", _combine_power_data(_auto_idle_power_data(), _auto_max_power_data())

    idle_auto: dict[int, float] | None = None
    max_auto: dict[int, float] | None = None
    resolved: dict[int, dict[str, float]] = {}

    for pkg_id, manual in manual_entries.items():
        idle_val = manual.get("idle", "auto")
        if idle_val == "auto" or idle_val is None:
            if idle_auto is None:
                idle_auto = _auto_idle_power_data()
            idle_val = idle_auto.get(pkg_id, 0.0)
        else:
            idle_val = _safe_float(idle_val, 0.0)

        max_val = manual.get("max", "auto")
        if max_val == "auto" or max_val is None:
            if max_auto is None:
                max_auto = _auto_max_power_data()
            max_val = max_auto.get(pkg_id, 0.0)
        else:
            max_val = _safe_float(max_val, 0.0)

        resolved[pkg_id] = {"idle": idle_val, "max": max_val}

    return "manual", resolved


def _auto_idle_power_data() -> dict[int, float]:
    """Auto-detect idle DRAM power using RAPL."""
    rapl_data = get_cpu_idle_power_data()
    idle_power: dict[int, float] = {}
    for pkg_id, socket_data in rapl_data.items():
        dram_val = 0.0
        if isinstance(socket_data, dict):
            dram_val = socket_data.get("dram", 0.0) or 0.0
        idle_power[pkg_id] = _safe_float(dram_val, 0.0)
    return idle_power


def _auto_max_power_data() -> dict[int, float]:
    """Auto-detect max DRAM power caps from RAPL domains."""
    power_data: dict[int, float] = {}

    for pkg_caps in collect_rapl_caps():
        package_id = int(pkg_caps.package)
        if not pkg_caps.dram:
            continue

        chosen: float | None = None
        for dram_cap in pkg_caps.dram:
            domain_max = _to_float(dram_cap.max_cap_w)
            domain_limit = _to_float(dram_cap.power_limit_w)
            candidate = domain_max if domain_max is not None else domain_limit
            if candidate is None:
                continue
            chosen = candidate if chosen is None else max(chosen, candidate)

        if chosen is None:
            continue

        power_data[package_id] = _safe_float(chosen, 0.0)

    return power_data


def _combine_power_data(idle_power: dict[int, float], max_power: dict[int, float]) -> dict[int, dict[str, float]]:
    """Merge idle/max maps into a per-socket dict."""
    combined: dict[int, dict[str, float]] = {}
    all_pkg_ids = set(idle_power.keys()) | set(max_power.keys())
    for pkg_id in all_pkg_ids:
        combined[pkg_id] = {
            "idle": idle_power.get(pkg_id, 0.0),
            "max": max_power.get(pkg_id, 0.0),
        }
    return combined


def _to_float(value: str | float | None) -> float | None:
    """Safely convert power values to float."""
    if value is None:
        return None
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: float | str | None, default: float = 0.0) -> float:
    """Convert to float with a fallback default."""
    try:
        converted = _to_float(value)
        return converted if converted is not None else default
    except Exception:
        return default


_MEM_TOTAL_PATTERN = re.compile(r"MemTotal:\s+(\d+)\s+kB")


def _mem_total_mb(meminfo_path: pathlib.Path) -> float:
    """Read MemTotal in MB from /proc/meminfo or a NUMA node's meminfo, both of which use the same line format."""
    try:
        match = _MEM_TOTAL_PATTERN.search(meminfo_path.read_text())
    except OSError:
        return 0.0
    return int(match.group(1)) / 1024 if match else 0.0


def get_memory_per_package(
    sysfs_root: pathlib.Path = pathlib.Path("/sys"),
    proc_meminfo: pathlib.Path = pathlib.Path("/proc/meminfo"),
) -> dict[int, float]:
    """
    Usable memory in MB per CPU socket, summed over the NUMA nodes attached to that socket.

    Kernels built without NUMA expose no nodes at all; the whole machine is then one socket.
    """
    node_dir = sysfs_root / "devices/system/node"
    memory_per_package = {
        package_id: sum(_mem_total_mb(node_dir / f"node{node_id}" / "meminfo") for node_id in nodes)
        for package_id, nodes in get_nodes_per_package(sysfs_root).items()
    }

    if not memory_per_package:
        logger.debug("Kernel exposes no NUMA nodes, treating the machine as a single socket")
        return {0: _mem_total_mb(proc_meminfo)}

    return memory_per_package


def get_info(config: dict) -> list[dict]:
    """
    Retrieves memory information per CPU socket from sysfs.

    Returns
    -------
    list of dict
        One entry per socket, each containing:
        - package_id (str): Socket the memory is attached to, as the kernel numbers it.
        - total_memory (float): Usable memory in MB.
        - power_mode (str): "auto" or "manual".
        - idle_power (float): Idle DRAM power in watts for the package.
        - max_power (float): Max DRAM power cap in watts for the package.
    """
    power_mode, power_data = _resolve_power_data(config.get("power", "auto"))
    sockets = []

    for package_id, total_memory in sorted(get_memory_per_package().items()):
        if power_mode == "manual" and package_id not in power_data:
            logger.warning(
                f"DRAM package {package_id} discovered on hardware but not in manual power config. "
                "Defaulting idle_power=0.0 and max_power=0.0."
            )

        package_power = power_data.get(package_id, {"idle": 0.0, "max": 0.0})
        sockets.append(
            {
                "package_id": str(package_id),
                "total_memory": total_memory,
                "power_mode": power_mode,
                "idle_power": package_power.get("idle", 0.0),
                "max_power": package_power.get("max", 0.0),
            }
        )

    return sockets
