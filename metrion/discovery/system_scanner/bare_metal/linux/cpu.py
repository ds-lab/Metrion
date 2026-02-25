import os
import pathlib

import polars as pl
from loguru import logger

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import collect_rapl_caps
from metrion.discovery.system_scanner.bare_metal.linux.util import get_cpu_idle_power_data, parse_cpu_list


def parse_cpuinfo() -> dict[int, dict]:
    """
    Parse /proc/cpuinfo and return CPU information grouped by physical socket.

    Returns:
        Dictionary mapping physical CPU ID to CPU information
    """
    cpuinfo_path = "/proc/cpuinfo"
    if not os.path.exists(cpuinfo_path):
        raise FileNotFoundError(f"{cpuinfo_path} not found. This script only runs on Linux.")

    with open(cpuinfo_path) as f:
        content = f.read().strip()

    if not content:
        return {}

    cpu_entries = content.split("\n\n")
    cpus_by_socket = {}

    for entry in cpu_entries:
        if not entry.strip():
            continue

        cpu_info = {}
        for line in entry.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                cpu_info[key.strip()] = value.strip()

        physical_id = int(cpu_info.get("physical id", "-1"))

        if physical_id == -1:
            logger.warning(f"Missing 'physical id' in CPU info: {cpu_info}")
            continue

        if physical_id not in cpus_by_socket:
            cpus_by_socket[physical_id] = {
                "physical_id": physical_id,  # This is the socket ID
                "model_name": cpu_info.get("model name", "Unknown"),
                "vendor": cpu_info.get("vendor_id", "Unknown"),
                "physical_cores": int(
                    cpu_info.get("cpu cores", "0")
                ),  # Physical cores per socket without hyperthreading
                "cache_size_kb": _parse_cache_size(cpu_info.get("cache size", "0")),
                "flags": cpu_info.get("flags", "Unknown"),
                "logical_cpus": [],
            }

        # Add logical CPU info
        cpus_by_socket[physical_id]["logical_cpus"].append(
            {"id": int(cpu_info.get("processor", "-1")), "frequency_mhz": float(cpu_info.get("cpu MHz", "0"))}
        )

    return cpus_by_socket


def _parse_cache_size(cache_str):
    """Helper function to parse cache size string."""
    try:
        return int(cache_str.split()[0]) if cache_str and cache_str != "Unknown" else 0
    except (ValueError, IndexError):
        return 0


def get_cpu_topology() -> tuple[pl.DataFrame, dict[int, int]]:
    """
    Read CPU topology from /sys/devices/system/cpu.

    Returns
    -------
    tuple
        A tuple containing:
        - pl.DataFrame: DataFrame with CPU topology information.
        - dict[int, int]: Mapping from CPU ID to its thread sibling ID.
    """
    cpu_base_path = pathlib.Path("/sys/devices/system/cpu")
    if not cpu_base_path.exists():
        return pl.DataFrame(), {}

    topology_data = []

    for cpu_path in cpu_base_path.glob("cpu[0-9]*"):
        cpu_id = int(cpu_path.name.replace("cpu", ""))
        topo_path = cpu_path / "topology"

        cpu_topo: dict = {"CPU": cpu_id}

        # Read topology files
        try:
            pkg_file = topo_path / "physical_package_id"
            if pkg_file.exists():
                cpu_topo["Package"] = int(pkg_file.read_text().strip())
            else:
                cpu_topo["Package"] = 0

            core_file = topo_path / "core_id"
            if core_file.exists():
                cpu_topo["Core ID"] = int(core_file.read_text().strip())
            else:
                cpu_topo["Core ID"] = 0

            sibling_file = topo_path / "thread_siblings_list"
            if sibling_file.exists():
                cpu_topo["Thread Siblings"] = parse_cpu_list(sibling_file.read_text().strip())
            else:
                cpu_topo["Thread Siblings"] = (cpu_id,)

            core_sibling_file = topo_path / "core_siblings_list"
            if core_sibling_file.exists():
                cpu_topo["Core Siblings"] = parse_cpu_list(core_sibling_file.read_text().strip())
            else:
                cpu_topo["Core Siblings"] = (cpu_id,)

        except (FileNotFoundError, ValueError):
            # Set defaults for missing data
            cpu_topo["Package"] = 0
            cpu_topo["Core ID"] = 0
            cpu_topo["Thread Siblings"] = (cpu_id,)
            cpu_topo["Core Siblings"] = (cpu_id,)

        topology_data.append(cpu_topo)

    if not topology_data:
        return pl.DataFrame(), {}

    # Build thread sibling mapping
    sibling_map = {}
    for entry in topology_data:
        cpu_id = entry["CPU"]
        siblings = entry["Thread Siblings"]
        # Find first sibling that's not this CPU
        sibling_map[cpu_id] = next((s for s in siblings if s != cpu_id), -1)

    return pl.DataFrame(topology_data).sort("CPU"), sibling_map


def get_info(config: dict) -> list[dict]:
    """Return CPU socket details with idle/max package power."""
    cpus_by_socket = parse_cpuinfo()
    topology_df, sibling_map = get_cpu_topology()

    include_logical_cpus = True
    create_core_entities = True

    power_config = config.get("power", "auto")
    power_mode, power_data = _resolve_power_data(power_config, cpus_by_socket)

    result = []
    for _socket_id, socket_data in cpus_by_socket.items():
        cpu_info = _build_socket_info(
            socket_data,
            topology_df,
            sibling_map,
            include_logical_cpus,
            create_core_entities,
            power_mode,
            power_data,
        )
        result.append(cpu_info)

    return result


def _resolve_power_data(
    power_config: dict | list | str, cpus_by_socket: dict[int, dict]
) -> tuple[str, dict[int, dict[str, float]]]:
    """Return (mode, per-socket power data) using manual or auto sources."""
    if isinstance(power_config, str):
        power_config = power_config.lower()

    if power_config == "auto":
        return "auto", _combine_power_data(_auto_idle_power_data(cpus_by_socket), _auto_max_power_data(cpus_by_socket))

    manual_entries: dict[int, dict[str, float | str | None]] = {}

    if isinstance(power_config, list):
        for entry in power_config:
            package_id = int(entry.get("ID", -1))
            manual_entries[package_id] = {
                "idle": entry.get("IDLE_POWER", "auto"),
                "max": entry.get("MAX_POWER", "auto"),
            }
    else:
        logger.warning(f"Unknown power config '{power_config}', falling back to auto")
        return "auto", _combine_power_data(_auto_idle_power_data(cpus_by_socket), _auto_max_power_data(cpus_by_socket))

    idle_auto: dict[int, float] | None = None
    max_auto: dict[int, float] | None = None

    resolved: dict[int, dict[str, float]] = {}
    for pkg_id in cpus_by_socket.keys():
        manual = manual_entries.get(pkg_id)

        if manual is None:
            # Socket exists on hardware but not in manual config — default to 0.0
            logger.warning(
                f"CPU package {pkg_id} discovered on hardware but not in manual power config. "
                "Defaulting idle_power=0.0 and max_power=0.0."
            )
            resolved[pkg_id] = {"idle": 0.0, "max": 0.0}
            continue

        idle_val = manual.get("idle", "auto")
        if idle_val == "auto" or idle_val is None:
            if idle_auto is None:
                idle_auto = _auto_idle_power_data(cpus_by_socket)
            idle_val = idle_auto.get(pkg_id, 0.0)
        else:
            idle_val = _safe_float(idle_val, 0.0)

        max_val = manual.get("max", "auto")
        if max_val == "auto" or max_val is None:
            if max_auto is None:
                max_auto = _auto_max_power_data(cpus_by_socket)
            max_val = max_auto.get(pkg_id, 0.0)
        else:
            max_val = _safe_float(max_val, 0.0)

        resolved[pkg_id] = {"idle": idle_val, "max": max_val}

    return "manual", resolved


def _auto_idle_power_data(cpus_by_socket: dict[int, dict]) -> dict[int, float]:
    """Auto-detect idle package power using RAPL."""
    try:
        rapl_data = get_cpu_idle_power_data()
    except Exception as exc:
        logger.warning(
            f"Could not measure idle power via RAPL (errno {getattr(exc, 'errno', None)}): {exc}. Defaulting to 0.0 W."
        )
        return {pkg_id: 0.0 for pkg_id in cpus_by_socket.keys()}
    idle_power: dict[int, float] = {}
    for pkg_id in cpus_by_socket.keys():
        socket_data = rapl_data.get(pkg_id, {})
        idle_val = socket_data.get("package", 0.0)
        idle_power[pkg_id] = _safe_float(idle_val, 0.0)
    return idle_power


def _auto_max_power_data(cpus_by_socket: dict[int, dict]) -> dict[int, float]:
    """Auto-detect max package power caps from RAPL domains."""
    power_data: dict[int, float] = {pkg_id: 0.0 for pkg_id in cpus_by_socket.keys()}

    for pkg_caps in collect_rapl_caps():
        package_id = int(pkg_caps.package)
        if package_id not in power_data:
            continue

        package_max = _to_float(pkg_caps.max_cap_w)
        package_limit = _to_float(pkg_caps.power_limit_w)

        # Prefer explicit max cap, fall back to enforced limit
        chosen = package_max if package_max is not None else package_limit
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


def _build_socket_info(
    socket_data: dict,
    topology_df: pl.DataFrame,
    sibling_map: dict[int, int],
    include_logical_cpus: bool,
    create_core_entities: bool,
    power_mode: str,
    power_data: dict[int, dict[str, float]] | None = None,
) -> dict:
    """Build CPU socket information dictionary."""
    logical_cpus = socket_data["logical_cpus"]
    logical_cpu_count = len(logical_cpus)

    # Build logical CPU list if requested
    logical_cpu_list = []
    if include_logical_cpus:
        logical_cpu_list = [{"id": cpu["id"], "frequency_mhz": cpu["frequency_mhz"]} for cpu in logical_cpus]

    # Get topology info for this socket's CPUs
    cpu_ids = [cpu["id"] for cpu in logical_cpus]
    socket_topology = {cpu_id: sibling_map.get(cpu_id, -1) for cpu_id in cpu_ids}
    idle_power = 0.0
    max_power = 0.0
    if power_data:
        package_power_data = power_data.get(socket_data["physical_id"], {})
        idle_power = package_power_data.get("idle", 0.0)
        max_power = package_power_data.get("max", 0.0)

    result = {
        "manufacturer": socket_data["vendor"],
        "model": socket_data["model_name"],
        "physical_cores": socket_data["physical_cores"],
        "logical_cores": logical_cpu_count,
        "package_id": str(socket_data["physical_id"]),
        "cache": socket_data["cache_size_kb"],
        "hyperthreading": logical_cpu_count > socket_data["physical_cores"],
        "logical_cpus": logical_cpu_list,
        "topology": socket_topology,
        "power_mode": power_mode,
        "idle_power_package": idle_power,
        "max_power_package": max_power,
    }

    return result
