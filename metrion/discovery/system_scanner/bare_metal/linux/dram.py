import re
import subprocess

from loguru import logger

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import collect_rapl_caps
from metrion.discovery.system_scanner.bare_metal.linux.util import get_cpu_idle_power_data


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


def get_info(config: dict) -> list[dict]:
    """
    Retrieves memory information, including details for multiple RAM modules using dmidecode.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary contains details of an individual RAM module:
        - manufacturer (str): The manufacturer of the RAM module.
        - mhz (int or str): Speed of the RAM in MT/s (effectively MHz in DDR memory).
        - total_memory (int or str): Size of the RAM module in MB.
        - hardware_id (str): Serial number of the RAM module.
        - package_id (str): The socket or locator information of the RAM module.
        - power_mode (str): "auto" or "manual".
        - idle_power (float): Idle DRAM power in watts for the package.
        - max_power (float): Max DRAM power cap in watts for the package.
    """
    power_mode, power_data = _resolve_power_data(config.get("power", "auto"))
    ram_modules = []

    try:
        output = subprocess.check_output(["sudo", "dmidecode", "--type", "17"], text=True)
        # https://www.redhat.com/en/blog/linux-tools-dmidecode
        modules = output.split("Memory Device")[1:]

        for module in modules:
            manufacturer = "Unknown"
            speed = "Unknown"  # Unit: MT/s (Megatransfers per second)
            size = "Unknown"  # Unit: MB (Megabytes) or GB (Gigabytes)
            serial_number = "Unknown"
            package_id = "Unknown"
            package_power = {"idle": 0.0, "max": 0.0}

            match_manufacturer = re.search(r"Manufacturer:\s*(.+)", module)
            match_speed = re.search(r"Speed:\s*(\d+)\s*MT/s", module)
            match_size = re.search(r"Size:\s*(\d+)\s*(GB|MB)", module)
            match_serial = re.search(r"Serial Number:\s*(\S+)", module)
            match_socket = re.search(r"(?:Bank Locator):\s*(P\d)", module)
            match_locator = re.search(r"Locator:\s*(\S+)", module)

            if match_manufacturer:
                manufacturer = match_manufacturer.group(1)
            if match_speed:
                speed = int(match_speed.group(1))  # Value in MT/s
            if match_size:
                # Convert size to MB (Megabytes) if in GB
                size = int(match_size.group(1)) * (1024 if match_size.group(2) == "GB" else 1)
            if match_serial:
                serial_number = match_serial.group(1)
            if match_socket:
                # Extract only the digit(s) from a locator like 'P0' or 'P10'
                digits = re.search(r"\d+", match_socket.group(1))
                package_id = digits.group(0) if digits else "Unknown"
            if match_locator and package_id == "Unknown":
                full_locator = match_locator.group(1)  # e.g., "A1" or "CPU1_DIMM_A1"
                # Heuristic A: Look for explicit "P0" or "CPU1" in the string
                if "P0" in full_locator or "CPU0" in full_locator:
                    package_id = "0"
                elif "P1" in full_locator or "CPU1" in full_locator:
                    package_id = "1"
                # Heuristic B: Mapping based on prefix letter (A=0, B=1)
                # This matches your specific hardware output where A1-A8 and B1-B8 are used.
                else:
                    prefix = full_locator[0].upper()
                    if prefix.isalpha():
                        package_id = str(ord(prefix) - ord("A"))

            if isinstance(package_id, str) and package_id.isdigit():
                pkg_idx = int(package_id)
                package_power = power_data.get(pkg_idx, {"idle": 0.0, "max": 0.0})
                if power_mode == "manual" and pkg_idx not in power_data:
                    logger.warning(
                        f"DRAM package {pkg_idx} discovered on hardware but not in manual power config. "
                        "Defaulting idle_power=0.0 and max_power=0.0."
                    )

            if manufacturer != "Unknown":
                ram_modules.append(
                    {
                        "manufacturer": manufacturer,
                        "mhz": speed if isinstance(speed, int) else "Unknown",  # MT/s (effectively MHz in DDR memory)
                        "total_memory": size if isinstance(size, int) else "Unknown",  # Size in MB
                        "hardware_id": serial_number,  # Serial number of the RAM module
                        "package_id": package_id,
                        "power_mode": power_mode,
                        "idle_power": package_power.get("idle", 0.0),
                        "max_power": package_power.get("max", 0.0),
                    }
                )

    except subprocess.CalledProcessError:
        pass

    return ram_modules
