import re
import shutil
import subprocess
from pathlib import Path

from loguru import logger

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import collect_gpu_caps


def get_info(config: dict) -> list[dict]:
    """Return GPU details with idle/max power, honoring manual or auto config."""
    gpus = _list_nvidia_gpus()

    if not gpus:
        return [{"gpu": "No GPU detected"}]

    power_config = config.get("power", "auto")
    power_mode, power_data = _resolve_power_data(power_config, gpus)

    result = []
    for gpu in gpus:
        gpu_id = str(gpu["id"])
        power_entry = power_data.get(gpu_id, {})
        result.append(
            {
                "manufacturer": gpu["manufacturer"],
                "model": gpu["model"],
                "serial_number": gpu["uuid"],
                "uuid": gpu["uuid"],
                "pci_id": gpu["pci_id"],
                "memory_mb": gpu["memory_mb"],
                "compute_capability_major": gpu.get("compute_capability_major", 0),
                "compute_capability_minor": gpu.get("compute_capability_minor", 0),
                "power_mode": power_mode,
                "idle_power": power_entry.get("idle", 0.0),
                "max_power": power_entry.get("max", 0.0),
            }
        )

    return result


def _list_nvidia_gpus() -> list[dict]:
    """Query NVIDIA GPUs via nvidia-smi and return normalized metadata.

    Returns
    -------
    list[dict]
        List of GPU info dicts with keys: id, manufacturer, model, uuid,
        pci_id, memory_mb, compute_capability_major, compute_capability_minor.
    """
    if not shutil.which("nvidia-smi"):
        return []

    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,uuid,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError as exc:
        logger.warning(f"nvidia-smi failed: {exc}")
        return []
    except FileNotFoundError:
        return []

    gpus = []
    for idx, line in enumerate(filter(None, output.split("\n"))):
        try:
            name, memory_total, uuid, pci_id, compute_cap = [part.strip() for part in line.split(",")]
            memory_mb = _safe_float(memory_total, 0.0)
            cc_major, cc_minor = _parse_compute_capability(compute_cap)
        except ValueError:
            # Unexpected format; skip this entry but keep others
            logger.warning(f"Unexpected nvidia-smi line: {line!r}")
            continue

        gpus.append(
            {
                "id": idx,
                "manufacturer": "NVIDIA",
                "model": name,
                "uuid": uuid,
                "pci_id": pci_id,
                "memory_mb": memory_mb,
                "compute_capability_major": cc_major,
                "compute_capability_minor": cc_minor,
            }
        )

    return gpus


def _parse_compute_capability(compute_cap: str) -> tuple[int, int]:
    """Parse compute capability string (e.g., '8.6') into major/minor ints.

    Parameters
    ----------
    compute_cap : str
        Compute capability string from nvidia-smi (e.g., '8.6', '7.5').

    Returns
    -------
    tuple[int, int]
        (major, minor) version numbers. Returns (0, 0) on parse failure.
    """
    try:
        parts = compute_cap.split(".")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, AttributeError):
        pass
    logger.warning(f"Failed to parse compute capability: {compute_cap!r}")
    return 0, 0


def _resolve_power_data(power_config: dict | str, gpus: list[dict]) -> tuple[str, dict[str, dict[str, float]]]:
    """Return (mode, per-GPU power data) using manual or auto sources."""
    if isinstance(power_config, str) and power_config.lower() == "auto":
        return "auto", _combine_power_data(_auto_idle_power_data(gpus), _auto_max_power_data())

    manual_entries = _normalize_manual_power_entries(power_config)
    if not manual_entries:
        logger.warning(f"Unknown GPU power config '{power_config}', falling back to auto")
        return "auto", _combine_power_data(_auto_idle_power_data(gpus), _auto_max_power_data())

    idle_auto: dict[str, float] | None = None
    max_auto: dict[str, float] | None = None

    resolved: dict[str, dict[str, float]] = {}
    for gpu in gpus:
        gpu_id = str(gpu["id"])
        manual = manual_entries.get(gpu_id, {})

        idle_val = manual.get("idle", "auto")
        if idle_val == "auto" or idle_val is None:
            if idle_auto is None:
                idle_auto = _auto_idle_power_data(gpus)
            idle_val = idle_auto.get(gpu_id, 0.0)
        else:
            idle_val = _safe_float(idle_val, 0.0)

        max_val = manual.get("max", "auto")
        if max_val == "auto" or max_val is None:
            if max_auto is None:
                max_auto = _auto_max_power_data()
            max_val = max_auto.get(gpu_id, 0.0)
        else:
            max_val = _safe_float(max_val, 0.0)

        resolved[gpu_id] = {"idle": idle_val, "max": max_val}

    return "manual", resolved


def _normalize_manual_power_entries(power_config) -> dict[str, dict[str, float | str]]:
    """Normalize manual power entries from config_schema or legacy dicts."""
    entries: dict[str, dict[str, float | str]] = {}

    if not isinstance(power_config, list):
        return entries

    for entry in power_config:
        gpu_id = None
        idle_power = "auto"
        max_power = "auto"

        if hasattr(entry, "id"):
            gpu_id = entry.id
            idle_power = getattr(entry, "idle_power", "auto")
            max_power = getattr(entry, "max_power", "auto")
        elif isinstance(entry, dict):
            gpu_id = entry.get("id", entry.get("ID"))
            idle_power = entry.get("idle_power", entry.get("IDLE_POWER", "auto"))
            max_power = entry.get("max_power", entry.get("MAX_POWER", "auto"))

        if gpu_id is None:
            continue

        entries[str(gpu_id)] = {"idle": idle_power, "max": max_power}

    return entries


def _auto_idle_power_data(gpus: list[dict]) -> dict[str, float]:
    """Measure idle GPU power using the idle_gpu.sh script if available."""
    script_path = Path(__file__).parent / "idle_gpu.sh"
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "gpu-idle.log"

    if not log_path.exists() and script_path.exists():
        cmd = ["sudo", str(script_path), "60", "60", str(log_dir)]
        try:
            logger.debug(f"Running GPU idle measurement script: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as exc:
            logger.warning(f"Unable to execute GPU idle measurement script: {exc}")

    measurements: dict[str, float] = {str(gpu["id"]): 0.0 for gpu in gpus}
    if not log_path.exists():
        return measurements

    pattern = re.compile(r"GPU\[(\d+)\].*:\s*([0-9.]+)\s+W")
    try:
        with log_path.open() as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    gpu_id = match.group(1)
                    measurements[gpu_id] = _safe_float(match.group(2), 0.0)
    except (OSError, ValueError):
        return measurements

    return measurements


def _auto_max_power_data() -> dict[str, float]:
    """Detect GPU max power caps via nvidia-smi using helper script."""
    _, caps = collect_gpu_caps()

    power_caps: dict[str, float] = {}
    for cap in caps:
        max_cap = _safe_float(cap.max_cap_w, -1)
        enforced = _safe_float(cap.enforced_w, -1)
        chosen = max_cap if max_cap != -1 else enforced
        power_caps[str(cap.index)] = chosen if chosen != -1 else 0.0

    return power_caps


def _combine_power_data(idle_power: dict[str, float], max_power: dict[str, float]) -> dict[str, dict[str, float]]:
    """Merge idle and max power maps keyed by GPU ID."""
    combined: dict[str, dict[str, float]] = {}
    for gpu_id in set(idle_power.keys()) | set(max_power.keys()):
        combined[gpu_id] = {
            "idle": idle_power.get(gpu_id, 0.0),
            "max": max_power.get(gpu_id, 0.0),
        }
    return combined


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float with a fallback."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
