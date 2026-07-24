import pathlib
import re
import shutil
import subprocess

from loguru import logger


def parse_cpu_list(cpu_list_str: str) -> tuple[int, ...]:
    """
    Parse a CPU list string into a tuple of integers.

    Handles various formats:
    - Single CPU: "0" → (0,)
    - CPU range: "0-3" → (0, 1, 2, 3)
    - Comma-separated values: "0,2,4" → (0, 2, 4)
    - Complex combinations: "0-2,4,6-8" → (0, 1, 2, 4, 6, 7, 8)

    Parameters
    ----------
    cpu_list_str : str
        String representation of CPU list

    Returns
    -------
    tuple
        Tuple of integers representing CPU IDs

    Raises
    ------
    ValueError
        If the CPU list format is invalid
    """
    if not cpu_list_str or not cpu_list_str.strip():
        return tuple()

    result = set()
    for part in cpu_list_str.split(","):
        if "-" in part:
            try:
                start, end = map(int, part.split("-"))
                result.update(range(start, end + 1))
            except ValueError as err:
                raise ValueError(f"Invalid CPU range format: '{part}'") from err
        else:
            try:
                result.add(int(part))
            except ValueError as err:
                raise ValueError(f"Invalid CPU ID: '{part}'") from err

    return tuple(sorted(result))


def run_rapl_idle_measurement(dwell_time=60, extra_units=None):
    """
    Run the freeze-idle-rapl.sh script to measure idle power consumption.

    Parameters
    ----------
    dwell_time : int, optional
        Time in seconds to freeze and measure, by default 60
    extra_units : list, optional
        List of extra systemd units/slices to freeze, by default None

    Returns
    -------
    subprocess.CompletedProcess
        The result of the script execution
    """
    current_dir = pathlib.Path(__file__).parent.absolute()
    log_folder = current_dir / "log"
    script_path = current_dir / "freeze-idle-rapl.sh"

    if not shutil.which("systemd-run"):
        raise RuntimeError(
            "systemd-run not found. The idle RAPL measurement requires systemd. "
            "This system may be using a non-systemd init (OpenRC, runit, SysVinit, etc.) "
            "or is running inside a container without systemd access."
        )

    if not script_path.exists():
        raise FileNotFoundError(f"freeze-idle-rapl.sh not found at expected path: {script_path}")

    # Probe D-Bus / systemd availability before launching the long-running command
    try:
        probe = subprocess.run(
            ["sudo", "systemctl", "is-system-running"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logger.debug(f"systemd state: '{probe.stdout.strip()}' (rc={probe.returncode})")
    except Exception as probe_exc:
        logger.warning(
            f"systemd health probe failed — D-Bus may be unavailable: {probe_exc} "
            f"(errno={getattr(probe_exc, 'errno', None)})"
        )

    log_folder.mkdir(exist_ok=True)
    log_file = log_folder / "rapl-idle.log"

    # Start from a clean log to avoid parsing stale data
    try:
        log_file.unlink()
    except FileNotFoundError:
        pass

    cmd = [
        "sudo",
        "systemd-run",
        "--unit=rapl-idle",
        "--collect",
        "--wait",
        "--property=StandardOutput=file:" + str(log_file),
        str(script_path),
        str(dwell_time),
        str(log_folder),
    ]

    if extra_units:
        cmd.extend(extra_units)
    else:
        cmd.extend(["kubepods.slice", "k3s.service", "containerd.service"])

    timeout = dwell_time + 120
    logger.debug(f"Starting RAPL idle measurement for {dwell_time}s (timeout {timeout}s), cmd: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
        logger.debug("RAPL idle measurement completed successfully.")
        if result.stdout:
            logger.debug(f"systemd-run stdout: {result.stdout.strip()}")
        if result.stderr:
            logger.debug(f"systemd-run stderr: {result.stderr.strip()}")
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Timed out after {timeout}s waiting for RAPL idle measurement")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(
            f"systemd-run failed (exit code {e.returncode})\n"
            f"  stdout: {e.stdout.strip() if e.stdout else '(empty)'}\n"
            f"  stderr: {e.stderr.strip() if e.stderr else '(empty)'}"
        )
        raise
    except OSError as e:
        import errno as errno_mod

        hints = {
            errno_mod.ENODATA: "D-Bus socket has no data, systemd may not be running or the session bus is unavailable",
            errno_mod.ENOENT: "sudo or systemd-run binary not found at runtime despite PATH check",
            errno_mod.EACCES: "Permission denied, check sudo rules for systemd-run",
            errno_mod.EPERM: "Operation not permitted, cgroup or namespace restrictions may be blocking systemd-run",
        }
        hint = hints.get(e.errno, "unexpected OS error")
        logger.error(f"OSError launching systemd-run (errno={e.errno}): {e}\n  Hint: {hint}")
        raise


# Accepts "digits", "digits.digits", or "digits,digits" -- a single optional
# decimal separator, either '.' or ','. Deliberately rejects mixed/repeated
# separators, signs, and non-numeric tokens so a locale-mangled or otherwise
# corrupted value is never silently misinterpreted. Locale-independent: does
# not use locale.atof() or read the process locale.
_RAPL_NUMBER_RE = re.compile(r"^[0-9]+(?:[.,][0-9]+)?$")


def _parse_rapl_watt_value(token: str, *, field: str, line: str) -> float:
    """Parse a single RAPL power token deterministically, independent of locale."""
    if not _RAPL_NUMBER_RE.match(token):
        raise ValueError(f"Malformed RAPL {field!r} value {token!r} in line: {line!r}")
    return float(token.replace(",", "."))


def parse_rapl_idle_log(log_path: str | pathlib.Path | None = None) -> dict[int, dict[str, float | None]]:
    """
    Parse the freeze-idle-rapl.sh log file and extract idle power consumption per package.

    Parameters
    ----------
    log_path : str or pathlib.Path, optional
        Path to the rapl-idle.log file. If None, uses the default log location,
        by default None

    Returns
    -------
    dict[int, dict[str, float | None]]
        Dictionary with package ID as key and power measurements as nested dict.
        Structure:
        {
            package_id: {
                'package': power_in_watts,      # Package total power (float, never None)
                'cpu': power_in_watts,           # CPU cores power (float or None)
                'integrated-gpu': power_in_watts, # GPU power (float or None)
                'dram': power_in_watts           # DRAM power (float or None)
            }
        }

    Notes
    -----
    Power values accept a dot or comma decimal separator (e.g. "10.0196" or
    "10,0196"); mixed/repeated separators or other non-numeric content raise
    ValueError. "Package Total" has no legitimate "N/A" form in the source
    script and therefore always raises rather than defaulting to None or 0.0
    if malformed. "CPU Cores" / "Integrated GPU" / "DRAM" keep their existing
    contract: "N/A" (domain not present on this hardware) parses to None, but
    any other malformed value for a present line still raises.

    Raises
    ------
    FileNotFoundError
        If the RAPL idle log file is not found
    ValueError
        If no package power data is found in the log file, or if a recognized
        labelled field is present with a malformed, empty, or ambiguous value

    Examples
    --------
    >>> result = parse_rapl_idle_log()
    >>> print(result)
    {
        0: {'package': 11.38, 'cpu': 8.52, 'integrated-gpu': 1.23, 'dram': 4.56},
        1: {'package': 10.90, 'cpu': 8.22, 'integrated-gpu': None, 'dram': 3.85}
    }
    """
    # Determine log path
    if log_path is None:
        current_dir = pathlib.Path(__file__).parent.absolute()
        log_path = current_dir / "log" / "rapl-idle.log"
    else:
        log_path = pathlib.Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"RAPL idle log file not found: {log_path}")

    # Dictionary to store results
    packages = {}
    current_package_id = None

    # Read and parse the log file
    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Look for package section headers: "=== Package N ==="
            package_header_match = re.match(r"^=== Package (\d+) ===$", line)
            if package_header_match:
                current_package_id = int(package_header_match.group(1))
                packages[current_package_id] = {"cpu": None, "integrated-gpu": None, "dram": None}
                continue

            # Skip if we're not in a package section
            if current_package_id is None:
                continue

            # Package Total      :  11.3824 W   (never legitimately "N/A")
            if line.startswith("Package Total"):
                package_match = re.match(r"^Package Total\s*:\s*(\S+)\s*W$", line)
                if not package_match:
                    raise ValueError(f"Malformed 'Package Total' line for package {current_package_id}: {line!r}")
                packages[current_package_id]["package"] = _parse_rapl_watt_value(
                    package_match.group(1), field="Package Total", line=line
                )
                continue

            # CPU Cores: "CPU Cores          :    85.20 W" or "CPU Cores          :      N/A"
            if line.startswith("CPU Cores"):
                cpu_match = re.match(r"^CPU Cores\s*:\s*(\S+)\s*W$", line)
                if cpu_match:
                    packages[current_package_id]["cpu"] = _parse_rapl_watt_value(
                        cpu_match.group(1), field="CPU Cores", line=line
                    )
                elif re.match(r"^CPU Cores\s*:\s*N/A$", line):
                    pass  # domain not available on this hardware -- keep None
                else:
                    raise ValueError(f"Malformed 'CPU Cores' line for package {current_package_id}: {line!r}")
                continue

            # Integrated GPU: "Integrated GPU     :    12.30 W" or "Integrated GPU     :      N/A"
            if line.startswith("Integrated GPU"):
                gpu_match = re.match(r"^Integrated GPU\s*:\s*(\S+)\s*W$", line)
                if gpu_match:
                    packages[current_package_id]["integrated-gpu"] = _parse_rapl_watt_value(
                        gpu_match.group(1), field="Integrated GPU", line=line
                    )
                elif re.match(r"^Integrated GPU\s*:\s*N/A$", line):
                    pass  # domain not available on this hardware -- keep None
                else:
                    raise ValueError(f"Malformed 'Integrated GPU' line for package {current_package_id}: {line!r}")
                continue

            # DRAM: "DRAM               :    45.60 W" or "DRAM               :      N/A"
            if line.startswith("DRAM"):
                dram_match = re.match(r"^DRAM\s*:\s*(\S+)\s*W$", line)
                if dram_match:
                    packages[current_package_id]["dram"] = _parse_rapl_watt_value(
                        dram_match.group(1), field="DRAM", line=line
                    )
                elif re.match(r"^DRAM\s*:\s*N/A$", line):
                    pass  # domain not available on this hardware -- keep None
                else:
                    raise ValueError(f"Malformed 'DRAM' line for package {current_package_id}: {line!r}")
                continue

            # Stop parsing when we reach system totals
            if line.startswith("=== System Totals ==="):
                break

    if not packages:
        raise ValueError(f"No package power data found in log file: {log_path}")

    return packages


def get_cpu_idle_power_data(log_path: str | pathlib.Path | None = None) -> dict[int, dict[str, float | None]]:
    if log_path is None:
        current_dir = pathlib.Path(__file__).parent.absolute()
        log_path = current_dir / "log" / "rapl-idle.log"
    else:
        log_path = pathlib.Path(log_path)

    if not log_path.exists():
        run_rapl_idle_measurement()

    if not log_path.exists():
        raise FileNotFoundError(f"RAPL idle log file not found: {log_path}")

    energy_data = parse_rapl_idle_log(log_path)
    if not energy_data:
        raise ValueError("No package power data found in the log file")
    return energy_data
