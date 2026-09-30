import pathlib


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


def _package_of_cpu(sysfs_root: pathlib.Path, cpu_id: int) -> int | None:
    """Socket a logical CPU belongs to, or None if the kernel does not say."""
    pkg_file = sysfs_root / "devices/system/cpu" / f"cpu{cpu_id}" / "topology/physical_package_id"
    try:
        return int(pkg_file.read_text().strip())
    except (OSError, ValueError):
        return None


def get_nodes_per_package(sysfs_root: pathlib.Path = pathlib.Path("/sys")) -> dict[int, list[int]]:
    """
    NUMA node ids per CPU socket.

    Sub-NUMA clustering (Intel SNC, AMD NPS) splits one socket into several nodes, so a socket
    owns a list rather than a single node. Nodes without CPUs are memory-only, e.g. CXL or
    persistent memory, and belong to no socket.
    """
    nodes_per_package: dict[int, list[int]] = {}

    for node_path in sorted((sysfs_root / "devices/system/node").glob("node[0-9]*")):
        try:
            cpus = parse_cpu_list((node_path / "cpulist").read_text().strip())
        except (OSError, ValueError):
            continue

        if not cpus:
            continue

        package_id = _package_of_cpu(sysfs_root, cpus[0])
        if package_id is None:
            continue

        nodes_per_package.setdefault(package_id, []).append(int(node_path.name.removeprefix("node")))

    return nodes_per_package
