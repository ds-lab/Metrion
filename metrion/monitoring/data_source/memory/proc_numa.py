# based on https://github.com/numactl/numactl/blob/b34da01b0017946a1a77057c247cfc6d231a0215/numastat.c#L952

import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import override

import polars as pl
from loguru import logger

from metrion.config_reader.config_schema import EbpfConfig, TimingConfig
from metrion.data_record import DataRecord
from metrion.discovery.system_scanner.bare_metal.linux.util import get_nodes_per_package
from metrion.entity.dram import DRAM
from metrion.entity.entity import Entity
from metrion.entity.logical_entity_registry import LogicalEntityRegistry
from metrion.measurement import Measurement
from metrion.metric import DRAM_SOCKET_PID_MEMORY, Metric
from metrion.monitoring.data_source.cpu.utils import get_pid_starttime_ns
from metrion.monitoring.data_source.data_source_base import DataSource

# Pre-compiled regex patterns for performance (only for numeric extraction)
_RE_NODE_PAGES = re.compile(r"N(\d+)=(\d+)")
_RE_KERNEL_PAGESIZE = re.compile(r"kernelpagesize_kB=(\d+)")

# Fast string-based category tokens (lowercase for matching)
_HUGE_TOKENS = ("huge", "hugetlb", "hugetlbfs", "thp", "anonhuge", "anon_hugepage")


def parse_numa_maps(lines: Iterable[str], default_page_kb: int, huge_page_kb: int) -> tuple[dict[int, float], float]:
    """
    Sum a process's memory per NUMA node from numa_maps lines, in MB, plus its total.

    Page sizes follow numastat: an explicit kernelpagesize_kB wins, otherwise a mapping that
    looks like huge pages is counted with the huge page size and everything else with the
    default one.
    """
    memory_per_node: dict[int, float] = {}
    total_mb = 0.0

    for line in lines:
        page_size_kb = default_page_kb
        pagesize_match = _RE_KERNEL_PAGESIZE.search(line)
        if pagesize_match:
            page_size_kb = int(pagesize_match.group(1))
        elif any(tok in line.lower() for tok in _HUGE_TOKENS):
            page_size_kb = huge_page_kb

        for match in _RE_NODE_PAGES.finditer(line):
            node_id = int(match.group(1))
            pages = int(match.group(2))
            mem_mb = (pages * page_size_kb) / 1024.0  # kB → MB

            memory_per_node[node_id] = memory_per_node.get(node_id, 0.0) + mem_mb
            total_mb += mem_mb

    return {node: memory_per_node[node] for node in sorted(memory_per_node)}, total_mb


class ProcNuma(DataSource):
    """
    A data source for NUMA (Non-Uniform Memory Access) memory statistics.

    This class collects NUMA memory usage data for all running processes by reading
    the `/proc/[pid]/numa_maps` file for each process. It can be used to
    determine how much memory each process is using on each NUMA node.

    Attributes
    ----------
    entities : Sequence[DRAM]
        A sequence of DRAM entities for which to collect data.
    _metric_lookup : dict[Metric, Callable[[], DataRecord]]
        A dictionary mapping metrics to the methods that collect them.
    """

    _entity_metric_mapping: dict[type[Entity], set[Metric]] = {DRAM: {DRAM_SOCKET_PID_MEMORY}}

    def __init__(
        self,
        entities: Sequence[DRAM],
        interval: float | None = None,
        enabled_metrics: list[Metric] | None = None,
        timing: TimingConfig | None = None,
        ebpf: EbpfConfig | None = None,
    ):
        """
        Initialize the ProcNuma data source.

        Parameters
        ----------
        entities : Sequence[DRAM]
            A sequence of DRAM entities to monitor.
        interval : float, optional
             The collection interval in seconds.
        """
        super().__init__(entities, interval=interval, enabled_metrics=enabled_metrics, timing=timing, ebpf=ebpf)
        self.entities = entities
        self._last_called = time.perf_counter()
        # check_entity_type(entities, expected_type=DRAM)
        self._logical_entity_registry = LogicalEntityRegistry.get_instance()
        self._nodes_per_package = get_nodes_per_package()

        self._metric_lookup: dict[Metric, Callable[[], DataRecord]] = {
            DRAM_SOCKET_PID_MEMORY: self._get_all_processes_numa_info,
        }

    def get_data(self, metrics: Sequence[Metric]) -> DataRecord:
        """
        Get data for the specified metrics.

        Parameters
        ----------
        metrics : Sequence[Metric]
            A sequence of metrics to collect data for.

        Returns
        -------
        DataRecord
            A DataRecord object containing the collected measurements and entities.
        """
        measurements: list[Measurement | pl.DataFrame] = []
        entities: list[Entity] = []
        unique_metrics = set(metrics)

        for metric in unique_metrics:
            if metric in self._metric_lookup:
                data_record = self._metric_lookup[metric]()
                measurements.extend(data_record.measurements)
                entities.extend(data_record.entities)
        self._last_called = time.perf_counter()

        return DataRecord(entities=entities, measurements=measurements)

    @classmethod
    def is_applicable(cls, entity: Entity) -> bool:
        """
        Check if the entity is applicable for this data source.

        Parameters
        ----------
        entity : Entity
            The entity to check.

        Returns
        -------
        bool
            True if the entity is a DRAM entity, False otherwise.
        """
        return type(entity) in cls._entity_metric_mapping.keys()

    def clean_up(self):
        pass

    def _iter_pids(self) -> list[int]:
        cands = []
        for e in os.scandir("/proc"):
            if not (e.is_dir() and e.name.isdigit()):
                continue
            pid = int(e.name)

            # fast zombie filter
            try:
                with open(f"/proc/{pid}/stat") as s:
                    # pid (comm) state ...
                    parts = s.read().split()
                    if len(parts) > 2 and parts[2] == "Z":
                        continue  # zombie
            except (FileNotFoundError, ProcessLookupError):
                continue  # raced

            # # require numa_maps to exist
            # if not os.path.exists(f"/proc/{pid}/numa_maps"):
            #     continue  # kernel thread or raced

            cands.append(pid)

        return cands

    def _get_system_page_sizes_kb(self) -> tuple[int, int]:
        """
        Get the system's default and huge page sizes in KiB.

        Returns
        -------
        tuple[int, int]
            A tuple containing the default page size and huge page size in KiB.
            (default_size_kb, huge_size_kb).
        """
        default_size_kb = os.sysconf("SC_PAGESIZE") // 1024
        huge_size_kb = 0  # Default to 0 if not found
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("Hugepagesize:"):
                        parts = line.split()
                        huge_size_kb = int(parts[1])
                        break
        except (OSError, IndexError, ValueError):
            pass  # If can't read or parse, huge_size_kb remains 0

        # Common default for huge pages is 2MB if not in meminfo for some reason
        if huge_size_kb == 0:
            huge_size_kb = 2048

        return default_size_kb, huge_size_kb

    def _get_numa_memory_for_pid(
        self, pid: int, default_page_kb: int, huge_page_kb: int
    ) -> tuple[dict[int, float] | None, float]:
        """Read /proc/<pid>/numa_maps and hand the lines to the parser."""
        try:
            with open(f"/proc/{pid}/numa_maps") as f:
                return parse_numa_maps(f, default_page_kb, huge_page_kb)
        except (FileNotFoundError, PermissionError, OSError):
            return None, 0.0

    def _memory_on_socket(self, numa_memory: dict[int, float], dram: DRAM) -> float:
        """
        Memory a process holds on one socket, summed over the NUMA nodes of that socket.

        Sub-NUMA clustering gives a socket several nodes. Without the mapping only the node whose
        id happens to match the socket number would be counted and the rest would read as zero.
        """
        package_id = int(dram.cpu.package_id)
        nodes = self._nodes_per_package.get(package_id, [package_id])
        return sum(numa_memory.get(node, 0.0) for node in nodes)

    def _get_all_processes_numa_info(self) -> DataRecord:
        """
        Gather NUMA memory statistics for all running processes.

        This function iterates over all running processes, collects their NUMA
        memory usage, and returns a DataRecord with detailed information for
        each process.

        Returns
        -------
        DataRecord
            A DataRecord object containing measurements of memory usage per
            NUMA node for each process, and the corresponding Process entities.
        """
        default_page_kb, huge_page_kb = self._get_system_page_sizes_kb()

        pids: list[int] = self._iter_pids()

        measurements: list[Measurement | pl.DataFrame] = []
        entities: list[Entity] = []
        measurement_timepoint = datetime.now(UTC)

        max_workers = 2

        def task(pid: int):
            return pid, self._get_numa_memory_for_pid(pid, default_page_kb, huge_page_kb)

        # Update the registry once before processing all PIDs
        self._logical_entity_registry.update()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(task, pid) for pid in pids]
            for fut in as_completed(futures):
                try:
                    pid, (numa_memory, total_memory) = fut.result()
                except Exception:
                    continue

                if not numa_memory or total_memory <= 0:
                    continue

                # Lookup the logical entity from the registry
                try:
                    pid_start_ns = get_pid_starttime_ns(pid)
                    logical_entity = self._logical_entity_registry.get_logical_entity(pid=pid, start_time=pid_start_ns)
                    if logical_entity is None:
                        # logger.debug(f"Could not find logical entity for pid {pid}")
                        continue

                    # append the memory this process holds on each socket
                    for e in self.entities:
                        if isinstance(e, DRAM):
                            m = Measurement(
                                value=self._memory_on_socket(numa_memory, e),
                                metric_id=DRAM_SOCKET_PID_MEMORY.name,
                                physical_entity_id=e.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=measurement_timepoint,
                                end_ts=measurement_timepoint,
                            )
                            measurements.append(m)

                except Exception as e:
                    # Process might have terminated or other errors
                    logger.debug(f"Error processing pid {pid}: {e}")
                    continue

        return DataRecord(entities=entities.copy(), measurements=measurements)

    @override
    def stop(self, metrics: list[Metric]) -> DataRecord | None:
        if time.perf_counter() - self._last_called < 10:
            return None
        return self.get_data(metrics)


if __name__ == "__main__":
    import time
    from dataclasses import dataclass

    # Simple mock classes for benchmarking
    @dataclass
    class MockCPU:
        package_id: int

    @dataclass
    class MockDRAM:
        cpu: MockCPU

        @property
        def id(self) -> str:
            return f"dram_{self.cpu.package_id}"

    # Discover NUMA nodes by scanning /sys/devices/system/node/
    numa_nodes = []
    node_dir = "/sys/devices/system/node"
    if os.path.isdir(node_dir):
        for entry in os.scandir(node_dir):
            if entry.is_dir() and entry.name.startswith("node"):
                try:
                    node_id = int(entry.name[4:])
                    numa_nodes.append(node_id)
                except ValueError:
                    pass
    numa_nodes.sort()

    if not numa_nodes:
        numa_nodes = [0]  # Fallback to single node

    # Create mock DRAM entities for each NUMA node
    entities = [MockDRAM(cpu=MockCPU(package_id=node_id)) for node_id in numa_nodes]  # type: ignore[arg-type]

    print(f"Discovered {len(entities)} NUMA nodes: {numa_nodes}")
    print("Starting benchmark...")

    proc_numa = ProcNuma(entities)  # type: ignore[arg-type]

    # Warm-up run
    proc_numa._get_all_processes_numa_info()

    # Timed run
    start = time.perf_counter()
    result = proc_numa._get_all_processes_numa_info()
    elapsed = time.perf_counter() - start

    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Measurements collected: {len(result.measurements)}")
    print(f"Entities collected: {len(result.entities)}")
