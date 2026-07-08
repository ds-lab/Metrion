import hashlib
from collections.abc import Callable
from typing import Optional

from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.gpu import GPU
from metrion.entity.network_interface import NetworkInterface


class Metric:
    """
    Represents a metric type with a name and unit.
    This class maintains a registry of all created metric types to enable easy lookup.
    """

    _registry: dict[str, "Metric"] = {}

    def __init__(
        self,
        name: str,
        unit: str,
        is_applicable: Callable[[type], bool],
        description: str = "",
        start_time: str = "",
        end_time: str = "",
        coverage: str = "",
    ):
        self.name = name
        self.unit = unit
        self.id = self._generate_id()
        self._is_applicable = is_applicable
        self._description = description

        # Create and set the docstring for this instance
        if description:
            doc = f"{description}\n\n"
            if start_time:
                doc += f"Start Time: {start_time}\n"
            if end_time:
                doc += f"End Time: {end_time}\n"
            if unit:
                doc += f"Unit: {unit}\n"
            if coverage:
                doc += f"Coverage: {coverage}\n"
            self.__doc__ = doc

        Metric._registry[name] = self  # Automatically register metric types

    def _generate_id(self) -> str:
        unique_string = f"{self.name}{self.unit}"
        hashed_id = hashlib.md5(unique_string.encode()).hexdigest()
        return hashed_id

    @classmethod
    def get(cls, name: str) -> Optional["Metric"]:
        return cls._registry.get(name, None)

    @classmethod
    def get_all_metrics(cls) -> dict[str, "Metric"]:
        """
        Returns all registered metrics.

        Returns
        -------
        dict
            Dictionary of all registered metrics with names as keys.
        """
        return cls._registry

    def __str__(self):
        return f"{self.name} ({self.unit})"

    def __repr__(self):
        return self.__str__()

    @property
    def description(self) -> str:
        return self._description

    def is_applicable(self, entity_type: type) -> bool:
        return self._is_applicable(entity_type)


# Define metrics with documentation
"""
Pattern: <COMPONENT>_<LOCATION>_<SCOPE>_<SIGNAL>

COMPONENT            Hardware/software block measured   CPU, GPU, RAM, NET, STORAGE, FPGA, ...
LOCATION             Physical aggregation level         HOST, SOCKET, CORE, NODE, DEVICE, PORT, ...
SCOPE                Sub-selection inside that location TOTAL, ACTIVE, IDLE, PID, PKG, CORES, PP0, PP1, DRAM, SM, ...
SIGNAL               What is being measured             TIME, ENERGY, POWER, MEMORY, UTILIZATION,
                                                        TEMPERATURE, CYCLES, BANDWIDTH, IOPS, ...

EXAMPLES:
- CPU_HOST_TOTAL_TIME: CPU component, across entire host, all activity, time measurement
- CPU_SOCKET_PKG_ENERGY: CPU component, single socket, package level, energy measurement
- CPU_CORE_PID_TIME: CPU component, single core, specific process, time measurement
- GPU_DEVICE_TOTAL_MEMORY: GPU component, single device, total capacity, memory measurement
- GPU_HOST_TOTAL_MEMORY: GPU component, all devices on host, total capacity, memory measurement
- RAM_SOCKET_DRAM_ENERGY: RAM component, single socket, DRAM controller, energy measurement
"""

NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED = Metric(
    name="NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED",
    unit="bytes",
    is_applicable=lambda e: e is NetworkInterface,
    description="Total bytes received by a specific process across all network interfaces on the host",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Total network bytes received by the process across all interfaces since process start",
)

NETWORK_HOST_PID_TOTAL_BYTES_SENT = Metric(
    name="NETWORK_HOST_PID_TOTAL_BYTES_SENT",
    unit="bytes",
    is_applicable=lambda e: e is NetworkInterface,
    description="Total bytes sent by a specific process across all network interfaces on the host",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Total network bytes sent by the process across all interfaces since process start",
)

CPU_SOCKET_PKG_ENERGY = Metric(
    name="CPU_SOCKET_PKG_ENERGY",
    unit="J",
    is_applicable=lambda e: e is CPU,
    description="Energy consumed by the entire CPU package of the socket",
    start_time="Last RAPL counter reset (boot time or counter overflow)",
    end_time="Current measurement time",
    coverage="Total energy consumed by the entire CPU package of the socket",
)

CPU_CORE_PID_TIME = Metric(
    name="CPU_CORE_PID_TIME",
    unit="ms",
    is_applicable=lambda e: e is CPU,
    description="CPU time consumed by a process on a specific core",
    start_time="Time when monitoring agent/eBPF tracepoint is attached",
    end_time="Current measurement time",
    coverage="Process execution time on a single core since monitoring began",
)

CPU_CORE_PID_WORK = Metric(
    name="CPU_CORE_PID_WORK",
    unit="work",
    is_applicable=lambda e: e is CPU,
    description="Work time of a process on a specific core",
    start_time="TODO",
    end_time="Scheduling end time of the process on the core",
    coverage="TODO",
)

CPU_PKG_PID_TOTAL_ENERGY = Metric(
    name="CPU_PKG_PID_TOTAL_ENERGY",
    unit="J",
    is_applicable=lambda e: e is CPU,
    description="Total energy consumed by the CPU package for a specific process between start and end times",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Total package energy attributed to the process on the socket including both idle and active contributions",  # noqa: E501
)

CPU_PKG_PID_IDLE_ENERGY = Metric(
    name="CPU_PKG_PID_IDLE_ENERGY",
    unit="J",
    is_applicable=lambda e: e is CPU,
    description="Energy consumed by the CPU package while in idle or low-activity states for a specific process on the package",  # noqa: E501
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Portion of package energy attributable to idle or background states during the process lifetime on the package",  # noqa: E501
)

CPU_PKG_PID_DYNAMIC_ENERGY = Metric(
    name="CPU_PKG_PID_DYNAMIC_ENERGY",
    unit="J",
    is_applicable=lambda e: e is CPU,
    description="Dynamic (active) energy consumed by the CPU package due to a specific process between start and end times",  # noqa: E501
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Package energy attributable to active CPU activity caused by the process (total minus idle)",
)

CPU_CORE_PID_APERF = Metric(
    name="CPU_CORE_PID_APERF",
    unit="cycles",
    is_applicable=lambda e: e is CPU,
    description="Actual performance frequency clock count for a specific process on a core between start and end times",
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Cycles counted at the actual operating frequency when CPU is not halted while executing the process",
)

CPU_CORE_PID_MPERF = Metric(
    name="CPU_CORE_PID_MPERF",
    unit="cycles",
    is_applicable=lambda e: e is CPU,
    description="Maximum performance frequency clock count for a specific process on a core between start and end times",  # noqa: E501
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Reference cycles counted at the nominal frequency, used to calculate average frequency regardless of CPU speed changes",  # noqa: E501
)

CPU_CORE_PID_UNHALTED_CYCLES = Metric(
    name="CPU_CORE_PID_UNHALTED_CYCLES",
    unit="cycles",
    is_applicable=lambda e: e is CPU,
    description="Cycles during which the CPU is actively executing instructions for a process on a core between start and end times",  # noqa: E501
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Cycles when the CPU core is not in a halted state, representing actual computational work",
)

CPU_CORE_PID_EXECUTION_TYPE = Metric(
    name="CPU_CORE_PID_EXECUTION_TYPE",
    unit="Indicator",
    is_applicable=lambda e: e is CPU,
    description="Cycles during which the CPU is actively executing instructions for a process on a core between "
    'start and end times can be either "alone" (0) or shared (1) with other processes on the core',
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Cycles when the CPU core is not in a halted state, representing actual computational work",
)

CPU_CORE_PID_L2_MISS = Metric(
    name="CPU_CORE_PID_L2_MISS",
    unit="requests",
    is_applicable=lambda e: e is CPU,
    description="Cache misses for a specific process on a core that require fetching data from a lower-level cache or main memory, contributing to increased energy consumption.",  # noqa: E501
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Counts L2 cache misses on P-Cores and Last-Level Cache (LLC) misses on E-Cores for hybrid CPUs. This unified metric represents requests that miss the core's private or mid-level cache.",  # noqa: E501
)

CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM = Metric(
    name="CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM",
    unit="requests",
    is_applicable=lambda e: e is CPU,
    description="Retired memory load instructions that missed the L3 cache and were serviced by local DRAM for a specific process on a core.",  # noqa: E501
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Counts retired memory loads for a process on a core that miss the L3 cache and are fulfilled by the DRAM controller on the same socket.",  # noqa: E501
)

CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM = Metric(
    name="CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM",
    unit="requests",
    is_applicable=lambda e: e is CPU,
    description="Retired memory load instructions that missed the L3 cache and were serviced by remote DRAM for a specific process on a core.",  # noqa: E501
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Counts retired memory loads for a process on a core that miss the L3 cache and are fulfilled by the DRAM controller on a different socket.",  # noqa: E501
)

CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES = Metric(
    name="CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES",
    unit="instructions",
    is_applicable=lambda e: e is CPU,
    description="All retired store instructions for a specific process on a core.",
    start_time="Scheduling start time of the process on the core",
    end_time="Scheduling end time of the process on the core",
    coverage="Counts all memory store instructions that have architecturally completed for a process on a core.",
)

CPU_SOCKET_PP0_ENERGY = Metric(
    name="CPU_SOCKET_PP0_ENERGY",
    unit="J",
    is_applicable=lambda e: e is CPU,
    description="Energy consumed by the CPU cores (Power Plane 0) of a specific socket",
    start_time="Last RAPL counter reset (boot time or counter overflow)",
    end_time="Current measurement time",
    coverage="Energy consumed by just the cores within the CPU socket",
)

DRAM_SOCKET_TOTAL_ENERGY = Metric(
    name="DRAM_SOCKET_TOTAL_ENERGY",
    unit="J",
    is_applicable=lambda e: e is DRAM,
    description="Energy consumed by DRAM memory of a specific socket",
    start_time="Last RAPL counter reset (boot time or counter overflow)",
    end_time="Current measurement time",
    coverage="Energy consumed by DRAM memory controller of the socket",
)

DRAM_SOCKET_PID_MEMORY = Metric(
    name="DRAM_SOCKET_PID_MEMORY",
    unit="MB",
    is_applicable=lambda e: e is DRAM,
    description="Amount of memory used by a specific process on a specific NUMA node (socket).",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Memory usage per process on a NUMA node, based on /proc/<pid>/numa_maps.",
)

DRAM_SOCKET_PID_TOTAL_ENERGY = Metric(
    name="DRAM_SOCKET_PID_TOTAL_ENERGY",
    unit="J",
    is_applicable=lambda e: e is DRAM,
    description="Total energy consumed by DRAM for a specific process on a socket between start and end times",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Total DRAM energy attributed to the process on the NUMA node (socket) including both idle and active contributions",  # noqa: E501
)

DRAM_SOCKET_PID_IDLE_ENERGY = Metric(
    name="DRAM_SOCKET_PID_IDLE_ENERGY",
    unit="J",
    is_applicable=lambda e: e is DRAM,
    description="Energy consumed by DRAM while the memory subsystem is in idle or low-activity states for a specific process on a socket",  # noqa: E501
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Portion of DRAM energy attributable to idle or background states during the process lifetime on the socket",  # noqa: E501
)

DRAM_SOCKET_PID_DYNAMIC_ENERGY = Metric(
    name="DRAM_SOCKET_PID_DYNAMIC_ENERGY",
    unit="J",
    is_applicable=lambda e: e is DRAM,
    description="Dynamic (active) energy consumed by DRAM due to a specific process on a socket between start and end times",  # noqa: E501
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="DRAM energy attributable to active memory accesses and activity caused by the process (total minus idle)",
)

# GPU metrics
GPU_DEVICE_TOTAL_MEMORY_USED = Metric(
    name="GPU_DEVICE_TOTAL_MEMORY_USED",
    unit="MB",
    is_applicable=lambda e: e is GPU,
    description="Amount of GPU memory currently in use",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Used memory across the entire GPU",
)

GPU_DEVICE_TOTAL_MEMORY_FREE = Metric(
    name="GPU_DEVICE_TOTAL_MEMORY_FREE",
    unit="MB",
    is_applicable=lambda e: e is GPU,
    description="Amount of GPU memory currently available",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Free memory across the entire GPU",
)

GPU_DEVICE_TOTAL_MEMORY = Metric(
    name="GPU_DEVICE_TOTAL_MEMORY",
    unit="MB",
    is_applicable=lambda e: e is GPU,
    description="Total amount of GPU memory",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Total memory capacity of the GPU",
)

GPU_DEVICE_TOTAL_ENERGY = Metric(
    name="GPU_DEVICE_TOTAL_ENERGY",
    unit="J",
    is_applicable=lambda e: e is GPU,
    description="Energy consumed by the GPU",
    start_time="Last NVML energy counter reset",
    end_time="Current measurement time",
    coverage="Total energy consumed by the GPU",
)

GPU_DEVICE_TOTAL_TEMPERATURE = Metric(
    name="GPU_DEVICE_TOTAL_TEMPERATURE",
    unit="°C",
    is_applicable=lambda e: e is GPU,
    description="Current GPU temperature",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Temperature of the GPU core",
)

GPU_DEVICE_TOTAL_MEMORY_UTILIZATION = Metric(
    name="GPU_DEVICE_TOTAL_MEMORY_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="GPU memory bandwidth utilization",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Percentage of memory bandwidth being used",
)

GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION = Metric(
    name="GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="GPU compute resources utilization",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Percentage of GPU computational resources being used",
)

GPU_DEVICE_PID_MEMORY = Metric(
    name="GPU_DEVICE_PID_MEMORY",
    unit="MB",
    is_applicable=lambda e: e is GPU,
    description="Amount of GPU memory used by a specific process",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="GPU memory allocated to the specific process",
)

GPU_DEVICE_PID_COMPUTE_UTILIZATION = Metric(
    name="GPU_DEVICE_PID_COMPUTE_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="GPU compute utilization by a specific process",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Percentage of GPU SM (Streaming Multiprocessor) units used by the process",
)

GPU_DEVICE_PID_ENCODER_UTILIZATION = Metric(
    name="GPU_DEVICE_PID_ENCODER_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="GPU encoder utilization by a specific process",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Percentage of GPU encoding resources used by the process",
)

GPU_DEVICE_PID_DECODER_UTILIZATION = Metric(
    name="GPU_DEVICE_PID_DECODER_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="GPU decoder utilization by a specific process",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Percentage of GPU decoding resources used by the process",
)

GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION = Metric(
    name="GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION",
    unit="%",
    is_applicable=lambda e: e is GPU,
    description="Frame buffer (memory) utilization by a specific process",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Percentage of GPU memory bandwidth used by the process",
)

GPU_DEVICE_PID_ACTIVE_TIME = Metric(
    name="GPU_DEVICE_PID_ACTIVE_TIME",
    unit="s",
    is_applicable=lambda e: e is GPU,
    description="Time a specific process has been actively using the GPU",
    start_time="Reference start time for the process",
    end_time="Reference end time for the process",
    coverage="Total time the process has been actively using the GPU since the last reset",
)

# GPU power metrics (generation-aware)
GPU_DEVICE_TOTAL_POWER_INSTANT = Metric(
    name="GPU_DEVICE_TOTAL_POWER_INSTANT",
    unit="W",
    is_applicable=lambda e: e is GPU,
    description="Instantaneous power draw for pre-Ampere GPUs and GA100",
    start_time="Current measurement time",
    end_time="Current measurement time",
    coverage="Instantaneous total power including GPU and memory",
)

GPU_DEVICE_TOTAL_POWER_AVERAGE = Metric(
    name="GPU_DEVICE_TOTAL_POWER_AVERAGE",
    unit="W",
    is_applicable=lambda e: e is GPU,
    description="1-second averaged power for Ampere+ GPUs (except GA100)",
    start_time="1s before measurement time",
    end_time="Current measurement time",
    coverage="Averaged total power including GPU and memory",
)

# GPU per-process energy attribution metrics
GPU_DEVICE_PID_TOTAL_ENERGY = Metric(
    name="GPU_DEVICE_PID_TOTAL_ENERGY",
    unit="J",
    is_applicable=lambda e: e is GPU,
    description="Total energy attributed to a process on the GPU",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Sum of idle and dynamic energy attributed to the process",
)

GPU_DEVICE_PID_IDLE_ENERGY = Metric(
    name="GPU_DEVICE_PID_IDLE_ENERGY",
    unit="J",
    is_applicable=lambda e: e is GPU,
    description="Idle energy share attributed to a process on the GPU",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Portion of GPU idle energy based on process time share",
)

GPU_DEVICE_PID_DYNAMIC_ENERGY = Metric(
    name="GPU_DEVICE_PID_DYNAMIC_ENERGY",
    unit="J",
    is_applicable=lambda e: e is GPU,
    description="Dynamic energy attributed to a process on the GPU",
    start_time="Process start time",
    end_time="Current measurement time",
    coverage="Active energy attributed via SM utilization ratio",
)
