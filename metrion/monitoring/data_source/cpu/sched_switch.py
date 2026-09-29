import ctypes as ct
import mmap
import multiprocessing as mp
import queue as q
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

import numpy as np
import polars as pl
import psutil
from loguru import logger

import metrion.metrion_rust as mar  # ty:ignore[unresolved-import]
from metrion.config_reader.config_schema import EbpfConfig, TimingConfig
from metrion.data_record import DataRecord
from metrion.discovery.system_scanner.bare_metal.linux.util import parse_cpu_list
from metrion.entity.cpu import CPU
from metrion.entity.entity import Entity
from metrion.entity.logical_entity_registry import LogicalEntityRegistry
from metrion.metric import (
    CPU_CORE_PID_APERF,
    CPU_CORE_PID_EXECUTION_TYPE,
    CPU_CORE_PID_L2_MISS,
    CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM,
    CPU_CORE_PID_MPERF,
    CPU_CORE_PID_TIME,
    CPU_CORE_PID_UNHALTED_CYCLES,
    CPU_CORE_PID_WORK,
    Metric,
)
from metrion.monitoring.data_source.cpu.collection_plan import (
    SchedSwitchCollectionPlan,
    resolve_sched_switch_plan,
)
from metrion.monitoring.data_source.cpu.ring_buff_base import EbpfRingBufferProcess, RingBufferConfig
from metrion.monitoring.data_source.cpu.utils import (
    CustomPerf,
    grouping_threshold_ms,
    read_perf_event,
    read_perf_type,
    resolve_with_strace,
)
from metrion.monitoring.data_source.data_source_base import DataSource
from metrion.monitoring.data_source.util import _validate_bpf_clock_domain
from metrion.monitoring.timing_context import TimingContext, TimingContextRegistry
from metrion.queue_util import drain_remaining, drain_until_sentinel
from metrion.util import boot_epoch_s, check_entity_type, read_temp_ipc, split_large_df

# Where an event sits in the energy interval it is grouped into. Events in the guard band at
# either end, and events that cross an interval boundary, are summed apart from the ones that
# sit wholly inside the usable window.
_BAND_HEAD = 0
_BAND_MIDDLE = 1
_BAND_TAIL = 2
_BAND_STRADDLING = 3

# Header bits of a ring buffer record, from include/uapi/linux/bpf.h.
_RINGBUF_BUSY_BIT = 1 << 31
_RINGBUF_DISCARD_BIT = 1 << 30


def _band_expr(interval_ns: int, threshold_ns: int) -> pl.Expr:
    """Band of each event, from group_id, end_group, pos_start and pos_end."""
    # Straddling first: pos_end belongs to the next interval for those, so the guard band
    # checks below would read it as an early position.
    return (
        pl.when(pl.col("group_id") != pl.col("end_group"))
        .then(_BAND_STRADDLING)
        .when(pl.col("pos_start") < threshold_ns)
        .then(_BAND_HEAD)
        .when(pl.col("pos_end") > interval_ns - threshold_ns)
        .then(_BAND_TAIL)
        .otherwise(_BAND_MIDDLE)
        .alias("band")
    )


class SchedSwitchBpfMonitor(EbpfRingBufferProcess):
    _overflow_hint = " Raise ebpf.ringbuf_mib in the config for a larger buffer."

    # NumPy structured dtype matching sched_event_v2 (328 bytes)
    # Must match the BPF struct layout exactly
    SCHED_EVENT_DTYPE = np.dtype(
        [
            # Monitoring data (136 bytes)
            ("pid_start", "<u8"),
            ("time_start", "<u8"),
            ("aperf_start", "<u8"),
            ("mperf_start", "<u8"),
            ("unhalted_core_cycles_start", "<u8"),
            ("loads_local_start", "<u8"),
            ("loads_remote_start", "<u8"),
            ("stores_start", "<u8"),
            ("time_end", "<u8"),
            ("aperf_end", "<u8"),
            ("mperf_end", "<u8"),
            ("unhalted_core_cycles_end", "<u8"),
            ("loads_local_end", "<u8"),
            ("loads_remote_end", "<u8"),
            ("stores_end", "<u8"),
            ("pid", "<u4"),
            ("tgid", "<u4"),
            ("cpu_core", "<u4"),
            ("event_type", "u1"),
            ("_pad", "3u1"),  # 3 bytes padding
            # Metadata (192 bytes)
            ("tgid_start", "<u8"),
            ("parent_tid_start", "<u8"),
            ("parent_tgid_start", "<u8"),
            ("cgroup_id", "<u8"),
            ("parent_tid", "<u4"),
            ("parent_tgid", "<u4"),
            ("uid", "<u4"),
            ("gid", "<u4"),
            ("comm", "S16"),  # 16-byte fixed string
            ("cmdline", "S128"),  # 128-byte fixed string
        ],
        align=True,
    )

    # One ring buffer record: the kernel's 8-byte header, then the event.
    RING_RECORD_DTYPE = np.dtype([("len", "<u4"), ("pg_off", "<u4"), ("event", SCHED_EVENT_DTYPE)])

    @classmethod
    def start_singleton(
        cls,
        config: RingBufferConfig,
        collection_plan: SchedSwitchCollectionPlan | None = None,
        ringbuf_mib: int | None = None,
        daemon: bool = True,
    ):
        with cls._singletons_lock:
            existing = cls._singletons.get(cls)
            if existing is not None and existing.is_alive():
                logger.debug("sched_switch bpf: reusing existing singleton process")
                return existing

            plan = collection_plan or SchedSwitchCollectionPlan.all_enabled()
            enabled_groups, disabled_groups = plan.perf_groups_summary()
            logger.info(
                "sched_switch bpf: starting process, collecting perf groups={}, skipping={}",
                enabled_groups,
                disabled_groups,
            )
            inst = cls(config=config, collection_plan=collection_plan, ringbuf_mib=ringbuf_mib, daemon=daemon)
            inst.start()
            cls._singletons[cls] = inst
            return inst

    def __init__(
        self,
        config: RingBufferConfig,
        collection_plan: SchedSwitchCollectionPlan | None = None,
        ringbuf_mib: int | None = None,
        name: str | None = None,
        daemon: bool = True,
    ) -> None:
        super().__init__(config, name=name, daemon=daemon)
        # Enable fast raw bytes mode
        self._use_raw_bytes = True
        self._collection_plan = collection_plan or SchedSwitchCollectionPlan.all_enabled()
        self._ringbuf_mib = ringbuf_mib or EbpfConfig().ringbuf_mib
        self._pending_events = 0

    @override
    def _parse_raw_events(self, raw_events: list[np.ndarray]) -> pl.DataFrame:
        """Turn the events collected by _consume_ring into a DataFrame."""
        if not raw_events:
            return pl.DataFrame(schema=self._df_schema)

        arr = np.concatenate(raw_events)

        # Convert to Polars DataFrame
        # Handle numeric fields directly
        df = pl.DataFrame(
            {
                "pid": arr["pid"],
                "tgid": arr["tgid"],
                "pid_start": arr["pid_start"],
                "cpu_core": arr["cpu_core"],
                "time_start": arr["time_start"],
                "time_end": arr["time_end"],
                "aperf_start": arr["aperf_start"],
                "aperf_end": arr["aperf_end"],
                "mperf_start": arr["mperf_start"],
                "mperf_end": arr["mperf_end"],
                "unhalted_core_cycles_start": arr["unhalted_core_cycles_start"],
                "unhalted_core_cycles_end": arr["unhalted_core_cycles_end"],
                "loads_local_start": arr["loads_local_start"],
                "loads_local_end": arr["loads_local_end"],
                "loads_remote_start": arr["loads_remote_start"],
                "loads_remote_end": arr["loads_remote_end"],
                "stores_start": arr["stores_start"],
                "stores_end": arr["stores_end"],
                "event_type": arr["event_type"],
                # Metadata fields (from raw bytes)
                "tgid_start": arr["tgid_start"],
                "parent_tid": arr["parent_tid"],
                "parent_tgid": arr["parent_tgid"],
                "parent_tid_start": arr["parent_tid_start"],
                "parent_tgid_start": arr["parent_tgid_start"],
                "cgroup_id": arr["cgroup_id"],
                "uid": arr["uid"],
                "gid": arr["gid"],
            }
        )

        # Only the first event of a thread carries its name and command line, so only those rows
        # are decoded; all others stay None.
        with_metadata = np.flatnonzero(arr["event_type"] == 1)
        comm = pl.repeat(None, len(arr), dtype=pl.String, eager=True).alias("comm")
        cmdline = pl.repeat(None, len(arr), dtype=pl.String, eager=True).alias("cmdline")
        if len(with_metadata):
            comm = comm.scatter(
                with_metadata,
                [b.rstrip(b"\x00").decode("utf-8", errors="replace") for b in arr["comm"][with_metadata]],
            )
            cmdline = cmdline.scatter(
                with_metadata,
                [b.rstrip(b"\x00").decode("utf-8", errors="replace").strip() for b in arr["cmdline"][with_metadata]],
            )
        df = df.with_columns(comm, cmdline)

        # Set metadata fields to None for data-only events (event_type == 0)
        df = df.with_columns(
            [
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("tgid_start")).alias("tgid_start"),
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("parent_tid")).alias("parent_tid"),
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("parent_tgid")).alias("parent_tgid"),
                pl.when(pl.col("event_type") == 0)
                .then(None)
                .otherwise(pl.col("parent_tid_start"))
                .alias("parent_tid_start"),
                pl.when(pl.col("event_type") == 0)
                .then(None)
                .otherwise(pl.col("parent_tgid_start"))
                .alias("parent_tgid_start"),
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("cgroup_id")).alias("cgroup_id"),
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("uid")).alias("uid"),
                pl.when(pl.col("event_type") == 0).then(None).otherwise(pl.col("gid")).alias("gid"),
            ]
        )

        return df

    @override
    def _create_bpf(self):
        num_cpus = psutil.cpu_count(logical=True)
        perf_arrays = ["cycles_core", "aperf_core", "mperf_core", "load_local_perf", "load_remote_perf", "store_perf"]
        max_entries = {name: num_cpus for name in perf_arrays}
        max_entries["events"] = self._ringbuf_mib * 1024 * 1024
        return mar.BpfProgram("sched_switch", max_entries)

    @override
    def _setup_perf_events(self) -> None:
        """
        Set up and configure performance monitoring events for CPU cores.

        Opens only the perf counter groups required by the enabled metrics in config.

        This method initializes performance counters for monitoring:
        - Unhalted core cycles (UCC) for both P-cores and E-cores
        - APERF (Actual Performance Frequency Clock Count)
        - MPERF (Maximum Performance Frequency Clock Count)

        The method creates the appropriate perf_event_attr structures for each counter type,
        opens the performance events for all CPU cores, and attaches the file descriptors
        to BPF maps for data collection.

        Notes
        -----
        This requires elevated privileges to access hardware performance counters.
        The CPU topology (P-cores vs E-cores) is currently hardcoded for a specific
        processor architecture with a hybrid core design.
        Information about the perf events are gathered from:
        - Intel® 64 and IA-32 Architectures Software Developer’s Manual Volume 3B: System Programming Guide, Part 2
        - https://perfwiki.github.io/main/print/253669.pdf
        - https://man7.org/linux/man-pages/man1/perf-stat.1.html
        """
        collecting, skipping = self._collection_plan.perf_groups_summary()
        logger.info(
            "sched_switch BPF: setting up perf events, collecting={}, skipping={}",
            collecting,
            skipping,
        )

        # Open perf events for unhalted core cycles, aperf, mperf
        # 4 means RAW_TYPE
        # Event: 0x00003C
        # - Event Select: 0x3C (CPU_CLK_UNHALTED.THREAD (fix counters Address: 30AH),
        #                       CPU_CLK_UNHALTED.THREAD_P (event and mask: 3CH OOH),
        #                       UnHalted Core Cycle(event and mask: 3CH OOH))
        # - UMASK: 0x00
        # - Other: default -> 0x00
        # https://man7.org/linux/man-pages/man1/perf-stat.1.html
        # INTEL HYBRID SUPPORT
        # /sys/bus/event_source/devices/cpu_atom/type
        # /sys/bus/event_source/devices/cpu_core/type
        # get which cores are atom and which are core
        # /sys/devices/cpu_core/cpus and /sys/devices/cpu_atom/cpus
        # L2_RQSTS.MISS
        # this is only available for P-Cores, not E-Cores
        # for E-Cores use all cache misses (cache-misses) not only L2
        # Event Number: 24H
        # Umask Value: 3FH
        # Type: raw: 0x4

        # check if the system has hybrid cores
        if Path("/sys/devices/cpu_atom/cpus").exists():
            with open("/sys/devices/cpu_atom/cpus") as f:
                # cat /sys/devices/cpu_core/cpus 0-15
                atom_cpus = parse_cpu_list(f.read().strip())
        else:
            atom_cpus = []

        if Path("/sys/devices/cpu_core/cpus").exists():
            with open("/sys/devices/cpu_core/cpus") as f:
                # cat /sys/devices/cpu_core/cpus 0-15
                core_cpus = parse_cpu_list(f.read().strip())
        else:
            core_cpus = []

        # Setup FDs lists
        fds_ucc: list[int] = []
        fds_load_local_dram: list[int] = []
        fds_load_remote_dram: list[int] = []
        fds_store: list[int] = []

        if self._collection_plan.needs_cpu_perf and len(atom_cpus) != 0 and len(core_cpus) != 0:
            # hybrid system detected
            # select only cores in topology
            # Unhalted Core Cycles for Atom cores, read from /sys/bus/event_source/devices/cpu_atom/events/cpu-cycles
            # and /sys/bus/event_source/devices/cpu_atom/type
            type = read_perf_type("/sys/bus/event_source/devices/cpu_atom/type")
            config = read_perf_event("/sys/bus/event_source/devices/cpu_atom/events/cpu-cycles")

            # Unhalted Core Cycles
            # sudo strace -e perf_event_open -v perf stat -e cpu_atom/r003c/ -C 15 sleep 1 2>&1 | grep perf_event_open
            attr_ucc_atom = CustomPerf.perf_event_attr()
            attr_ucc_atom.type = type
            attr_ucc_atom.config = config
            attr_ucc_atom.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_ucc_atom.sample_period = 0  # Count all events, don't sample
            attr_ucc_atom.freq = 0  # Use period-based counting, not frequency
            attr_ucc_atom.sample_type = 0
            attr_ucc_atom.disabled = 0  # 0 is enabled, 1 is disabled
            attr_ucc_atom.exclude_kernel = 0  # 0 is included, 1 is excluded
            attr_ucc_atom.exclude_user = 0  # 0 is included, 1 is excluded
            attr_ucc_atom.exclude_guest = 0  # 0 is included, 1 is excluded
            fds_ucc_atom = CustomPerf.perf_custom_event_open_with_results(
                attr=attr_ucc_atom, pid=-1, cpus=list(atom_cpus)
            )

            # select only cores in topology
            # Unhalted Core Cycles for Core cores, read from /sys/bus/event_source/devices/cpu_core/events/cpu-cycles
            # and /sys/bus/event_source/devices/cpu_core/type
            type = read_perf_type("/sys/bus/event_source/devices/cpu_core/type")
            config = read_perf_event("/sys/bus/event_source/devices/cpu_core/events/cpu-cycles")

            # sudo strace -e perf_event_open -v perf stat -e cpu_core/r003c/ -C 0 sleep 1 2>&1 | grep perf_event_open
            attr_ucc_core = CustomPerf.perf_event_attr()
            attr_ucc_core.type = type
            attr_ucc_core.config = config
            attr_ucc_core.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_ucc_core.sample_period = 0  # Count all events, don't sample
            attr_ucc_core.freq = 0  # Use period-based counting, not frequency
            attr_ucc_core.sample_type = 0
            attr_ucc_core.disabled = 0
            attr_ucc_core.exclude_kernel = 0
            attr_ucc_core.exclude_user = 0
            attr_ucc_core.exclude_guest = 0
            fds_ucc_core = CustomPerf.perf_custom_event_open_with_results(
                attr=attr_ucc_core, pid=-1, cpus=list(core_cpus)
            )
            fds_ucc = fds_ucc_core + fds_ucc_atom

            # L2_RQSTS.MISS for Core cores
            attr_l2_rqsts_miss_core = CustomPerf.perf_event_attr()
            attr_l2_rqsts_miss_core.type = 0x4
            attr_l2_rqsts_miss_core.config = 0x3F24
            attr_l2_rqsts_miss_core.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_l2_rqsts_miss_core.sample_period = 0
            attr_l2_rqsts_miss_core.sample_type = 0
            attr_l2_rqsts_miss_core.disabled = 0
            attr_l2_rqsts_miss_core.exclude_kernel = 0
            attr_l2_rqsts_miss_core.exclude_user = 0
            attr_l2_rqsts_miss_core.freq = 0
            attr_l2_rqsts_miss_core.exclude_guest = 0
            fds_l2_rqsts_miss = CustomPerf.perf_custom_event_open_with_results(
                attr=attr_l2_rqsts_miss_core, pid=-1, cpus=list(core_cpus), group_fds=fds_ucc_core
            )

            # LLC misses for Atom cores
            type_llc_misses = read_perf_type("/sys/bus/event_source/devices/cpu_atom/type")
            config_llc_misses = read_perf_event("/sys/bus/event_source/devices/cpu_atom/events/cache-misses")
            attr_llc_miss_atom = CustomPerf.perf_event_attr()
            attr_llc_miss_atom.type = type_llc_misses
            attr_llc_miss_atom.config = config_llc_misses
            attr_llc_miss_atom.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_llc_miss_atom.sample_period = 0
            attr_llc_miss_atom.sample_type = 0
            attr_llc_miss_atom.disabled = 0
            attr_llc_miss_atom.exclude_kernel = 0
            attr_llc_miss_atom.exclude_user = 0
            attr_llc_miss_atom.freq = 0
            attr_llc_miss_atom.exclude_guest = 0
            fds_llc_miss_atom = CustomPerf.perf_custom_event_open_with_results(
                attr=attr_llc_miss_atom, pid=-1, cpus=list(atom_cpus), group_fds=fds_ucc_atom
            )

            fds_cache_miss = fds_l2_rqsts_miss + fds_llc_miss_atom  # noqa: F841

        elif self._collection_plan.needs_cpu_perf:
            # no hybrid system, use only unhalted core cycles for all cores
            type = read_perf_type("/sys/bus/event_source/devices/cpu/type")
            config = read_perf_event("/sys/bus/event_source/devices/cpu/events/cpu-cycles")

            attr_ucc = CustomPerf.perf_event_attr()
            attr_ucc.type = type
            attr_ucc.config = config
            attr_ucc.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_ucc.sample_period = 0  # Count all events, don't sample
            attr_ucc.freq = 0  # Use period-based counting, not frequency
            attr_ucc.sample_type = 0
            attr_ucc.disabled = 0
            attr_ucc.exclude_kernel = 0
            attr_ucc.exclude_user = 0
            attr_ucc.exclude_guest = 0
            fds_ucc = CustomPerf.perf_custom_event_open_with_results(
                attr=attr_ucc,
                pid=-1,
            )

        needs_memory_perf = (
            self._collection_plan.needs_l3_local
            or self._collection_plan.needs_l3_remote
            or self._collection_plan.needs_stores
        )
        if needs_memory_perf and not (len(atom_cpus) != 0 and len(core_cpus) != 0):
            # Events for DRAM Energy Consumption (non-hybrid systems only)
            # https://perfmon-events.intel.com/skylake_server.html
            if self._collection_plan.needs_l3_local:
                event_load_local_dram = "mem_load_l3_miss_retired.local_dram"
                config_load_local_dram = resolve_with_strace(event_load_local_dram, cpu=0)
                attr_load_local_dram = CustomPerf.perf_event_attr()
                attr_load_local_dram.type = config_load_local_dram["type"]
                attr_load_local_dram.config = config_load_local_dram["config"]
                attr_load_local_dram.size = ct.sizeof(CustomPerf.perf_event_attr)
                attr_load_local_dram.sample_period = 0
                attr_load_local_dram.sample_type = 0
                attr_load_local_dram.disabled = 0
                attr_load_local_dram.exclude_kernel = 0
                attr_load_local_dram.exclude_user = 0
                attr_load_local_dram.freq = 0
                attr_load_local_dram.exclude_guest = 0
                attr_load_local_dram.config1 = config_load_local_dram.get("config1", 0)
                fds_load_local_dram = CustomPerf.perf_custom_event_open_with_results(attr=attr_load_local_dram, pid=-1)

            if self._collection_plan.needs_l3_remote:
                event_load_remote_dram = "mem_load_l3_miss_retired.remote_dram"
                config_load_remote_dram = resolve_with_strace(event_load_remote_dram, cpu=0)
                attr_load_remote_dram = CustomPerf.perf_event_attr()
                attr_load_remote_dram.type = config_load_remote_dram["type"]
                attr_load_remote_dram.config = config_load_remote_dram["config"]
                attr_load_remote_dram.size = ct.sizeof(CustomPerf.perf_event_attr)
                attr_load_remote_dram.sample_period = 0
                attr_load_remote_dram.sample_type = 0
                attr_load_remote_dram.disabled = 0
                attr_load_remote_dram.exclude_kernel = 0
                attr_load_remote_dram.exclude_user = 0
                attr_load_remote_dram.freq = 0
                attr_load_remote_dram.exclude_guest = 0
                attr_load_remote_dram.config1 = config_load_remote_dram.get("config1", 0)
                fds_load_remote_dram = CustomPerf.perf_custom_event_open_with_results(
                    attr=attr_load_remote_dram, pid=-1
                )

            if self._collection_plan.needs_stores:
                event_store = "mem_inst_retired.all_stores"
                config_store = resolve_with_strace(event_store, cpu=0)
                attr_store = CustomPerf.perf_event_attr()
                attr_store.type = config_store["type"]
                attr_store.config = config_store["config"]
                attr_store.size = ct.sizeof(CustomPerf.perf_event_attr)
                attr_store.sample_period = 0
                attr_store.sample_type = 0
                attr_store.disabled = 0
                attr_store.exclude_kernel = 0
                attr_store.exclude_user = 0
                attr_store.freq = 0
                attr_store.exclude_guest = 0
                attr_store.config1 = config_store.get("config1", 0)
                fds_store = CustomPerf.perf_custom_event_open_with_results(attr=attr_store, pid=-1)

        fds_aperf: list[int] = []
        fds_mperf: list[int] = []
        if self._collection_plan.needs_cpu_perf:
            # run the following command to obtain correct attribute values
            # sudo strace -e perf_event_open -v perf stat -e msr/mperf/ -C 0 sleep 1 2>&1 | grep perf_event_open
            # MPERF
            mperf_type = read_perf_type("/sys/bus/event_source/devices/msr/type")
            mperf_config = read_perf_event("/sys/bus/event_source/devices/msr/events/mperf")
            attr_mperf = CustomPerf.perf_event_attr()
            attr_mperf.type = mperf_type
            attr_mperf.config = mperf_config
            attr_mperf.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_mperf.sample_period = 0
            attr_mperf.sample_type = 0
            attr_mperf.disabled = 0
            attr_mperf.exclude_kernel = 0
            attr_mperf.exclude_user = 0
            attr_mperf.freq = 0
            attr_mperf.exclude_guest = 0
            fds_mperf = CustomPerf.perf_custom_event_open_with_results(attr=attr_mperf, pid=-1, group_fds=fds_ucc)

            # APERF
            aperf_type = read_perf_type("/sys/bus/event_source/devices/msr/type")
            aperf_config = read_perf_event("/sys/bus/event_source/devices/msr/events/aperf")
            attr_aperf = CustomPerf.perf_event_attr()
            attr_aperf.type = aperf_type
            attr_aperf.config = aperf_config
            attr_aperf.size = ct.sizeof(CustomPerf.perf_event_attr)
            attr_aperf.sample_period = 0
            attr_aperf.sample_type = 0
            attr_aperf.disabled = 0
            attr_aperf.exclude_kernel = 0
            attr_aperf.exclude_user = 0
            attr_aperf.freq = 0
            attr_aperf.exclude_guest = 0
            fds_aperf = CustomPerf.perf_custom_event_open_with_results(attr=attr_aperf, pid=-1, group_fds=fds_ucc)

        # Attach the perf events to the BPF program
        if self._collection_plan.needs_cpu_perf and self._bpf is not None:
            for i, fd in enumerate(fds_ucc):
                self._bpf.set_perf_fd("cycles_core", i, fd)
            for i, apfd in enumerate(fds_aperf):
                self._bpf.set_perf_fd("aperf_core", i, apfd)
            for i, mpfd in enumerate(fds_mperf):
                self._bpf.set_perf_fd("mperf_core", i, mpfd)

        if self._collection_plan.needs_l3_local and self._bpf is not None:
            for i, fd in enumerate(fds_load_local_dram):
                self._bpf.set_perf_fd("load_local_perf", i, fd)

        if self._collection_plan.needs_l3_remote and self._bpf is not None:
            for i, fd in enumerate(fds_load_remote_dram):
                self._bpf.set_perf_fd("load_remote_perf", i, fd)

        if self._collection_plan.needs_stores and self._bpf is not None:
            for i, fd in enumerate(fds_store):
                self._bpf.set_perf_fd("store_perf", i, fd)

        logger.info(
            "sched_switch BPF: attached perf fds (ucc={}, aperf={}, mperf={}, l3_local={}, l3_remote={}, stores={})",
            len(fds_ucc),
            len(fds_aperf),
            len(fds_mperf),
            len(fds_load_local_dram),
            len(fds_load_remote_dram),
            len(fds_store),
        )

    @override
    def _attach_ringbuf(self) -> None:
        assert self._bpf is not None
        fd = self._bpf.map_fd("events")
        page = mmap.PAGESIZE
        size = self._ringbuf_mib * 1024 * 1024
        self._rb_mask = size - 1

        # Page 1: the consumer position, the only part the reader writes.
        self._rb_cons_map = mmap.mmap(fd, page, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
        # Page 2: the producer position, followed by the data area, which the kernel maps twice
        # in a row so that a record running past the end can be read in one piece.
        self._rb_data_map = mmap.mmap(fd, page + 2 * size, mmap.MAP_SHARED, mmap.PROT_READ, offset=page)

        self._rb_cons = np.frombuffer(self._rb_cons_map, np.uint64, 1)
        self._rb_prod = np.frombuffer(self._rb_data_map, np.uint64, 1)
        self._rb_data = np.frombuffer(self._rb_data_map, np.uint8, offset=page)

    def _consume_ring(self) -> None:
        record_size = self.RING_RECORD_DTYPE.itemsize
        cons = int(self._rb_cons[0])
        prod = int(self._rb_prod[0])
        n = (prod - cons) // record_size
        if n == 0:
            return

        start = cons & self._rb_mask
        records = self._rb_data[start : start + n * record_size].view(self.RING_RECORD_DTYPE)

        # Records are handed out in order: stop at the first one the kernel is still filling.
        busy = (records["len"] & _RINGBUF_BUSY_BIT) != 0
        taken = int(np.argmax(busy)) if busy.any() else n
        if taken == 0:
            return

        ready = records[:taken]
        keep = (ready["len"] & _RINGBUF_DISCARD_BIT) == 0
        self._raw_buffer.append(ready["event"][keep])
        self._pending_events += int(keep.sum())
        self._rb_cons[0] = cons + taken * record_size

    @override
    def _consume(self) -> None:
        self._consume_ring()
        if self._pending_events >= self._buffer_flush_threshold or self._time_flush_due():
            self._pending_events = 0
            self._flush_requested = True

    @override
    def _buffered_events(self) -> int:
        return self._pending_events

    @override
    def _event_to_record(self, event: Any) -> dict[str, Any]:
        record = {
            "pid": int(event.pid),
            "tgid": int(event.tgid),
            "pid_start": int(event.pid_start),  # pid creation time in ns since boot
            "cpu_core": int(event.cpu_core),
            # time in ns since boot
            "time_start": int(event.time_start),
            "time_end": int(event.time_end),
            # perf event data start
            "aperf_start": int(event.aperf_start),
            "aperf_end": int(event.aperf_end),
            "mperf_start": int(event.mperf_start),
            "mperf_end": int(event.mperf_end),
            "unhalted_core_cycles_start": int(event.unhalted_core_cycles_start),
            "unhalted_core_cycles_end": int(event.unhalted_core_cycles_end),
            "loads_local_start": int(event.loads_local_start),
            "loads_local_end": int(event.loads_local_end),
            "loads_remote_start": int(event.loads_remote_start),
            "loads_remote_end": int(event.loads_remote_end),
            "stores_start": int(event.stores_start),
            "stores_end": int(event.stores_end),
            # Event type: 0 = data only, 1 = with inline metadata
            "event_type": int(event.event_type),
        }

        # Include inline metadata when event_type indicates it's present
        if event.event_type == LogicalEntityRegistry.EVENT_TYPE_WITH_METADATA:
            record.update(
                {
                    "tgid_start": int(event.tgid_start),
                    "parent_tid": int(event.parent_tid),
                    "parent_tgid": int(event.parent_tgid),
                    "parent_tid_start": int(event.parent_tid_start),
                    "parent_tgid_start": int(event.parent_tgid_start),
                    "cgroup_id": int(event.cgroup_id),
                    "uid": int(event.uid),
                    "gid": int(event.gid),
                    "comm": event.comm.decode("utf-8", errors="replace").rstrip("\x00"),
                    "cmdline": event.cmdline.decode("utf-8", errors="replace").rstrip("\x00").strip(),
                }
            )
        else:
            # Set metadata fields to None for data-only events
            record.update(
                {
                    "tgid_start": None,
                    "parent_tid": None,
                    "parent_tgid": None,
                    "parent_tid_start": None,
                    "parent_tgid_start": None,
                    "cgroup_id": None,
                    "uid": None,
                    "gid": None,
                    "comm": None,
                    "cmdline": None,
                }
            )

        return record


class SchedSwitchMonitor(DataSource):
    """
    Data source for CPU scheduler context switch monitoring.

    Uses a SchedSwitchEBPFMonitor to observe scheduler events in the kernel.
    For each context switch, records:
    - The logical CPU core on which the process ran
    - The execution time (how long the process was scheduled)
    - Delta values for APERF and MPERF counters
    - Delta unhalted core cycles, adjusted for hyperthreading

    Attributes
    ----------
    SENTINEL : float
        Constant value (2**64 - 1) used to detect counter overflows.
    HT_FACTOR : float
        Adjustment factor (1.15) for hyperthreaded workloads.
    """

    _entity_metric_mapping = {
        CPU: {
            CPU_CORE_PID_APERF,
            CPU_CORE_PID_EXECUTION_TYPE,
            CPU_CORE_PID_L2_MISS,
            CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES,
            CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM,
            CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM,
            CPU_CORE_PID_MPERF,
            CPU_CORE_PID_TIME,
            CPU_CORE_PID_UNHALTED_CYCLES,
            CPU_CORE_PID_WORK,
        }
    }

    _instance = None
    SENTINEL = float(2**64 - 1)
    HT_FACTOR = 1.15

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # (optionally) initialize here or in __init__
        return cls._instance

    def __init__(
        self,
        entities: Sequence[CPU],
        interval: float | None = None,
        enabled_metrics: Sequence[Metric] | None = None,
        timing: TimingConfig | None = None,
        ebpf: EbpfConfig | None = None,
    ):
        """Initialize the SchedSwitchMonitor.

        Parameters
        ----------
        entities : Sequence[CPU]
            A list of CPU entities to monitor. It is expected that this list
            contains a single CPU entity representing all cores on the system.
        interval : float, optional
             The collection interval in seconds.
        enabled_metrics : Sequence[Metric], optional
            Metrics enabled in config for the CPU component.
        """
        if not hasattr(self, "_initialized"):
            super().__init__(entities, interval=interval, enabled_metrics=enabled_metrics, timing=timing, ebpf=ebpf)
            self._initialized = True

            check_entity_type(entities)
            self._entities = entities
            self._collection_plan = resolve_sched_switch_plan(enabled_metrics)
            self._enabled_metrics = set(enabled_metrics) if enabled_metrics else set(self._entity_metric_mapping[CPU])
            logger.info(
                "sched_switch enabled metrics from config: {}",
                sorted(metric.name for metric in self._enabled_metrics),
            )
            collecting, skipping = self._collection_plan.perf_groups_summary()
            logger.info(
                "sched_switch resolved perf counter plan: collecting={}, skipping={}",
                collecting,
                skipping,
            )

            mp.set_start_method("spawn", force=True)
            self._queue_monitoring_data = mp.Queue()
            self._stop_event = mp.Event()

            # cache values for one monitoring round
            self._cache_cpu_core_pid_aperf: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_mperf: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_unhalted_core_cycles: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_time: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_execution_type: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_work: list[pl.DataFrame] = []
            # self._cache_cpu_core_pid_l2_miss: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_load_local: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_load_remote: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_all_stores: list[pl.DataFrame] = []
            self._cache_cpu_core_pid_demand_rfo_remote: list[pl.DataFrame] = []

            # Track the background monitoring process
            self._ring_buffer_config = RingBufferConfig(
                buffer_flush_threshold=10_000,
                buffer_flush_time_threshold=self._timing.ebpf_flush,
                writer_flush_s=self._timing.storage_flush,
                output_queue=self._queue_monitoring_data,
                stop_event=self._stop_event,
                sentinel="SENTINEL",
                df_schema={
                    # Core monitoring data
                    "pid": pl.UInt32,
                    "tgid": pl.UInt32,
                    "pid_start": pl.UInt64,
                    "cpu_core": pl.UInt32,
                    "time_start": pl.UInt64,
                    "time_end": pl.UInt64,
                    "aperf_start": pl.UInt64,
                    "aperf_end": pl.UInt64,
                    "mperf_start": pl.UInt64,
                    "mperf_end": pl.UInt64,
                    "unhalted_core_cycles_start": pl.UInt64,
                    "unhalted_core_cycles_end": pl.UInt64,
                    "loads_local_start": pl.UInt64,
                    "loads_local_end": pl.UInt64,
                    "loads_remote_start": pl.UInt64,
                    "loads_remote_end": pl.UInt64,
                    "stores_start": pl.UInt64,
                    "stores_end": pl.UInt64,
                    # Event type flag for inline metadata
                    "event_type": pl.UInt8,
                    # Inline metadata fields (nullable for data-only events)
                    "tgid_start": pl.UInt64,
                    "parent_tid": pl.UInt32,
                    "parent_tgid": pl.UInt32,
                    "parent_tid_start": pl.UInt64,
                    "parent_tgid_start": pl.UInt64,
                    "cgroup_id": pl.UInt64,
                    "uid": pl.UInt32,
                    "gid": pl.UInt32,
                    "comm": pl.String,
                    "cmdline": pl.String,
                },
            )

            self._rb_sched_switch = SchedSwitchBpfMonitor.start_singleton(
                self._ring_buffer_config,
                collection_plan=self._collection_plan,
                ringbuf_mib=self._ebpf.ringbuf_mib,
            )

            # timing data
            self._boot_time_s = boot_epoch_s()
            self._boot_time_dt = datetime.fromtimestamp(self._boot_time_s, tz=UTC)

            # core map
            self._core_map_df, self._topology_map = self._core_entities_to_df()

            # logical entity registry
            self._logical_entity_registry = LogicalEntityRegistry.get_instance()

            # debug counters
            self._total_work_data = 0
            self._raw_events_received = 0
            self._clock_validated = False
            self._producer_finished = False
            self._producer_death_reported = False

    def _core_entities_to_df(self) -> tuple[pl.DataFrame, list[Any]]:
        ids_cpu_cores = []
        ids_cpu_siblings = []
        id_core_physical_entity = []

        for entity in self._entities:
            core_by_id = {c.core_id: c for c in entity.cores}

            for core_id, sibling_core in entity.topology.items():
                ids_cpu_cores.append(core_id)
                ids_cpu_siblings.append(sibling_core)

                core_obj = core_by_id.get(core_id)
                if core_obj is None:
                    # decide: skip, append null, or raise
                    id_core_physical_entity.append(None)
                else:
                    id_core_physical_entity.append(core_obj.id)

        core_map_df = pl.DataFrame(
            {
                "cpu_core": ids_cpu_cores,
                "physical_entity_id": id_core_physical_entity,
            }
        )

        topology_map = list(set(tuple(sorted(pair)) for pair in zip(ids_cpu_cores, ids_cpu_siblings, strict=True)))

        return core_map_df, topology_map

    def _add_logical_entity_id_to_df(self, data_df: pl.DataFrame) -> pl.DataFrame:
        """
        Add logical entity IDs to the dataframe using a three-tier matching strategy:

        2. REGISTRY LOOKUP: Check the LogicalEntityRegistry for known entities.
        """

        # Step 1: Update the logical entity registry (may have new events from task_newtask)
        self._logical_entity_registry.update()

        # Step 2: Match all (pid, pid_start) combinations to entity IDs
        pids_creation_time_tuple = data_df.select([pl.col("pid"), pl.col("pid_start")]).unique().rows()

        entity_id_list = []
        pid_list = []
        pid_creation_time_list = []
        before_unique_count = len(pids_creation_time_tuple)

        registry_matched = 0
        unmatched = []

        for pid, pid_start in pids_creation_time_tuple:
            entity = self._logical_entity_registry.get_logical_entity(pid=pid, start_time=pid_start)

            if entity is not None:
                entity_id_list.append(entity.id)
                pid_list.append(pid)
                pid_creation_time_list.append(pid_start)
                registry_matched += 1

            else:
                unmatched.append((pid, pid_start))

        if len(unmatched) > 0:
            # Log first 10 for debugging
            logger.warning(f"Failed to match {len(unmatched)} entities. Examples: {unmatched[:10]}")

        entity_df = pl.DataFrame(
            {
                "pid": pid_list,
                "pid_start": pid_creation_time_list,
                "logical_entity_id": entity_id_list,
            }
        )

        # Use left join to preserve unmatched records
        res = data_df.join(entity_df, on=["pid", "pid_start"], how="left")

        # Add a column to flag unmatched records
        res = res.with_columns(pl.col("logical_entity_id").is_null().alias("unmatched"))

        # Log match statistics
        unmatched_count = res.filter(pl.col("unmatched")).height
        total_events = len(data_df)
        match_rate = 100.0 * (1.0 - unmatched_count / total_events) if total_events > 0 else 100.0

        if unmatched_count > 0:
            logger.warning(f"Unmatched records: {unmatched_count} / {total_events} ({100.0 - match_rate:.2f}%)")

        after_unique_count = len(res.select([pl.col("pid"), pl.col("pid_start")]).unique().rows())
        if before_unique_count > 0 and (before_unique_count - after_unique_count) / before_unique_count > 0.01:
            logger.warning(
                f"Number of unique pids before: {before_unique_count}, "
                f"after: {after_unique_count}, diff: {before_unique_count - after_unique_count}"
            )
        return res

    def _get_raw_monitoring_data(self) -> pl.DataFrame:
        df_paths = []
        timeout = 0.01  # 10ms timeout
        max_iterations = 10000  # Higher limit for sched_switch (more events)

        for _ in range(max_iterations):
            try:
                item = self._queue_monitoring_data.get(timeout=timeout)
            except q.Empty:
                break
            if item == self._ring_buffer_config.sentinel:
                self._producer_finished = True
                break
            df_paths.append(item)

        self._report_dead_producer()

        if len(df_paths) == 0:
            return pl.DataFrame()

        dfs = []
        for df_path in df_paths:
            df = read_temp_ipc(df_path)
            dfs.append(df)
        df = pl.concat(dfs, how="vertical")
        return df

    def _report_dead_producer(self) -> None:
        # A producer killed mid-run, by the OOM killer say, sends no sentinel and no final count;
        # this is the only place the run learns that its CPU data stops.
        if self._producer_finished or self._producer_death_reported or self._rb_sched_switch.is_alive():
            return
        self._producer_death_reported = True
        logger.error(
            f"sched_switch: the eBPF producer died (exit code {self._rb_sched_switch.exitcode}; -9 is "
            "usually the OOM killer). No CPU measurements are recorded from here on."
        )

    def _add_physical_entity_id_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        # core id is the physical entity where the work was executed
        # assumes df hast cpu_core in dataframe
        # adds the phyiscal_entity_id to the dataframe
        return df.join(self._core_map_df, on="cpu_core")

    def _group_data(
        self,
        data_df: pl.DataFrame,
        metric_columns: list[str],
        tc: TimingContext | None,
        extra_group_by_columns: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Sum events onto the energy interval grid of ``tc``.

        Groups by (pid, pid_start, cpu_core, interval, band) plus any extra columns, where the
        band keeps events near an interval boundary apart from those inside the usable window.

        Parameters
        ----------
        data_df : pl.DataFrame
            Events with time_start, time_end, pid, pid_start and cpu_core columns.
        metric_columns : list[str]
            Columns to sum per group.
        tc : TimingContext | None
            Grid to group onto. None returns data_df unchanged.
        extra_group_by_columns : list[str] | None
            Further columns to group by, e.g. ["shared_state"].
        """
        if tc is None or data_df.is_empty():
            return data_df

        interval_ms = int(tc.interval_seconds * 1000)
        interval_ns = interval_ms * 1_000_000
        start_ns = tc.start_time_ms * 1_000_000
        threshold_ns = grouping_threshold_ms(interval_ms) * 1_000_000
        boot_epoch_ns = int(self._boot_time_s * 1e9)

        group_by_columns = ["pid", "pid_start", "cpu_core", "group_id", "band"]
        if extra_group_by_columns is not None:
            group_by_columns = group_by_columns + extra_group_by_columns

        grouped_df = (
            data_df.lazy()
            # Cast to Int64 before subtracting: events before the grid start would otherwise
            # wrap to huge positive u64 values.
            .with_columns(
                (pl.lit(boot_epoch_ns) + pl.col("time_start").cast(pl.Int64)).alias("time_start_ns"),
                (pl.lit(boot_epoch_ns) + pl.col("time_end").cast(pl.Int64)).alias("time_end_ns"),
            )
            .with_columns(
                ((pl.col("time_start_ns") - pl.lit(start_ns)) // pl.lit(interval_ns)).alias("group_id"),
                ((pl.col("time_end_ns") - pl.lit(start_ns)) // pl.lit(interval_ns)).alias("end_group"),
                ((pl.col("time_start_ns") - pl.lit(start_ns)) % pl.lit(interval_ns)).alias("pos_start"),
                ((pl.col("time_end_ns") - pl.lit(start_ns)) % pl.lit(interval_ns)).alias("pos_end"),
            )
            .with_columns(_band_expr(interval_ns, threshold_ns))
            .group_by(group_by_columns)
            .agg(
                pl.col("time_start_ns").min(),
                pl.col("time_end_ns").max(),
                pl.col(metric_columns).sum(),
            )
            .collect()
        )

        return self._convert_ns_to_datetime(grouped_df)

    def _convert_ns_to_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert time_start_ns and time_end_ns columns to naive UTC datetimes."""
        return df.with_columns(
            pl.from_epoch(pl.col("time_start_ns"), time_unit="ns").dt.cast_time_unit("us").alias("time_start"),
            pl.from_epoch(pl.col("time_end_ns"), time_unit="ns").dt.cast_time_unit("us").alias("time_end"),
        )

    def _calculate_memory_metrics(self, raw_data_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        rapl_dram_ctx = TimingContextRegistry.instance().get("intel_rapl_dram")
        if raw_data_df.is_empty() or rapl_dram_ctx is None:
            return (pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

        # Memory measurements
        memory_data_df = raw_data_df.select(
            [
                "pid",
                "pid_start",
                "cpu_core",
                "time_start",
                "time_end",
                # "l2_cache_misses_start",
                "loads_local_start",
                "loads_remote_start",
                "stores_start",
                # "l2_cache_misses_end",
                "loads_local_end",
                "loads_remote_end",
                "stores_end",
            ]
        )

        if memory_data_df.is_empty():
            return (pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

        # prepare memory data
        memory_data_df = memory_data_df.filter(pl.col("time_end") > 0).with_columns(
            # (pl.col("l2_cache_misses_end") - pl.col("l2_cache_misses_start")).alias("delta_l2_cache_misses"),
            (pl.col("loads_local_end") - pl.col("loads_local_start")).alias("delta_loads_local"),
            (pl.col("loads_remote_end") - pl.col("loads_remote_start")).alias("delta_loads_remote"),
            (pl.col("stores_end") - pl.col("stores_start")).alias("delta_stores"),
        )

        memory_data_df = memory_data_df.select(
            "pid",
            "pid_start",
            "cpu_core",
            "time_start",
            "time_end",
            # "delta_l2_cache_misses",
            "delta_loads_local",
            "delta_loads_remote",
            "delta_stores",
        )

        memory_data_df = self._group_data(
            memory_data_df,
            ["delta_loads_local", "delta_loads_remote", "delta_stores"],
            rapl_dram_ctx,
        )

        memory_data_df = self._add_logical_entity_id_to_df(memory_data_df)
        memory_data_df = self._add_physical_entity_id_to_df(memory_data_df)

        base_cols = [
            pl.col("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("time_start").alias("start_ts"),
            pl.col("time_end").alias("end_ts"),
        ]

        # CPU_CORE_PID_L2_MISS
        # l2_miss_measurements_df = memory_data_df.select(
        #     pl.lit(CPU_CORE_PID_L2_MISS.name).alias("metric_id"),
        #     *base_cols,
        #     pl.col("delta_l2_cache_misses").alias("value").cast(pl.Float64),
        # ).filter(pl.col("value") > 0)

        # CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM
        mem_load_l3_miss_local_dram_df = memory_data_df.select(
            pl.lit(CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_loads_local").alias("value").cast(pl.Float64),
        ).filter(pl.col("value") > 0)

        # CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM
        mem_load_l3_miss_remote_dram_df = memory_data_df.select(
            pl.lit(CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_loads_remote").alias("value").cast(pl.Float64),
        ).filter(pl.col("value") > 0)

        # CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES
        mem_inst_retired_all_stores_df = memory_data_df.select(
            pl.lit(CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_stores").alias("value").cast(pl.Float64),
        ).filter(pl.col("value") > 0)

        return (
            # l2_miss_measurements_df,
            mem_load_l3_miss_local_dram_df,
            mem_load_l3_miss_remote_dram_df,
            mem_inst_retired_all_stores_df,
        )

    def _calculate_cpu_metrics(
        self, raw_data_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        rapl_pkg_ctx = TimingContextRegistry.instance().get("intel_rapl_pkg")

        if raw_data_df.is_empty():
            empty = pl.DataFrame()
            return (empty, empty, empty, empty, empty, empty)

        # calculate metrics in rust for performance
        raw_data_df = raw_data_df.with_row_index(name="exec_count")
        raw_data_df = raw_data_df.with_columns(pl.col("exec_count").cast(pl.UInt64))
        raw_data_df = raw_data_df.select(
            pl.col("pid"),
            pl.col("pid_start"),
            pl.col("cpu_core"),
            pl.col("time_start"),
            pl.col("time_end"),
            pl.col("aperf_start"),
            pl.col("aperf_end"),
            pl.col("mperf_start"),
            pl.col("mperf_end"),
            pl.col("unhalted_core_cycles_start"),
            pl.col("unhalted_core_cycles_end"),
            pl.col("exec_count"),
        )
        cpu_data_df = mar.prepare_monitoring_data_for_tuples(raw_data_df, self._topology_map, self.HT_FACTOR)
        # output columns: 'pid', 'pid_start', 'start_time', 'end_time', 'cpu_core', 'delta_aperf',
        #                 'delta_mperf', 'delta_unhalted_core_cycles', 'execution_time', 'shared_state'

        if cpu_data_df.is_empty():
            empty = pl.DataFrame()
            return (empty, empty, empty, empty, empty, empty)

        cpu_data_df = cpu_data_df.with_columns(
            pl.col("start_time").alias("time_start"),
            pl.col("end_time").alias("time_end"),
        )

        cpu_data_df = self._group_data(
            cpu_data_df,
            ["delta_aperf", "delta_mperf", "delta_unhalted_core_cycles", "execution_time"],
            rapl_pkg_ctx,
            ["shared_state"],
        )

        cpu_data_df = self._add_logical_entity_id_to_df(cpu_data_df)
        cpu_data_df = self._add_physical_entity_id_to_df(cpu_data_df)

        # prepare columns
        cpu_data_df = cpu_data_df.with_columns(
            # Convert execution time to milliseconds
            (pl.col("execution_time") / 1e6).alias("execution_time_ms"),
            # Work value once (avoid creating a second DF just for work)
            (
                pl.when(pl.col("delta_mperf") == 0).then(1.0).otherwise(pl.col("delta_aperf") / pl.col("delta_mperf"))
                * pl.col("delta_unhalted_core_cycles")
            ).alias("work_value"),
        )

        # Common columns to reuse in each metric DF
        base_cols = [
            pl.col("physical_entity_id"),
            pl.col("logical_entity_id"),
            pl.col("time_start").alias("start_ts"),
            pl.col("time_end").alias("end_ts"),
        ]

        time_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_TIME.name).alias("metric_id"),
            *base_cols,
            pl.col("execution_time_ms").cast(pl.Float64).alias("value"),
        )

        aperf_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_APERF.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_aperf").cast(pl.Float64).alias("value"),
        )

        mperf_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_MPERF.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_mperf").cast(pl.Float64).alias("value"),
        )

        unhalted_cycles_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_UNHALTED_CYCLES.name).alias("metric_id"),
            *base_cols,
            pl.col("delta_unhalted_core_cycles").cast(pl.Float64).alias("value"),
        )

        execution_type_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_EXECUTION_TYPE.name).alias("metric_id"),
            *base_cols,
            pl.col("shared_state").cast(pl.Float64).alias("value"),
        )

        pid_work_measurements_df = cpu_data_df.select(
            pl.lit(CPU_CORE_PID_WORK.name).alias("metric_id"),
            *base_cols,
            pl.col("work_value").cast(pl.Float64).alias("value"),
        )

        return (
            time_measurements_df,
            aperf_measurements_df,
            mperf_measurements_df,
            unhalted_cycles_measurements_df,
            execution_type_measurements_df,
            pid_work_measurements_df,
        )

    def _validate_bpf_clock_domain(self, df: pl.DataFrame) -> None:
        """One-time check: compare converted BPF timestamp against wall clock."""
        if self._clock_validated or df.is_empty():
            return

        res = _validate_bpf_clock_domain(df, col_name="time_end")

        self._clock_validated = True if res else False

    def _calculate_and_cache_metrics(self, raw_data_df: pl.DataFrame) -> None:
        if raw_data_df.is_empty():
            return

        self._validate_bpf_clock_domain(raw_data_df)
        self._raw_events_received += raw_data_df.height

        # Only the first event of a thread carries its metadata, and both registry paths below
        # ignore the others.
        raw_data_inline_metadata_df = (
            raw_data_df.filter(pl.col("event_type") == 1)
            .select(
                pl.col("pid"),
                pl.col("pid_start"),
                pl.col("tgid"),
                pl.col("tgid_start"),
                pl.col("parent_tid"),
                pl.col("parent_tgid"),
                pl.col("parent_tid_start"),
                pl.col("parent_tgid_start"),
                pl.col("cgroup_id"),
                pl.col("uid"),
                pl.col("gid"),
                pl.col("comm"),
                pl.col("cmdline"),
                pl.col("event_type"),
            )
            .unique()
        )

        # Register entities from inline metadata (zero-loss path)
        self._logical_entity_registry._register_entities_from_inline_metadata(raw_data_inline_metadata_df)

        # Update existing entities with better metadata (enrichment path)
        self._logical_entity_registry.update_entity_metadata_from_df(raw_data_inline_metadata_df)

        raw_data_measurement_data_df = raw_data_df.select(
            pl.col("pid"),
            pl.col("pid_start"),
            pl.col("cpu_core"),
            pl.col("time_start"),
            pl.col("time_end"),
            pl.col("aperf_start"),
            pl.col("aperf_end"),
            pl.col("mperf_start"),
            pl.col("mperf_end"),
            pl.col("unhalted_core_cycles_start"),
            pl.col("unhalted_core_cycles_end"),
            pl.col("loads_local_start"),
            pl.col("loads_local_end"),
            pl.col("loads_remote_start"),
            pl.col("loads_remote_end"),
            pl.col("stores_start"),
            pl.col("stores_end"),
        )

        pid_work_measurements_df = pl.DataFrame()
        if self._collection_plan.needs_cpu_perf:
            (
                time_measurements_df,
                aperf_measurements_df,
                mperf_measurements_df,
                unhalted_cycles_measurements_df,
                execution_type_measurements_df,
                pid_work_measurements_df,
            ) = self._calculate_cpu_metrics(raw_data_measurement_data_df)

            if CPU_CORE_PID_APERF in self._enabled_metrics:
                self._cache_cpu_core_pid_aperf.extend(split_large_df(aperf_measurements_df))
            if CPU_CORE_PID_MPERF in self._enabled_metrics:
                self._cache_cpu_core_pid_mperf.extend(split_large_df(mperf_measurements_df))
            if CPU_CORE_PID_UNHALTED_CYCLES in self._enabled_metrics:
                self._cache_cpu_core_pid_unhalted_core_cycles.extend(split_large_df(unhalted_cycles_measurements_df))
            if CPU_CORE_PID_TIME in self._enabled_metrics:
                self._cache_cpu_core_pid_time.extend(split_large_df(time_measurements_df))
            if CPU_CORE_PID_EXECUTION_TYPE in self._enabled_metrics:
                self._cache_cpu_core_pid_execution_type.extend(split_large_df(execution_type_measurements_df))
            if CPU_CORE_PID_WORK in self._enabled_metrics:
                self._cache_cpu_core_pid_work.extend(split_large_df(pid_work_measurements_df))

        needs_memory_metrics = (
            self._collection_plan.needs_l3_local
            or self._collection_plan.needs_l3_remote
            or self._collection_plan.needs_stores
        )
        if needs_memory_metrics:
            (
                mem_load_l3_miss_local_dram_df,
                mem_load_l3_miss_remote_dram_df,
                mem_inst_retired_all_stores_df,
            ) = self._calculate_memory_metrics(raw_data_measurement_data_df)

            if self._collection_plan.needs_l3_local and mem_load_l3_miss_local_dram_df.height > 0:
                self._cache_cpu_core_pid_load_local.extend(split_large_df(mem_load_l3_miss_local_dram_df))
            if self._collection_plan.needs_l3_remote and mem_load_l3_miss_remote_dram_df.height > 0:
                self._cache_cpu_core_pid_load_remote.extend(split_large_df(mem_load_l3_miss_remote_dram_df))
            if self._collection_plan.needs_stores and mem_inst_retired_all_stores_df.height > 0:
                self._cache_cpu_core_pid_all_stores.extend(split_large_df(mem_inst_retired_all_stores_df))

        self._total_work_data += pid_work_measurements_df.height

    @override
    def get_data(self, metrics: Sequence[Metric]) -> DataRecord:
        """
        Retrieves performance data for the specified metrics.
        This method returns CPU performance data for the requested metrics. If the internal cache
        is empty, it will first build the cache by calling _build_cache_of_metrics().
        Parameters:
            metrics (list[Metric]): A list of metrics to retrieve data for.
        Returns:
            DataRecord: A data record containing the cached entities and measurements
                        for the requested metrics.
        """
        measurements = []
        if (
            len(self._cache_cpu_core_pid_aperf) == 0
            and len(self._cache_cpu_core_pid_mperf) == 0
            and len(self._cache_cpu_core_pid_unhalted_core_cycles) == 0
            and len(self._cache_cpu_core_pid_time) == 0
            and len(self._cache_cpu_core_pid_execution_type) == 0
            and len(self._cache_cpu_core_pid_work) == 0
            # and len(self._cache_cpu_core_pid_l2_miss) == 0
            and len(self._cache_cpu_core_pid_load_local) == 0
            and len(self._cache_cpu_core_pid_load_remote) == 0
            and len(self._cache_cpu_core_pid_all_stores) == 0
            and len(self._cache_cpu_core_pid_demand_rfo_remote) == 0
        ):
            raw_data = self._get_raw_monitoring_data()
            self._calculate_and_cache_metrics(raw_data)

        for metric in metrics:
            if metric == CPU_CORE_PID_TIME:
                measurements = measurements + self._cache_cpu_core_pid_time
            elif metric == CPU_CORE_PID_APERF:
                measurements = measurements + self._cache_cpu_core_pid_aperf
            elif metric == CPU_CORE_PID_MPERF:
                measurements = measurements + self._cache_cpu_core_pid_mperf
            elif metric == CPU_CORE_PID_UNHALTED_CYCLES:
                measurements = measurements + self._cache_cpu_core_pid_unhalted_core_cycles
            elif metric == CPU_CORE_PID_EXECUTION_TYPE:
                measurements = measurements + self._cache_cpu_core_pid_execution_type
            elif metric == CPU_CORE_PID_WORK:
                measurements = measurements + self._cache_cpu_core_pid_work
            # elif metric == CPU_CORE_PID_L2_MISS:
            #     measurements = measurements + self._cache_cpu_core_pid_l2_miss
            elif metric == CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM:
                measurements = measurements + self._cache_cpu_core_pid_load_local
            elif metric == CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM:
                measurements = measurements + self._cache_cpu_core_pid_load_remote
            elif metric == CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES:
                measurements = measurements + self._cache_cpu_core_pid_all_stores

        return DataRecord(entities=[], measurements=measurements)

    @override
    @classmethod
    def is_applicable(cls, entity: Entity) -> bool:
        """Check if this monitor is applicable to the given entity.

        This class method determines if the SchedSwitchMonitor can be applied to the
        provided entity by checking if it's an instance of CPU.

        Parameters
        ----------
        entity : Entity
            The entity to check for applicability.

        Returns
        -------
        bool
            True if the entity is a CPU instance, False otherwise.
        """
        return isinstance(entity, CPU)

    @override
    def clean_up(self):
        """Clear all cached entities and measurements."""
        self._cache_cpu_core_pid_time.clear()
        self._cache_cpu_core_pid_aperf.clear()
        self._cache_cpu_core_pid_mperf.clear()
        self._cache_cpu_core_pid_execution_type.clear()
        self._cache_cpu_core_pid_unhalted_core_cycles.clear()
        self._cache_cpu_core_pid_work.clear()
        # self._cache_cpu_core_pid_l2_miss.clear()
        self._cache_cpu_core_pid_load_local.clear()
        self._cache_cpu_core_pid_load_remote.clear()
        self._cache_cpu_core_pid_all_stores.clear()
        self._cache_cpu_core_pid_demand_rfo_remote.clear()

    @override
    def stop(self, metrics: list[Metric]) -> DataRecord | None:
        self._rb_sched_switch.stop()

        if self._producer_finished:
            # The sentinel was already consumed during normal operation, so there is nothing
            # left to wait for beyond the items still in flight.
            df_paths = drain_remaining(self._queue_monitoring_data, name="sched_switch")
        else:
            df_paths, seen = drain_until_sentinel(
                self._queue_monitoring_data,
                self._ring_buffer_config.sentinel,
                deadline_s=30.0,
                name="sched_switch",
            )
            self._producer_finished = seen
            if not seen:
                self._report_dead_producer()
                logger.warning("sched_switch: sentinel not received, continuing anyway")

        df_list = [read_temp_ipc(path) for path in df_paths]

        # build final cache of metrics
        if df_list:
            df = pl.concat(df_list, how="vertical")
            self._calculate_and_cache_metrics(df)
        final_data_record = self.get_data(metrics)

        self._rb_sched_switch.join(timeout=15)
        if self._rb_sched_switch.is_alive():
            logger.warning("sched_switch BPF process did not exit, killing it")
            self._rb_sched_switch.kill()
            self._rb_sched_switch.join(timeout=5)

        return final_data_record
