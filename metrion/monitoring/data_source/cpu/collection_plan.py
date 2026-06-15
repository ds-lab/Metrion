"""Resolve which perf primitives sched_switch must collect for enabled metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from metrion.metric import (
    CPU_CORE_PID_APERF,
    CPU_CORE_PID_EXECUTION_TYPE,
    CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM,
    CPU_CORE_PID_MPERF,
    CPU_CORE_PID_TIME,
    CPU_CORE_PID_UNHALTED_CYCLES,
    CPU_CORE_PID_WORK,
    Metric,
)

_CPU_PERF_METRICS = frozenset(
    {
        CPU_CORE_PID_TIME,
        CPU_CORE_PID_WORK,
        CPU_CORE_PID_UNHALTED_CYCLES,
        CPU_CORE_PID_APERF,
        CPU_CORE_PID_MPERF,
        CPU_CORE_PID_EXECUTION_TYPE,
    }
)


@dataclass(frozen=True, slots=True)
class SchedSwitchCollectionPlan:
    """Perf counter groups required by the enabled sched_switch metrics."""

    needs_cpu_perf: bool
    needs_l3_local: bool
    needs_l3_remote: bool
    needs_stores: bool

    @classmethod
    def all_enabled(cls) -> SchedSwitchCollectionPlan:
        return cls(
            needs_cpu_perf=True,
            needs_l3_local=True,
            needs_l3_remote=True,
            needs_stores=True,
        )

    def perf_groups_summary(self) -> tuple[list[str], list[str]]:
        """Return (enabled, disabled) perf counter group labels for logging."""
        groups = (
            ("needs_cpu_perf", "cpu_perf (ucc+aperf+mperf)"),
            ("needs_l3_local", "l3_miss_local_dram"),
            ("needs_l3_remote", "l3_miss_remote_dram"),
            ("needs_stores", "mem_inst_retired_all_stores"),
        )
        enabled: list[str] = []
        disabled: list[str] = []
        for field, label in groups:
            if getattr(self, field):
                enabled.append(label)
            else:
                disabled.append(label)
        return enabled, disabled


def resolve_sched_switch_plan(enabled_metrics: Sequence[Metric] | None) -> SchedSwitchCollectionPlan:
    """Map enabled metrics to the minimal perf counter groups to open."""
    if not enabled_metrics:
        return SchedSwitchCollectionPlan.all_enabled()

    enabled = set(enabled_metrics)
    enabled_names = {metric.name for metric in enabled}

    return SchedSwitchCollectionPlan(
        needs_cpu_perf=bool(enabled & _CPU_PERF_METRICS),
        needs_l3_local=CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM.name in enabled_names,
        needs_l3_remote=CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM.name in enabled_names,
        needs_stores=CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES.name in enabled_names,
    )
