"""Tests for sched_switch collection plan resolution."""

from metrion.metric import (
    CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM,
    CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM,
    CPU_CORE_PID_TIME,
    CPU_CORE_PID_UNHALTED_CYCLES,
    CPU_CORE_PID_WORK,
)
from metrion.monitoring.data_source.cpu.collection_plan import (
    SchedSwitchCollectionPlan,
    resolve_sched_switch_plan,
)


def test_minimal_cpu_energy_config_excludes_dram_perf():
    plan = resolve_sched_switch_plan(
        [
            CPU_CORE_PID_WORK,
            CPU_CORE_PID_TIME,
        ]
    )
    assert plan == SchedSwitchCollectionPlan(
        needs_cpu_perf=True,
        needs_l3_local=False,
        needs_l3_remote=False,
        needs_stores=False,
    )


def test_l3_metrics_enable_dram_perf_counters():
    plan = resolve_sched_switch_plan(
        [
            CPU_CORE_PID_WORK,
            CPU_CORE_PID_MEM_LOAD_L3_MISS_LOCAL_DRAM,
            CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM,
        ]
    )
    assert plan.needs_cpu_perf is True
    assert plan.needs_l3_local is True
    assert plan.needs_l3_remote is True
    assert plan.needs_stores is False


def test_unhalted_cycles_without_work_still_needs_cpu_perf():
    plan = resolve_sched_switch_plan([CPU_CORE_PID_UNHALTED_CYCLES])
    assert plan.needs_cpu_perf is True
    assert plan.needs_l3_local is False


def test_stores_metric_enables_store_counter():
    plan = resolve_sched_switch_plan([CPU_CORE_PID_MEM_INST_RETIRED_ALL_STORES])
    assert plan.needs_stores is True
    assert plan.needs_cpu_perf is False


def test_empty_metrics_defaults_to_all_enabled():
    plan = resolve_sched_switch_plan(None)
    assert plan == SchedSwitchCollectionPlan.all_enabled()


def test_perf_groups_summary_minimal_cpu_energy_config():
    plan = resolve_sched_switch_plan([CPU_CORE_PID_WORK, CPU_CORE_PID_TIME])
    collecting, skipping = plan.perf_groups_summary()
    assert collecting == ["cpu_perf (ucc+aperf+mperf)"]
    assert "l3_miss_local_dram" in skipping
    assert "l3_miss_remote_dram" in skipping
    assert "mem_inst_retired_all_stores" in skipping
