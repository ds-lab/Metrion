from unittest.mock import MagicMock

import polars as pl
import pytest

pytest.importorskip("bcc", reason="metrion.monitoring.data_source.cpu.sched_switch imports bcc")

from metrion.monitoring.data_source.cpu.collection_plan import SchedSwitchCollectionPlan  # noqa: E402
from metrion.monitoring.data_source.cpu.sched_switch import SchedSwitchMonitor  # noqa: E402


class FakeMonitor:
    _calculate_and_cache_metrics = SchedSwitchMonitor._calculate_and_cache_metrics

    def __init__(self):
        self._raw_events_received = 0
        self._total_work_data = 0
        self._validate_bpf_clock_domain = MagicMock()
        self._logical_entity_registry = MagicMock()
        self._collection_plan = SchedSwitchCollectionPlan(
            needs_cpu_perf=False, needs_l3_local=False, needs_l3_remote=False, needs_stores=False
        )


def _raw_events(event_types):
    n = len(event_types)
    counters = {
        f"{name}_{end}": [1] * n
        for name in ["aperf", "mperf", "unhalted_core_cycles", "loads_local", "loads_remote", "stores"]
        for end in ["start", "end"]
    }
    return pl.DataFrame(
        {
            "pid": list(range(100, 100 + n)),
            "pid_start": [7] * n,
            "tgid": [100] * n,
            "cpu_core": [0] * n,
            "time_start": list(range(n)),
            "time_end": list(range(1, n + 1)),
            **counters,
            "event_type": event_types,
            "tgid_start": [7 if t == 1 else None for t in event_types],
            "parent_tid": [1 if t == 1 else None for t in event_types],
            "parent_tgid": [1 if t == 1 else None for t in event_types],
            "parent_tid_start": [0 if t == 1 else None for t in event_types],
            "parent_tgid_start": [0 if t == 1 else None for t in event_types],
            "cgroup_id": [5 if t == 1 else None for t in event_types],
            "uid": [0 if t == 1 else None for t in event_types],
            "gid": [0 if t == 1 else None for t in event_types],
            "comm": ["worker" if t == 1 else None for t in event_types],
            "cmdline": ["worker --run" if t == 1 else None for t in event_types],
        }
    )


def test_only_events_with_metadata_reach_the_registry():
    monitor = FakeMonitor()

    monitor._calculate_and_cache_metrics(_raw_events([0, 1, 0, 0]))

    registry = monitor._logical_entity_registry
    registered = registry._register_entities_from_inline_metadata.call_args.args[0]
    enriched = registry.update_entity_metadata_from_df.call_args.args[0]
    for df in (registered, enriched):
        assert df["event_type"].to_list() == [1]
        assert df["pid"].to_list() == [101]
        assert df["comm"].to_list() == ["worker"]


def test_every_event_is_still_counted():
    monitor = FakeMonitor()

    monitor._calculate_and_cache_metrics(_raw_events([0, 1, 0, 0]))

    assert monitor._raw_events_received == 4
