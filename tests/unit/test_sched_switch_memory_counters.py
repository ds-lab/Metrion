import ctypes
from types import SimpleNamespace

import pytest
from loguru import logger

ss = pytest.importorskip(
    "metrion.monitoring.data_source.cpu.sched_switch", reason="needs bcc and the built metrion_rust extension"
)

REMOTE = "mem_load_l3_miss_retired.remote_dram"


class _Attr(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("config", ctypes.c_uint64),
        ("config1", ctypes.c_uint64),
    ]


@pytest.fixture
def monitor(monkeypatch):
    """A machine where perf knows every memory counter except the remote one."""
    resolved = []

    def resolve(event, cpu=None):
        resolved.append(event)
        if event == REMOTE:
            raise RuntimeError("event syntax error")
        return {"type": 4, "config": 0x1D3}

    monkeypatch.setattr(ss, "resolve_with_strace", resolve)
    monkeypatch.setattr(
        ss,
        "CustomPerf",
        SimpleNamespace(perf_event_attr=_Attr, perf_custom_event_open_with_results=lambda attr, pid: [7, 8]),
    )
    m = ss.SchedSwitchBpfMonitor.__new__(ss.SchedSwitchBpfMonitor)
    m._collection_plan = SimpleNamespace(needs_l3_local=True, needs_l3_remote=True, needs_stores=True)
    m.resolved = resolved
    return m


def _warnings(action):
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        result = action()
    finally:
        logger.remove(sink_id)
    return result, messages


def test_a_missing_counter_is_warned_about_and_the_others_still_open(monitor):
    """A missing counter must not end the probe process and take the CPU data with it."""
    (local, remote, stores), warnings = _warnings(lambda: monitor._open_memory_events(False))

    assert (local, remote, stores) == ([7, 8], [], [7, 8])
    assert len(warnings) == 1
    assert REMOTE in warnings[0]


def test_hybrid_cpus_say_that_memory_counters_are_skipped(monitor):
    fds, warnings = _warnings(lambda: monitor._open_memory_events(True))

    assert fds == ([], [], [])
    assert len(warnings) == 1
    assert "hybrid" in warnings[0]
    assert monitor.resolved == []


def test_without_memory_metrics_nothing_is_opened_or_warned(monitor):
    monitor._collection_plan = SimpleNamespace(needs_l3_local=False, needs_l3_remote=False, needs_stores=False)

    fds, warnings = _warnings(lambda: monitor._open_memory_events(True))

    assert fds == ([], [], [])
    assert warnings == []
