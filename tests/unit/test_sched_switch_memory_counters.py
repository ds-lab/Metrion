import ctypes
from types import SimpleNamespace

import pytest
from loguru import logger

ss = pytest.importorskip(
    "metrion.monitoring.data_source.cpu.sched_switch", reason="needs the built metrion_rust extension"
)

from metrion.config_reader.config_schema import MemoryEventsConfig  # noqa: E402
from metrion.monitoring.data_source.cpu.ring_buff_base import RingBufferConfig  # noqa: E402

LOCAL = "mem_load_l3_miss_retired.local_dram"
REMOTE = "mem_load_l3_miss_retired.remote_dram"
STORES = "mem_inst_retired.all_stores"


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
        SimpleNamespace(perf_event_attr=_Attr, perf_custom_event_open_with_results=lambda attr, pid, cpus: [7, 8]),
    )
    m = ss.SchedSwitchBpfMonitor.__new__(ss.SchedSwitchBpfMonitor)
    m._collection_plan = SimpleNamespace(needs_l3_local=True, needs_l3_remote=True, needs_stores=True)
    m._memory_events = MemoryEventsConfig()
    m.resolved = resolved
    return m


def _logged(action, level):
    """Run action and return its result with the (level name, message) of every record at level or above."""
    records = []
    sink_id = logger.add(lambda message: records.append((message.record["level"].name, str(message))), level=level)
    try:
        result = action()
    finally:
        logger.remove(sink_id)
    return result, records


def _warnings(action):
    result, records = _logged(action, "WARNING")
    return result, [message for _, message in records]


def test_a_missing_counter_is_warned_about_and_the_others_still_open(monitor):
    """A missing counter must not end the probe process and take the CPU data with it."""
    (local, remote, stores), warnings = _warnings(lambda: monitor._open_memory_events(False, [0, 1]))

    assert (local, remote, stores) == ([7, 8], [], [7, 8])
    assert len(warnings) == 1
    assert REMOTE in warnings[0]


def test_hybrid_cpus_say_that_memory_counters_are_skipped(monitor):
    fds, warnings = _warnings(lambda: monitor._open_memory_events(True, [0, 1]))

    assert fds == ([], [], [])
    assert len(warnings) == 1
    assert "hybrid" in warnings[0]
    assert monitor.resolved == []


def test_without_memory_metrics_nothing_is_opened_or_warned(monitor):
    monitor._collection_plan = SimpleNamespace(needs_l3_local=False, needs_l3_remote=False, needs_stores=False)

    fds, warnings = _warnings(lambda: monitor._open_memory_events(True, [0, 1]))

    assert fds == ([], [], [])
    assert warnings == []


def test_default_events_are_the_skylake_names(monitor):
    monitor._open_memory_events(False, [0, 1])

    assert monitor.resolved == [LOCAL, REMOTE, STORES]


def test_configured_events_are_opened_per_enabled_role(monitor):
    monitor._memory_events = MemoryEventsConfig(
        l3_miss_local_dram="a.local", l3_miss_remote_dram="b.remote", all_stores="c.stores"
    )
    monitor._collection_plan = SimpleNamespace(needs_l3_local=True, needs_l3_remote=False, needs_stores=True)

    fds = monitor._open_memory_events(False, [0, 1])

    assert monitor.resolved == ["a.local", "c.stores"]
    assert fds == ([7, 8], [], [7, 8])


def test_a_disabled_role_is_not_opened(monitor):
    monitor._collection_plan = SimpleNamespace(needs_l3_local=False, needs_l3_remote=False, needs_stores=True)

    fds = monitor._open_memory_events(False, [0, 1])

    assert monitor.resolved == [STORES]
    assert fds == ([], [], [7, 8])


def test_the_warning_names_event_key_and_perf_list(monitor):
    """The user has to find the config entry to fix and the command that lists this CPU's events."""
    _, warnings = _warnings(lambda: monitor._open_memory_events(False, [0, 1]))

    assert len(warnings) == 1
    assert REMOTE in warnings[0]
    assert "ebpf.memory_events.l3_miss_remote_dram" in warnings[0]
    assert "perf list mem_load" in warnings[0]


def test_an_opened_counter_is_logged_with_its_encoding(monitor):
    """The log shows which raw encoding perf resolved a configured name to."""
    _, records = _logged(lambda: monitor._open_memory_events(False, [0, 1]), "INFO")
    infos = [message for level, message in records if level == "INFO"]

    assert any(
        "ebpf.memory_events.l3_miss_local_dram" in message
        and LOCAL in message
        and "type=4" in message
        and "config=0x1d3" in message
        for message in infos
    )
    assert not any("l3_miss_remote_dram" in message for message in infos)


def test_resolve_with_strace_does_not_print(monkeypatch, capsys):
    from metrion.monitoring.data_source.cpu import utils

    monkeypatch.setattr(utils, "run_strace", lambda event, cpu=None, perf="perf": "type=PERF_TYPE_RAW, config=0x1d3")

    assert utils.resolve_with_strace("r01d3", cpu=0)["config"] == 0x1D3
    assert capsys.readouterr().out == ""


def test_memory_events_reach_the_probe_process(monkeypatch):
    monkeypatch.setattr(ss.SchedSwitchBpfMonitor, "start", lambda self: None)
    monkeypatch.setattr(ss.EbpfRingBufferProcess, "_singletons", {})
    custom = MemoryEventsConfig(all_stores="mem_uops_retired.all_stores")
    config = RingBufferConfig(buffer_flush_threshold=10, buffer_flush_time_threshold=1.0)

    started = ss.SchedSwitchBpfMonitor.start_singleton(config, memory_events=custom)
    built = ss.SchedSwitchBpfMonitor(config)

    assert started._memory_events is custom
    assert built._memory_events == MemoryEventsConfig()
