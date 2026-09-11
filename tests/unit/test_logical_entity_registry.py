import queue
import threading
from collections import deque
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

from metrion.util import write_temp_ipc

ler = pytest.importorskip("metrion.entity.logical_entity_registry", reason="needs bcc")

NEW, EXEC, EXIT = 1, 2, 4
START_NS = 8_120_000_000_000


@pytest.fixture
def registry(monkeypatch):
    """__init__ scans /proc and starts a BPF process, so the state it sets up is built here."""
    monkeypatch.setattr(ler, "build_cgroup_id_map", dict)
    registry = object.__new__(ler.LogicalEntityRegistry)
    registry._entities, registry._starts_by_tid, registry._exited = {}, {}, deque()
    registry._forgotten = []
    registry._pending = set()
    registry._state_lock = threading.RLock()
    registry._queue = queue.Queue()
    registry._config = MagicMock(sentinel="SENTINEL")
    registry._producer_finished = False
    registry._clk_tck = 100
    registry._boot_time_dt = datetime(2026, 9, 10, tzinfo=UTC)
    registry._cgroup_id_map = {}
    registry._no_match_count = 0
    return registry


def _event(event_type, tid, comm, cmdline, cgroup_id=0):
    return {
        "tid": tid,
        "tgid": tid,
        "tid_start_time": START_NS + tid,
        "tgid_start_time": START_NS + tid,
        "parent_tid": 3000,
        "parent_tgid": 3000,
        "parent_tid_start_time": 0,
        "parent_tgid_start_time": 0,
        "event_type": event_type,
        "uid": 1000,
        "gid": 1000,
        "cgroup_id": cgroup_id,
        "old_tid": 0,
        "comm": comm,
        "cmdline": cmdline,
    }


def _deliver(registry, events):
    """Hand the events to the registry the way the BPF process does: as one IPC file."""
    registry._queue.put(write_temp_ipc(pl.DataFrame(events)))


def test_update_applies_the_events_in_the_order_they_happened(registry):
    """EXEC renames the entity its NEW created; applied the other way round, the fork-time name wins."""
    events = []
    for tid in range(1000, 1200):
        events += [_event(NEW, tid, "bash", "/bin/bash"), _event(EXEC, tid, "python3", "python3 train.py")]
    _deliver(registry, events)

    registry.update()

    assert {entity.executable_name for entity in registry._entities.values()} == {"python3"}


def test_an_exec_after_the_report_is_reported_again(registry):
    """Otherwise storage keeps the name the process had when it was first reported."""
    # The names are read right away: the report hands out the entity objects, which EXEC changes later.
    _deliver(registry, [_event(NEW, 4711, "bash", "/bin/bash")])
    registry.update()
    first = [entity.executable_name for entity in registry.get_new_entities()]
    unchanged = registry.get_new_entities()

    _deliver(registry, [_event(EXEC, 4711, "python3", "python3 train.py")])
    registry.update()
    after_exec = [entity.executable_name for entity in registry.get_new_entities()]

    assert first == ["bash"]
    assert unchanged == [], "nothing changed, so nothing is reported"
    assert after_exec == ["python3"]


def test_the_cgroup_map_is_walked_only_for_an_id_it_does_not_know(registry, monkeypatch):
    """In run 2, 263 threads sat in one cgroup the walk never finds; each must not walk /sys/fs/cgroup again."""
    walks = 0

    def walk():
        nonlocal walks
        walks += 1
        return {7: "/system.slice/train.service"}

    monkeypatch.setattr(ler, "build_cgroup_id_map", walk)

    for tid, cgroup_id in [(1000, 7), (1001, 7), (1002, 99), (1003, 99)]:
        _deliver(registry, [_event(NEW, tid, "python3", "python3 train.py", cgroup_id)])
        registry.update()

    names = [registry._entities[(tid, START_NS + tid)].resource_group_name for tid in range(1000, 1004)]
    assert names == ["/system.slice/train.service"] * 2 + ["unknown"] * 2
    assert walks == 2, "one walk for each id the map did not know, none for the ones it did"


def test_a_start_cut_to_clock_ticks_still_finds_the_thread(registry):
    """NVML and ProcNuma read the start from /proc in 10-ms ticks; the registry keeps it from BPF in ns."""
    start_ns = START_NS + 4_500_000
    _deliver(registry, [dict(_event(NEW, 4712, "python3", "python3 train.py"), tid_start_time=start_ns)])
    registry.update()

    assert registry.get_logical_entity(4712, START_NS) is not None, "4.5 ms off: the same thread"
    assert registry.get_logical_entity(4712, start_ns + 70_000_000) is None, "70 ms off: another thread"


def test_an_exited_thread_is_forgotten_after_the_grace_period(registry, monkeypatch):
    """Its late measurements still find it until then, and a new thread that reuses its tid stays."""
    now = [0.0]
    monkeypatch.setattr(ler, "time", SimpleNamespace(monotonic=lambda: now[0]))
    first_start, second_start = START_NS + 4711, START_NS + 9_000_000_000

    _deliver(registry, [_event(NEW, 4711, "python3", "python3 train.py")])
    # A thread from before metrion started exits with start 0: BPF never saw it being created.
    _deliver(registry, [dict(_event(EXIT, 4711, "python3", "python3 train.py"), tid_start_time=0)])
    registry.update()
    _deliver(registry, [dict(_event(NEW, 4711, "cron", "/usr/sbin/cron"), tid_start_time=second_start)])
    registry.update()

    registry.get_new_entities()
    first = registry.get_logical_entity(4711, first_start)
    assert first is not None, "still inside the grace period"
    assert registry.get_forgotten_ids() == []

    now[0] = registry._EXIT_GRACE_S + 1
    registry.get_new_entities()

    assert registry.get_logical_entity(4711, first_start) is None
    assert registry.get_logical_entity(4711, second_start).executable_name == "cron"
    assert registry.get_forgotten_ids() == [first.id], "the saver drops its identity in turn"
