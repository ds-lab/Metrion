import queue
import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock

import polars as pl
import pytest

from metrion.util import write_temp_ipc

ler = pytest.importorskip("metrion.entity.logical_entity_registry", reason="needs bcc")

NEW, EXEC = 1, 2
START_NS = 8_120_000_000_000


@pytest.fixture
def registry(monkeypatch):
    """__init__ scans /proc and starts a BPF process, so the state it sets up is built here."""
    monkeypatch.setattr(ler, "build_cgroup_id_map", dict)
    registry = object.__new__(ler.LogicalEntityRegistry)
    registry._entities, registry._entities_by_pid, registry._bucket_index = {}, {}, {}
    registry._pending = set()
    registry._state_lock = threading.RLock()
    registry._queue = queue.Queue()
    registry._config = MagicMock(sentinel="SENTINEL")
    registry._producer_finished = False
    registry._clk_tck = 100
    registry._boot_time_dt = datetime(2026, 9, 10, tzinfo=UTC)
    return registry


def _event(event_type, tid, comm, cmdline):
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
        "cgroup_id": 0,
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
