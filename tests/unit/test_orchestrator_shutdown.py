import os
import signal
import threading
import time
from unittest.mock import MagicMock

import pytest

from metrion.orchestrator import Orchestrator

HANDLED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


@pytest.fixture(autouse=True)
def restore_signal_handlers():
    """Orchestrator() installs process-wide handlers; keep them out of other tests."""
    saved = {sig: signal.getsignal(sig) for sig in HANDLED_SIGNALS}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


@pytest.fixture
def orchestrator(tmp_path):
    orch = Orchestrator(str(tmp_path / "config.yaml"))
    orch._config = MagicMock()
    orch._config.general.collection_interval = 0.05
    orch._logical_entity_registry = MagicMock()
    orch._logical_entity_registry.get_new_entities.return_value = []
    orch._data_collector = MagicMock()
    orch._entity_monitors = []
    orch.stop = MagicMock(side_effect=lambda: setattr(orch, "_running", False))
    return orch


def test_first_signal_stops_the_loop_and_calls_stop_once(orchestrator):
    thread = threading.Thread(target=orchestrator.start_monitoring, daemon=True)
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive(), "monitoring loop should still be running"

    orchestrator._signal_handler(signal.SIGINT, None)
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "loop must terminate after the signal"
    assert orchestrator.stop.call_count == 1


def test_handler_does_not_ignore_further_signals(orchestrator):
    orchestrator._signal_handler(signal.SIGINT, None)

    assert orchestrator._force_exit_armed is True
    for sig in HANDLED_SIGNALS:
        assert signal.getsignal(sig) is not signal.SIG_IGN


def test_second_signal_exits_hard(orchestrator, monkeypatch):
    exited = []
    monkeypatch.setattr(os, "_exit", lambda code: exited.append(code))

    orchestrator._signal_handler(signal.SIGINT, None)
    assert exited == []

    orchestrator._signal_handler(signal.SIGINT, None)
    assert exited == [1]


def test_signal_in_a_child_process_is_ignored(orchestrator):
    orchestrator._main_pid = os.getpid() + 1

    orchestrator._signal_handler(signal.SIGINT, None)

    assert orchestrator._force_exit_armed is False
    assert not orchestrator._shutdown_requested.is_set()


def test_timeout_stops_the_loop_and_calls_stop_once(orchestrator):
    orchestrator.start_monitoring(timeout_minutes=0.5 / 60)

    assert orchestrator.stop.call_count == 1
    assert orchestrator._shutdown_requested.is_set()


def test_loop_returns_promptly_on_shutdown_instead_of_sleeping(orchestrator):
    orchestrator._config.general.collection_interval = 10.0

    thread = threading.Thread(target=orchestrator.start_monitoring, daemon=True)
    thread.start()
    time.sleep(0.3)

    start = time.monotonic()
    orchestrator._signal_handler(signal.SIGINT, None)
    thread.join(timeout=5.0)
    elapsed = time.monotonic() - start

    assert not thread.is_alive()
    assert elapsed < 2.0, f"waited {elapsed:.1f}s for a 10s interval to elapse"
