from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from loguru import logger

from metrion.monitoring.data_source.cpu.ring_buff_base import (
    _DROP_WARNING_INTERVAL_S,
    EbpfRingBufferProcess,
    RingBufferConfig,
)


@contextmanager
def captured(level):
    """Collect loguru messages at or above level; loguru does not write to caplog."""
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level=level)
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class _Fake(EbpfRingBufferProcess):
    """Concrete subclass with a drop count the test controls."""

    def __init__(self, config):
        super().__init__(config)
        self.drops = 0
        self._queue = MagicMock()

    def _get_drop_count(self):
        return self.drops

    def _create_bpf(self):
        raise NotImplementedError

    def _attach_ringbuf(self):
        raise NotImplementedError

    def _event_to_record(self, event):
        raise NotImplementedError


@pytest.fixture
def proc():
    return _Fake(RingBufferConfig(buffer_flush_threshold=10, buffer_flush_time_threshold=1.0))


def test_new_drops_are_warned_about_with_counts_and_rate(proc):
    proc._total_events = 1000
    proc.drops = 500

    with captured("WARNING") as messages:
        proc._check_missing_events()

    assert len(messages) == 1
    assert "500 of 1500 events" in messages[0]
    assert "33.33 %" in messages[0]


def test_no_warning_without_drops(proc):
    proc._total_events = 1000

    with captured("WARNING") as messages:
        proc._check_missing_events()

    assert messages == []


def test_a_drop_count_that_stops_growing_stops_warning(proc):
    proc._total_events = 1000
    proc.drops = 500

    with captured("WARNING") as messages:
        proc._check_missing_events()
        proc._check_missing_events()  # same count, nothing new happened

    assert len(messages) == 1


def test_further_drops_are_throttled(proc):
    """Every poll iteration checks for drops, so the warning has to be rate limited."""
    proc._total_events = 1000

    with captured("WARNING") as messages:
        for extra in range(1, 6):
            proc.drops = 500 + extra
            proc._check_missing_events()

    assert len(messages) == 1

    # Once the interval has passed, the next batch of drops is reported again.
    proc._last_drop_warning -= _DROP_WARNING_INTERVAL_S
    with captured("WARNING") as messages:
        proc.drops += 1
        proc._check_missing_events()

    assert len(messages) == 1


def test_buffered_events_count_towards_the_rate(proc):
    """Nothing is flushed yet, so without the buffered events one drop would read as 100 %."""
    proc._total_events = 0
    proc._buffer = [{}] * 999
    proc.drops = 1

    with captured("WARNING") as messages:
        proc._check_missing_events()

    assert "1 of 1000 events" in messages[0]
    assert "0.10 %" in messages[0]


def test_shutdown_reports_both_totals(proc):
    proc._total_events = 4200
    proc._missing_events = 7

    with captured("INFO") as messages:
        proc._stop_monitoring()

    assert any("Events processed: 4200, dropped in the kernel: 7" in m for m in messages)


def test_shutdown_still_reports_when_the_final_flush_fails(proc):
    proc._total_events = 4200
    proc._flush_to_queue = MagicMock(side_effect=RuntimeError("boom"))

    with captured("INFO") as messages:
        proc._stop_monitoring()

    assert any("Events processed: 4200" in m for m in messages)
