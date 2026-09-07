import queue
import time
from unittest.mock import MagicMock

from metrion.data_record import DataRecord
from metrion.monitoring.data_source.data_source_runner import DataSourceRunner

WORK_S = 0.05


class SlowDataSource:
    """Data source whose get_data takes a fixed amount of time and records when it ran."""

    def __init__(self, entities, interval=None, enabled_metrics=None, timing=None):
        self.timestamps: list[float] = []

    def get_data(self, metrics):
        self.timestamps.append(time.monotonic())
        time.sleep(WORK_S)
        return DataRecord(entities=[], measurements=[])

    def clean_up(self):
        pass

    def stop(self, metrics):
        return None


def _run_for(interval, seconds):
    out = queue.Queue()
    runner = DataSourceRunner(SlowDataSource, [MagicMock()], [], interval, out)
    runner.start()
    time.sleep(seconds)
    runner.request_stop()
    runner.join(timeout=5)
    return runner


def test_period_stays_at_interval_despite_work_time():
    """The period is the interval, not the interval plus the time get_data takes."""
    runner = _run_for(0.2, 1.3)
    stamps = runner._datasource.timestamps
    assert len(stamps) >= 5, f"only {len(stamps)} samples"

    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    mean_gap = sum(gaps) / len(gaps)
    assert 0.18 < mean_gap < 0.24, f"mean gap {mean_gap:.3f}s, expected ~0.20s"

    drift = stamps[-1] - stamps[0] - (len(stamps) - 1) * 0.2
    assert abs(drift) < 0.1, f"offset of {drift:.3f}s adds up over the run"


def test_late_ticks_are_counted_when_work_exceeds_interval():
    runner = _run_for(0.02, 0.6)  # 20 ms interval, 50 ms of work
    assert runner._late_ticks > 0
    assert runner._datasource.timestamps


def test_no_sample_is_skipped_when_the_grid_is_realigned():
    """Realigning moves the next wake-up, it never skips a get_data call."""
    runner = _run_for(0.02, 0.5)
    stamps = runner._datasource.timestamps

    # 50 ms of work per round is the floor for the period, so ~10 rounds fit into 0.5 s.
    assert len(stamps) >= 6, f"only {len(stamps)} samples"
    assert all(b - a >= WORK_S for a, b in zip(stamps, stamps[1:], strict=False))
