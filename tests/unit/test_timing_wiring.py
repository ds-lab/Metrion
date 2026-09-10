import queue

import pytest

from metrion.config_reader.config_schema import TimingConfig
from metrion.data_collector import DataCollector
from metrion.entity.entity import Entity
from metrion.monitoring.data_source.cpu.ring_buff_base import EbpfRingBufferProcess, RingBufferConfig
from metrion.monitoring.data_source.data_source_base import DataSource
from metrion.monitoring.data_source.data_source_runner import DataSourceRunner
from metrion.monitoring.entity_monitor import EntityTypeMonitor


class _Thing(Entity):
    pass


class _RecordingSource(DataSource):
    """Records what the runner constructed it with, without touching any hardware."""

    seen = {}

    def __init__(self, entities, interval=None, enabled_metrics=None, timing=None):
        _RecordingSource.seen = {"interval": interval, "timing": timing}

    def get_data(self, metrics):
        return None

    def clean_up(self):
        pass

    @classmethod
    def is_applicable(cls, entity):
        return True


class _BareProcess(EbpfRingBufferProcess):
    """Concrete subclass so the base __init__ can run; nothing is started."""

    def _create_bpf(self):
        raise NotImplementedError

    def _attach_ringbuf(self):
        raise NotImplementedError

    def _event_to_record(self, event):
        raise NotImplementedError


def test_runner_hands_the_timing_section_to_the_data_source():
    timing = TimingConfig(ebpf_flush=0.25, storage_flush=0.5)

    DataSourceRunner(_RecordingSource, [], [], 1.0, queue.Queue(), timing)

    assert _RecordingSource.seen["timing"] is timing


def test_runner_without_timing_leaves_the_source_on_defaults():
    DataSourceRunner(_RecordingSource, [], [], 1.0, queue.Queue())

    assert _RecordingSource.seen["timing"] is None


def test_data_source_falls_back_to_the_defaults():
    """A source constructed without timing must not crash on self._timing."""
    source = DataSource.__new__(_RecordingSource)
    DataSource.__init__(source, [_Thing()], interval=1.0)

    assert source._timing.ebpf_flush == 1.0
    assert source._timing.storage_flush == 2.0


def test_entity_monitor_passes_its_timing_to_every_runner(monkeypatch):
    timing = TimingConfig(storage_flush=0.5)
    monitor = EntityTypeMonitor.__new__(EntityTypeMonitor)
    monitor._timing = timing
    monitor._entities = []
    monitor._data_source_metrics = {_RecordingSource: []}
    monitor._data_source_runners = {}

    monitor._setup()

    assert next(iter(monitor._data_source_runners))._datasource is not None
    assert _RecordingSource.seen["timing"] is timing


def test_writer_flush_reaches_the_producer_process():
    """timing.storage_flush travels into the eBPF producer through RingBufferConfig."""
    config = RingBufferConfig(buffer_flush_threshold=10, buffer_flush_time_threshold=0.25, writer_flush_s=0.5)

    process = _BareProcess(config)

    assert process._buffer_flush_time_threshold == 0.25
    assert process._writer_flush_s == 0.5


def test_writer_flush_has_a_default_for_the_entity_registry():
    """LogicalEntityRegistry builds a RingBufferConfig without the field and must keep working."""
    config = RingBufferConfig(buffer_flush_threshold=100, buffer_flush_time_threshold=0.5)

    assert config.writer_flush_s == 2.0


def test_flush_interval_reaches_the_agents_writer(tmp_path, monkeypatch):
    """timing.storage_flush is the age at which the writer in front of storage stops waiting."""
    monkeypatch.setattr(DataCollector, "_start_collection_process", lambda self: None)

    collector = DataCollector("duckdb", str(tmp_path / "t.db"), flush_interval=0.25)

    assert collector.ipc_writer._max_age_s == 0.25


@pytest.mark.parametrize("field", ["ebpf_flush", "storage_flush"])
def test_new_timing_fields_reject_non_positive(field):
    with pytest.raises(ValueError):
        TimingConfig(**{field: 0})
