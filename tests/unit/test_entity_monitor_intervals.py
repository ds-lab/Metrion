import pytest

from metrion.config_reader.config_schema import SourceIntervals, TimingConfig
from metrion.monitoring.entity_monitor import EntityTypeMonitor

_FIELD_BY_SOURCE = EntityTypeMonitor._INTERVAL_FIELD_BY_SOURCE


def _monitor(sources):
    """EntityTypeMonitor without __init__, which would start data source threads."""
    monitor = EntityTypeMonitor.__new__(EntityTypeMonitor)
    monitor._timing = TimingConfig(sources=sources)
    return monitor


def _source(name):
    return type(name, (), {})


def test_defaults_reproduce_previous_behaviour():
    """_setup used to hardcode 1.0 s for the energy sources and 5.0 s for everything else."""
    monitor = _monitor(SourceIntervals())

    assert monitor._interval_for(_source("RAPLCPU")) == 1.0
    assert monitor._interval_for(_source("NVMLDataSource")) == 1.0
    assert monitor._interval_for(_source("SchedSwitchMonitor")) == 5.0
    assert monitor._interval_for(_source("ProcNuma")) == 5.0
    assert monitor._interval_for(_source("TCPDataSource")) == 5.0


def test_every_sampling_field_is_wired():
    """A new field in SourceIntervals without a data source behind it fails here."""
    assert set(_FIELD_BY_SOURCE.values()) == set(SourceIntervals.model_fields)


@pytest.mark.parametrize("source_name,field", sorted(_FIELD_BY_SOURCE.items()))
def test_config_value_reaches_the_source(source_name, field):
    monitor = _monitor(SourceIntervals(**{field: 0.75}))

    assert monitor._interval_for(_source(source_name)) == 0.75


def test_unknown_source_falls_back_to_five_seconds():
    monitor = _monitor(SourceIntervals())

    assert monitor._interval_for(_source("SomethingNew")) == 5.0


def test_the_registered_data_source_classes_are_all_wired():
    """
    Catches a renamed or added data source class.

    The name table cannot notice either on its own, and a miss is silent: the source keeps
    running at the 5.0 s fallback while the config says otherwise.
    """
    pytest.importorskip("metrion.metrion_rust")
    pytest.importorskip("pynvml")
    from metrion.monitoring.data_source.data_source_factory import DataSourceFactory

    DataSourceFactory.auto_register()
    registered = {
        source.__name__
        for by_metric in DataSourceFactory.registered_sources.values()
        for sources in by_metric.values()
        for source in sources
    }

    assert registered == set(_FIELD_BY_SOURCE), (
        f"not wired: {sorted(registered - set(_FIELD_BY_SOURCE))}, "
        f"stale entries: {sorted(set(_FIELD_BY_SOURCE) - registered)}"
    )
