from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest

from metrion.data_storage.parquet_data_storage import ParquetDataStorage
from metrion.measurement import Measurement
from metrion.util import read_temp_ipc, write_temp_ipc

BERLIN = timezone(timedelta(hours=2))


def _frame(start, end=None):
    return pl.DataFrame(
        {
            "id": [1],
            "metric_id": ["CPU_CORE_PID_WORK"],
            "physical_entity_id": ["core_0"],
            "logical_entity_id": ["le_0"],
            "value": [1.0],
            "start_ts": [start],
            "end_ts": [end or start],
        }
    )


def test_a_measurement_is_stored_without_a_time_zone():
    df = Measurement(1.0, "M", "e1").to_df()

    assert df.schema["start_ts"] == pl.Datetime("us")
    assert df.schema["end_ts"] == pl.Datetime("us")


def test_a_measurement_in_another_time_zone_keeps_its_instant():
    """The wall clock used to be stamped as UTC, which moved the measurement by the offset."""
    berlin = datetime(2026, 9, 8, 11, 0, tzinfo=BERLIN)

    df = Measurement(1.0, "M", "e1", start_ts=berlin, end_ts=berlin).to_df()

    assert df["start_ts"][0] == datetime(2026, 9, 8, 9, 0)
    assert df["end_ts"][0] == datetime(2026, 9, 8, 9, 0)


def test_objects_and_frames_reach_one_parquet_file(tmp_path):
    """Both used to land in the same writer with different dtypes, which failed on flush."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_measurements([Measurement(1.0, "M", "e1"), _frame(datetime(2026, 9, 8, 9, 0))])
    storage.close()

    written = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert written.height == 2
    assert written.schema["start_ts"] == pl.Datetime("us")


def test_parquet_stores_a_zoned_frame_as_naive_utc(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_measurements([_frame(datetime(2026, 9, 8, 11, 0, tzinfo=BERLIN))])
    storage.close()

    written = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert written.schema["start_ts"] == pl.Datetime("us")
    assert written["start_ts"][0] == datetime(2026, 9, 8, 9, 0)


def test_the_ipc_handover_keeps_the_datetime_column():
    """The queue between the monitors and the collector carries datetimes, never strings."""
    df = _frame(datetime(2026, 9, 8, 9, 0))

    read_back = read_temp_ipc(write_temp_ipc(df))

    assert read_back.schema["start_ts"] == pl.Datetime("us")
    assert read_back["start_ts"][0] == datetime(2026, 9, 8, 9, 0)


def test_the_buffer_takes_measurement_objects():
    """Objects used to be run through a string parser, which their datetime columns rejected."""
    from metrion.energy_attribution.energy_attribution import DyncamicEnergyAttributionModel

    model = DyncamicEnergyAttributionModel()
    model.update_measurements([Measurement(1.0, "CPU_CORE_PID_WORK", "core_0")])

    assert model._measurements_df.height == 1


def test_sched_switch_hands_over_naive_utc():
    pytest.importorskip("bcc", reason="sched_switch imports bcc")
    from metrion.monitoring.data_source.cpu.sched_switch import SchedSwitchMonitor

    monitor = object.__new__(SchedSwitchMonitor)
    instant = datetime(2026, 9, 8, 9, 0)
    epoch_ns = int(instant.replace(tzinfo=UTC).timestamp() * 1e9)
    df = pl.DataFrame({"time_start_ns": [epoch_ns], "time_end_ns": [epoch_ns]})

    converted = monitor._convert_ns_to_datetime(df)

    assert converted.schema["time_start"] == pl.Datetime("us")
    assert converted["time_start"][0] == instant
