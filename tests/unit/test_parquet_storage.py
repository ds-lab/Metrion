import json
from datetime import UTC, datetime

import polars as pl

from metrion.data_storage.parquet_data_storage import ParquetDataStorage
from metrion.measurement import Measurement
from metrion.metric import CPU_CORE_PID_WORK


def _measurement_df(n=5, offset=0):
    now = datetime.now(UTC)
    return pl.DataFrame(
        {
            "id": [1000 + offset + i for i in range(n)],
            "metric_id": ["CPU_CORE_PID_WORK"] * n,
            "physical_entity_id": [f"core_{i}" for i in range(n)],
            "logical_entity_id": [f"le_{i}" for i in range(n)],
            "value": [float(i) for i in range(n)],
            "start_ts": [now] * n,
            "end_ts": [now] * n,
        }
    )


def test_measurements_are_written_and_readable(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_measurements([_measurement_df(5)])
    storage.close()

    df = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert df.height == 5
    assert set(df.columns) >= {"id", "metric_id", "value", "start_ts", "end_ts"}


def test_repeated_saves_append_and_do_not_overwrite(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    for batch in range(4):
        storage.save_measurements([_measurement_df(5, offset=batch * 100)])
    storage.close()

    df = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert df.height == 20
    assert df["id"].n_unique() == 20


def test_rotation_creates_multiple_parts(tmp_path):
    storage = ParquetDataStorage(str(tmp_path), rotate_rows=10)
    for _ in range(5):
        storage.save_measurements([_measurement_df(5)])
    storage.close()

    parts = sorted((tmp_path / "measurements").glob("*.parquet"))
    assert len(parts) > 1, "after every 10 rows a new part file has to be started"
    assert pl.read_parquet(tmp_path / "measurements" / "*.parquet").height == 25


def test_close_flushes_what_is_still_pending(tmp_path):
    """Without close() the last, incomplete part would stay unwritten."""
    storage = ParquetDataStorage(str(tmp_path), rotate_rows=1_000)
    storage.save_measurements([_measurement_df(5)])
    assert not list((tmp_path / "measurements").glob("*.parquet"))

    storage.close()

    assert pl.read_parquet(tmp_path / "measurements" / "*.parquet").height == 5


def test_measurement_objects_are_accepted(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    measurement = Measurement(value=1.5, metric_id=CPU_CORE_PID_WORK.name, physical_entity_id="core_0")
    storage.save_measurements([measurement])
    storage.close()

    df = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert df.height == 1
    assert df["value"][0] == 1.5


def test_communication_measurements_go_to_own_stream(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    df = _measurement_df(3).with_columns(pl.lit("a").alias("src_endpoint"), pl.lit("b").alias("dst_endpoint"))
    storage.save_measurements([df])
    storage.close()

    assert (tmp_path / "communication_measurements").exists()
    assert not (tmp_path / "measurements").exists()


def test_frames_without_ids_get_them(tmp_path):
    """The interface allows frames without an id; both sinks have to produce the column."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_measurements([_measurement_df(3).drop("id")])
    storage.close()

    df = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert df["id"].null_count() == 0
    assert df["id"].n_unique() == 3


def test_a_missing_column_does_not_cost_the_pending_buffer(tmp_path):
    """Frames are buffered across calls, so one odd schema must not take the rest with it."""
    storage = ParquetDataStorage(str(tmp_path), rotate_rows=1_000)
    storage.save_measurements([_measurement_df(2)])
    storage.save_measurements([_measurement_df(2, offset=100).drop("logical_entity_id")])
    storage.close()

    df = pl.read_parquet(tmp_path / "measurements" / "*.parquet")
    assert df.height == 4
    assert df["logical_entity_id"].null_count() == 2


def test_unknown_measurement_types_are_reported_not_dropped_silently(tmp_path):
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        storage = ParquetDataStorage(str(tmp_path))
        storage.save_measurements(["/tmp/some_ipc_file.arrow"])
        storage.close()
    finally:
        logger.remove(sink_id)

    assert any("Unknown measurement type" in message for message in messages)


def test_manifest_records_row_counts(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_measurements([_measurement_df(7)])
    storage.close()

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["streams"]["measurements"] == 7
