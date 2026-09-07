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


def _logical_entity(pid, cmdline="python3", cgroup=None):
    from metrion.entity.logical_entity import LogicalEntity

    t = datetime(2026, 1, 1, tzinfo=UTC)
    return LogicalEntity(
        native_thread_id=pid,
        native_pid=pid,
        native_parent_pid=1,
        native_parent_thread_id=1,
        executable_name="python3",
        cmdline=cmdline,
        start_time=t,
        pid_start_time=t,
        parent_tid_start_time=t,
        parent_pid_start_time=t,
        resource_group_name=cgroup,
    )


def _cpu_with_cores(count=2):
    """Builds a CPU the way hardware detection does - package_id is a string there."""
    from metrion.entity.core import Core
    from metrion.entity.cpu import CPU

    cores = [Core(core_id=i, package_id="0", thread_siblings=[i], mhz=3000, parent_id="host-1") for i in range(count)]
    return CPU(
        manufacturer="GenuineIntel",
        model="test",
        physical_cores=count,
        logical_cores=count,
        package_id="0",
        cache=8192,
        hyperthreading=False,
        topology={i: 0 for i in range(count)},
        cores=cores,
        idle_power_package=1.0,
        parent_id="host-1",
    )


def test_entities_are_written_per_type(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_logical_entity(100), _logical_entity(101)])
    storage.close()

    df = pl.read_parquet(tmp_path / "entities" / "logicalentity" / "*.parquet")
    assert df.height == 2
    assert set(df["native_pid"].to_list()) == {100, 101}


def test_types_do_not_share_a_stream(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_logical_entity(100), _cpu_with_cores(0)])
    storage.close()

    streams = sorted(p.name for p in (tmp_path / "entities").iterdir())
    assert streams == ["cpu", "logicalentity"]


def test_cpu_cores_get_their_own_stream(tmp_path):
    """CPUs carry their cores as sub-objects; the DB path writes them, so this has to as well."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_cpu_with_cores(4)])
    storage.close()

    cores = pl.read_parquet(tmp_path / "entities" / "core" / "*.parquet")
    assert cores.height == 4
    assert sorted(cores["core_id"].to_list()) == [0, 1, 2, 3]


def test_same_entity_written_twice_is_kept_twice(tmp_path):
    """Append-only is deliberate: the second version carries better metadata."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_logical_entity(100, cmdline="")])
    storage.save_entities([_logical_entity(100, cmdline="/usr/bin/python3 -m foo")])
    storage.close()

    df = pl.read_parquet(tmp_path / "entities" / "logicalentity" / "*.parquet")
    assert df.height == 2
    assert df["id"].n_unique() == 1


def test_seq_orders_the_versions_of_one_entity(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_logical_entity(100, cmdline="")])
    storage.save_entities([_logical_entity(100, cmdline="besser")])
    storage.close()

    df = pl.read_parquet(tmp_path / "entities" / "logicalentity" / "*.parquet")
    latest = df.sort("_seq").group_by("id").last()
    assert latest.height == 1
    assert latest["cmdline"][0] == "besser"


def test_seq_is_unique_per_save_call(tmp_path):
    """Two calls without a pause must stay distinguishable - a timestamp would not be."""
    storage = ParquetDataStorage(str(tmp_path))
    for _ in range(3):
        storage.save_entities([_logical_entity(100)])
    storage.close()

    df = pl.read_parquet(tmp_path / "entities" / "logicalentity" / "*.parquet")
    assert df["_seq"].to_list() == [1, 2, 3]


def test_one_call_shares_a_seq_across_streams(tmp_path):
    """A CPU and its cores are one snapshot and must not look like two versions."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_cpu_with_cores(2)])
    storage.close()

    cpu = pl.read_parquet(tmp_path / "entities" / "cpu" / "*.parquet")
    cores = pl.read_parquet(tmp_path / "entities" / "core" / "*.parquet")
    assert cpu["_seq"].to_list() == [1]
    assert cores["_seq"].to_list() == [1, 1]


def test_empty_entity_list_is_a_noop(tmp_path):
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([])
    storage.close()

    assert not (tmp_path / "entities").exists()


def test_physical_entities_keep_their_parent_id(tmp_path):
    """The dram analysis joins entity.parent_id, so it has to survive into the stream."""
    from metrion.entity.dram import DRAM

    cpu = _cpu_with_cores(2)
    dram = DRAM(total_memory=16.0, cpu=cpu, idle_power=1.0, max_power=8.0, parent_id=cpu.id)

    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([cpu, dram])
    storage.close()

    written = pl.read_parquet(tmp_path / "entities" / "dram" / "*.parquet")
    assert written["parent_id"].to_list() == [cpu.id]
    assert written["package_id"].to_list() == ["0"]


def test_two_versions_in_one_call_are_both_kept(tmp_path):
    """Append-only holds within a single call too - nothing is deduplicated by id."""
    storage = ParquetDataStorage(str(tmp_path))
    storage.save_entities([_logical_entity(100, cmdline=""), _logical_entity(100, cmdline="besser")])
    storage.close()

    df = pl.read_parquet(tmp_path / "entities" / "logicalentity" / "*.parquet")
    assert df.height == 2
    assert df["id"].n_unique() == 1
