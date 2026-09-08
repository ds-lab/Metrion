import os
import time
from datetime import UTC, datetime

import duckdb
import polars as pl

from metrion.data_storage.parquet_data_storage import ParquetDataStorage
from metrion.data_storage.parquet_to_duckdb import convert, ensure_duckdb
from metrion.entity.core import Core
from metrion.entity.cpu import CPU
from metrion.entity.host import Host, HostType
from metrion.entity.logical_entity import LogicalEntity

T = datetime(2026, 1, 1, tzinfo=UTC)


def _mdf(n, offset=0):
    return pl.DataFrame(
        {
            "id": [offset + i for i in range(n)],
            "metric_id": ["CPU_CORE_PID_WORK"] * n,
            "physical_entity_id": ["core_0"] * n,
            "logical_entity_id": ["le_1"] * n,
            "value": [1.0] * n,
            "start_ts": [T] * n,
            "end_ts": [T] * n,
        }
    )


def _le(cmdline="python3", cgroup=None):
    return LogicalEntity(
        native_thread_id=42,
        native_pid=42,
        native_parent_pid=1,
        native_parent_thread_id=1,
        executable_name="python3",
        cmdline=cmdline,
        start_time=T,
        pid_start_time=T,
        parent_tid_start_time=T,
        parent_pid_start_time=T,
        resource_group_name=cgroup,
    )


def _cpu_with_cores(count=2):
    """Builds a CPU the way hardware_detection.py:461 does: cores carry the CPU's id."""
    cpu = CPU(
        manufacturer="Intel",
        model="i9",
        physical_cores=count,
        logical_cores=count,
        package_id="0",
        cache=1024,
        hyperthreading=False,
        topology={i: i for i in range(count)},
        cores=[],
        idle_power_package=1.0,
        parent_id="host_0",
    )
    cpu.cores = [Core(core_id=i, package_id="0", thread_siblings=[i], mhz=3000, parent_id=cpu.id) for i in range(count)]
    return cpu


def _query(db, sql):
    con = duckdb.connect(str(db))
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_measurements_survive_the_conversion(tmp_path):
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(10)])
    storage.save_measurements([_mdf(10, offset=100)])
    storage.close()

    counts = convert(tmp_path / "run", tmp_path / "out.db")

    assert counts["measurement"] == 20
    assert _query(tmp_path / "out.db", "SELECT count(*) FROM measurement")[0][0] == 20


def test_entity_duplicates_are_reduced_to_one_row(tmp_path):
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le(cmdline="")])
    storage.save_entities([_le(cmdline="/usr/bin/python3 -m foo")])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(tmp_path / "out.db", "SELECT cmdline FROM logical_entity")
    assert len(rows) == 1, "one row per id must remain"
    assert rows[0][0] == "/usr/bin/python3 -m foo"


def test_a_later_version_does_not_erase_what_an_earlier_one_filled(tmp_path):
    """Regression: _add_entity sets resource_group_name, _add_entity_from_bpf_metadata does not.

    Row-wise "newest wins" would throw the cgroup name away as soon as the sparse path writes
    last. The DuckDB upsert does not do that, so the conversion must not either.
    """
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le(cmdline="", cgroup="/system.slice/test.service")])
    storage.save_entities([_le(cmdline="/usr/bin/python3 -m foo", cgroup=None)])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    cmdline, cgroup = _query(tmp_path / "out.db", "SELECT cmdline, resource_group_name FROM logical_entity")[0]

    assert cmdline == "/usr/bin/python3 -m foo", "the later, better cmdline wins"
    assert cgroup == "/system.slice/test.service", "the cgroup name filled earlier survives"


def test_stream_directories_map_onto_the_database_table_names(tmp_path):
    """The .sql analyses join on logical_entity, not on the class name logicalentity."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le()])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    tables = {row[0] for row in _query(tmp_path / "out.db", "SHOW TABLES")}
    assert "logical_entity" in tables
    assert "logicalentity" not in tables


def test_bookkeeping_columns_do_not_leak_into_the_tables(tmp_path):
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le()])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    columns = {row[0] for row in _query(tmp_path / "out.db", "DESCRIBE logical_entity")}
    assert "_seq" not in columns
    assert "_entity_type" not in columns


def test_entity_table_carries_every_id_with_its_type(tmp_path):
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le(), Host("h", HostType.SERVER, "Linux")])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(tmp_path / "out.db", "SELECT entity_type FROM entity ORDER BY entity_type")
    assert [row[0] for row in rows] == ["Host", "LogicalEntity"]


def test_cores_of_a_cpu_reach_their_own_table(tmp_path):
    """save_entities unpacks cpu.cores, so they must land in their own stream and table.

    package_id is a string on purpose: discovery builds it with str() (linux/cpu.py:339),
    and both _to_df schemas declare it as pl.String, whatever the type hints say.
    """
    cpu = _cpu_with_cores(2)

    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([cpu])
    storage.close()

    counts = convert(tmp_path / "run", tmp_path / "out.db")

    assert counts.get("core", 0) == len(cpu.cores)


def test_target_columns_match_the_database_schema(tmp_path):
    """Pins the rename map: every DB column must exist after the conversion.

    Without this the conversion drifts silently and the .sql analyses break on a column
    that is simply missing.
    """
    from metrion.data_storage.db_data_storage import SQLDatabaseStorage

    db = SQLDatabaseStorage(f"duckdb:///{tmp_path / 'schema.db'}")
    expected = {column.name for column in db._logical_entity_table.columns}

    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le()])
    storage.close()
    convert(tmp_path / "run", tmp_path / "out.db")

    actual = {row[0] for row in _query(tmp_path / "out.db", "DESCRIBE logical_entity")}
    missing = expected - actual

    assert not missing, f"logical_entity is missing {sorted(missing)} after the conversion"


def test_the_converted_file_answers_the_joins_the_sql_analyses_use(tmp_path):
    """End to end: the point of the conversion is that the shipped .sql files still run.

    Mirrors dram_energy_analysis_by_thread.sql:17 (measurement -> entity on
    physical_entity_id) and the logical_entity join the per-pid analyses do.
    """
    cpu = _cpu_with_cores(2)
    cores = cpu.cores
    le = _le(cmdline="/usr/bin/python3 -m foo")

    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([Host("h", HostType.SERVER, "Linux"), cpu, le])
    storage.save_measurements(
        [
            pl.DataFrame(
                {
                    "id": [1, 2],
                    "metric_id": ["CPU_CORE_PID_WORK"] * 2,
                    "physical_entity_id": [cores[0].id, cores[1].id],
                    "logical_entity_id": [le.id, le.id],
                    "value": [1.0, 2.0],
                    "start_ts": [T, T],
                    "end_ts": [T, T],
                }
            )
        ]
    )
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(
        tmp_path / "out.db",
        """
        SELECT le.cmdline, e.parent_id, sum(m.value)
        FROM measurement AS m
        JOIN entity AS e ON m.physical_entity_id = e.id
        JOIN logical_entity AS le ON m.logical_entity_id = le.id
        GROUP BY le.cmdline, e.parent_id
        """,
    )

    assert len(rows) == 1, f"expected one group, got {rows}"
    cmdline, parent_id, total = rows[0]
    assert cmdline == "/usr/bin/python3 -m foo"
    assert parent_id == cpu.id, "cores must keep their parent CPU in the entity table"
    assert total == 3.0


def test_timestamps_match_the_duckdb_path(tmp_path):
    """The DuckDB path stores naive TIMESTAMP; Parquet carries the time zone along.

    Handing the analyses a TIMESTAMP WITH TIME ZONE column moves every comparison against a
    literal into the session time zone, and reading one back needs pytz, which is not a
    dependency.
    """
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(1)])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    types = dict(_query(tmp_path / "out.db", "SELECT column_name, column_type FROM (DESCRIBE measurement)"))
    assert types["start_ts"] == "TIMESTAMP"
    assert types["end_ts"] == "TIMESTAMP"

    stored = _query(tmp_path / "out.db", "SELECT start_ts FROM measurement")[0][0]
    assert stored == T.replace(tzinfo=None), "the UTC wall clock has to survive unshifted"


def test_the_time_filter_of_the_analyses_finds_the_row(tmp_path):
    """cpu_energy_analysis_by_thread.sql:31 compares start_ts against a naive UTC string."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(1)])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(
        tmp_path / "out.db",
        """
        SELECT count(*) FROM measurement
        WHERE "start_ts" >= '2025-12-31 23:59:00.000000'
          AND "start_ts" <= '2026-01-01 00:01:00.000000'
        """,
    )
    assert rows[0][0] == 1


def test_entity_timestamps_are_normalised_too(tmp_path):
    """_to_df() writes nanosecond timestamps; the DB path stores microsecond TIMESTAMP."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le()])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    types = dict(_query(tmp_path / "out.db", "SELECT column_name, column_type FROM (DESCRIBE logical_entity)"))
    assert types["start_time"] == "TIMESTAMP"
    assert _query(tmp_path / "out.db", "SELECT start_time FROM logical_entity")[0][0] == T.replace(tzinfo=None)


def test_every_table_the_duckdb_path_creates_exists(tmp_path):
    """A component that did not run writes no stream, but its table still has to be there."""
    from metrion.data_storage.db_data_storage import SQLDatabaseStorage

    SQLDatabaseStorage(f"duckdb:///{tmp_path / 'reference.db'}").close()
    expected = {row[0] for row in _query(tmp_path / "reference.db", "SHOW TABLES")}

    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le()])
    storage.close()
    convert(tmp_path / "run", tmp_path / "out.db")

    actual = {row[0] for row in _query(tmp_path / "out.db", "SHOW TABLES")}
    assert not expected - actual, f"missing after the conversion: {sorted(expected - actual)}"


def test_a_join_on_a_component_that_did_not_run_stays_empty(tmp_path):
    """Without this the gpu analysis dies with a catalog error on a machine without a GPU."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(3)])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(
        tmp_path / "out.db",
        "SELECT count(*) FROM measurement AS m JOIN gpu AS g ON m.physical_entity_id = g.id",
    )
    assert rows[0][0] == 0


def test_converting_twice_into_the_same_file_works(tmp_path):
    """The file is derived from the streams, so a repeated conversion has to rebuild it."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(4)])
    storage.save_entities([_le()])
    storage.close()

    first = convert(tmp_path / "run", tmp_path / "out.db")
    second = convert(tmp_path / "run", tmp_path / "out.db")

    assert first == second
    assert _query(tmp_path / "out.db", "SELECT count(*) FROM measurement")[0][0] == 4


def test_an_empty_string_arriving_later_does_not_erase_a_filled_value(tmp_path):
    """The other half of the DuckDB upsert: coalesce(nullif(excluded.col, ''), table.col).

    max_by skips NULLs on its own, so only this case pins the FILTER in the projection. The
    registry does write empty strings: a later, sparser version of an entity carries "" where
    an earlier one had the full value.
    """
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_entities([_le(cmdline="/usr/bin/python3 -m foo")])
    storage.save_entities([_le(cmdline="")])
    storage.close()

    convert(tmp_path / "run", tmp_path / "out.db")

    rows = _query(tmp_path / "out.db", "SELECT cmdline FROM logical_entity")
    assert rows[0][0] == "/usr/bin/python3 -m foo"


def test_a_run_directory_is_converted_on_first_use(tmp_path):
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(3)])
    storage.close()

    database = ensure_duckdb(tmp_path / "run")

    assert database == tmp_path / "run" / "monitor.db"
    assert _query(database, "SELECT count(*) FROM measurement")[0][0] == 3


def test_the_converted_file_is_reused_while_it_is_current(tmp_path):
    """Analysing the same run twice must not rebuild the database twice."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(3)])
    storage.close()

    first = ensure_duckdb(tmp_path / "run")
    stamp = first.stat().st_mtime_ns

    assert ensure_duckdb(tmp_path / "run").stat().st_mtime_ns == stamp


def test_a_newer_stream_makes_the_conversion_run_again(tmp_path):
    """A run that was still being written when it was first analysed must not stay stale."""
    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(3)])
    storage.close()

    database = ensure_duckdb(tmp_path / "run")
    stamp = database.stat().st_mtime_ns

    later = time.time() + 10
    for part in (tmp_path / "run" / "measurements").glob("*.parquet"):
        os.utime(part, (later, later))

    assert ensure_duckdb(tmp_path / "run").stat().st_mtime_ns != stamp


def test_the_analysis_accepts_a_run_directory(tmp_path):
    """prepare_data is the one gate every analysis goes through, so the hook sits there."""
    from metrion.energy_attribution import energy_attribution_cli

    storage = ParquetDataStorage(str(tmp_path / "run"))
    storage.save_measurements([_mdf(2)])
    storage.close()

    seen = {}

    def fake_connect(db_path):
        seen["db_path"] = db_path
        return None

    original = energy_attribution_cli.connect_to_db
    energy_attribution_cli.connect_to_db = fake_connect
    try:
        energy_attribution_cli.prepare_data(
            tmp_path / "run", "python", datetime(2026, 1, 1), datetime(2026, 1, 2), unit="J"
        )
    finally:
        energy_attribution_cli.connect_to_db = original

    assert seen["db_path"] == tmp_path / "run" / "monitor.db", "the directory has to become a file"
    assert _query(seen["db_path"], "SELECT count(*) FROM measurement")[0][0] == 2
