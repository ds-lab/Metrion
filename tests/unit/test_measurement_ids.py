import multiprocessing as mp

import polars as pl
import pytest

from metrion.measurement import Measurement
from metrion.util import IpcAsyncWriter, _add_id_to_measurement_df, read_temp_ipc


def test_ids_stay_complete_beyond_one_millisecond():
    """
    Regression: the snowflake generator returns None once its 4096 ids per millisecond are used up.

    Those None values used to end up in the column, and measurement.id is a NOT NULL primary key,
    so a single one rejected the whole batch on insert.
    """
    ids = Measurement.next_ids(100_000)

    assert None not in ids
    assert len(set(ids)) == 100_000


def test_generated_column_is_int64_without_nulls():
    df = _add_id_to_measurement_df(pl.DataFrame({"value": [1.0] * 10_000}))

    assert df["id"].dtype == pl.Int64
    assert df["id"].null_count() == 0
    assert df["id"].n_unique() == 10_000


def test_column_of_only_nulls_is_filled():
    """A frame whose id column is entirely null arrives with dtype Null."""
    df = _add_id_to_measurement_df(pl.DataFrame({"value": [1.0, 2.0], "id": [None, None]}))

    assert df["id"].dtype == pl.Int64
    assert df["id"].null_count() == 0


def test_existing_ids_and_row_order_survive():
    df = _add_id_to_measurement_df(pl.DataFrame({"value": [1.0, 2.0, 3.0], "id": [111, None, 333]}))

    assert df["value"].to_list() == [1.0, 2.0, 3.0]
    assert df["id"][0] == 111
    assert df["id"][2] == 333
    assert df["id"][1] not in (None, 111, 333)


def test_a_full_column_is_left_alone():
    df = _add_id_to_measurement_df(pl.DataFrame({"value": [1.0, 2.0], "id": [111, 222]}))

    assert df["id"].to_list() == [111, 222]


def test_empty_frame():
    df = _add_id_to_measurement_df(pl.DataFrame({"value": [], "id": []}))

    assert df.height == 0
    assert df["id"].dtype == pl.Int64


def test_the_writer_hands_out_a_file_without_null_ids():
    """The path that broke in production: writer assigns ids, writes IPC, storage inserts."""
    out = mp.Queue()
    writer = IpcAsyncWriter(out, max_rows=500_000, max_age_s=0.3, assign_ids=True)
    try:
        writer.enqueue([pl.DataFrame({"value": [1.0] * 20_000})], "measurement")
        written = read_temp_ipc(out.get(timeout=10))
    finally:
        writer.close()
        out.cancel_join_thread()

    assert written["id"].null_count() == 0
    assert written["id"].n_unique() == 20_000


def test_duckdb_rejects_the_whole_batch_on_a_single_null_id():
    """Why the bug was expensive: the insert is all or nothing, so one null lost every row."""
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    con.execute("CREATE TABLE measurement (id BIGINT PRIMARY KEY, value DOUBLE)")
    con.register("tmp_df", pl.DataFrame({"id": [1, None, 3], "value": [1.0, 2.0, 3.0]}))

    with pytest.raises(duckdb.ConstraintException):
        con.execute("INSERT INTO measurement (id, value) SELECT id, value FROM tmp_df")

    assert con.execute("SELECT count(*) FROM measurement").fetchone()[0] == 0
