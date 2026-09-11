import multiprocessing as mp
from pathlib import Path

import polars as pl
import pytest

from metrion.config_reader.config_schema import LiveAttributionConfig, StorageConfig
from metrion.data_storage.data_storage_factory import DataStorageFactory
from metrion.data_storage.parquet_data_storage import ParquetDataStorage


def test_factory_creates_parquet_storage(tmp_path):
    storage = DataStorageFactory().create_data_storage("parquet", str(tmp_path / "run"))
    try:
        assert isinstance(storage, ParquetDataStorage)
        assert (tmp_path / "run").is_dir()
    finally:
        storage.close()


def test_factory_still_creates_duckdb(tmp_path):
    from metrion.data_storage.db_data_storage import SQLDatabaseStorage

    storage = DataStorageFactory().create_data_storage("duckdb", f"duckdb:///{tmp_path / 'x.db'}")
    try:
        assert isinstance(storage, SQLDatabaseStorage)
    finally:
        storage.close()


def test_factory_rejects_an_unknown_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown storage type"):
        DataStorageFactory().create_data_storage("mongodb", str(tmp_path))


def test_parquet_target_is_a_directory_path():
    """The database backends want a connection string, parquet wants a plain path."""
    target = StorageConfig(type="parquet", path="./runs/x").get_target(Path("/base"))
    assert target == "/base/runs/x"
    assert "://" not in target


def test_duckdb_target_is_a_connection_string():
    assert StorageConfig(type="duckdb", path="./monitor.db").get_target(Path("/base")) == "duckdb:////base/monitor.db"


def test_absolute_parquet_path_is_kept():
    assert StorageConfig(type="parquet", path="/srv/runs/x").get_target(Path("/base")) == "/srv/runs/x"


def test_parquet_with_a_database_path_warns():
    """The likely mistake is switching type to parquet and leaving path at ./monitor.db."""
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        StorageConfig(type="parquet", path="./monitor.db")
    finally:
        logger.remove(sink_id)

    assert any("names a database file" in message for message in messages)


def test_duckdb_with_a_database_path_does_not_warn():
    from loguru import logger

    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        StorageConfig(type="duckdb", path="./monitor.db")
    finally:
        logger.remove(sink_id)

    assert messages == []


def test_the_saver_writes_parquet_when_the_config_says_so(tmp_path):
    """The whole path in one test: config -> factory -> saver -> files on disk."""
    dc = pytest.importorskip("metrion.data_collector", reason="needs bcc")
    from metrion.entity.host import Host, HostType

    config = StorageConfig(type="parquet", path=str(tmp_path / "run"))

    ctx = mp.get_context("spawn")
    saver = dc.DataSaver(
        ctx.Queue(), ctx.Queue(), ctx.Event(), config.type, config.get_target(), LiveAttributionConfig(), []
    )

    saver._entities_queue.put([Host("test-host", HostType.SERVER, "Linux")])
    saver._entities_queue.put(dc.DataCollector.SENT_E)
    saver._measurements_queue.put(dc.DataCollector.SENT_M)
    saver._stop_event.set()

    saver.store_data()

    written = pl.read_parquet(tmp_path / "run" / "entities" / "host" / "*.parquet")
    assert written["name"].to_list() == ["test-host"]
