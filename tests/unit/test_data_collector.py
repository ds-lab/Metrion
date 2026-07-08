import multiprocessing
from unittest.mock import patch

import polars as pl
import pytest

from metrion.data_collector import DataCollector
from metrion.data_record import DataRecord
from metrion.entity.host import Host, HostType


@pytest.fixture
def mock_storage_factory():
    with patch("metrion.data_collector.DataStorageFactory") as mock:
        yield mock


@pytest.fixture
def data_collector(tmp_path):
    # Patch Process so it doesn't start
    with patch("metrion.data_collector.Process"):
        dc = DataCollector("sqlite", str(tmp_path / "test.db"))
        yield dc
        # No need to stop real process


def test_add_entities_pushes_to_queue(data_collector):
    host = Host("test-host", HostType.SERVER, "Linux")
    record = DataRecord([host], [])

    data_collector.add(record)

    # Use get with timeout instead of empty()
    item = data_collector._entities_queue.get(timeout=1)
    assert len(item) == 1
    assert item[0].id == host.id


def test_add_measurements_buffers_and_chunks(data_collector):
    # Create a dataframe smaller than buffer
    df_small = pl.DataFrame({"a": range(100)})
    record_small = DataRecord([], [df_small])

    with patch("metrion.data_collector.write_temp_ipc") as mock_write:
        mock_write.return_value = "/tmp/fake_path.ipc"

        # 1. Add small chunk - should buffer, NOT queue
        data_collector.add(record_small)

        # Verify queue is empty (use small timeout)
        with pytest.raises(multiprocessing.queues.Empty):
            data_collector._measurements_queue.get(timeout=0.1)

        assert data_collector._buffer_row_count == 100
        assert not mock_write.called

        # 2. Force flush or add enough to trigger
        # Let's just manually flush for testing small data
        data_collector._flush_measurements()

        assert mock_write.called
        assert data_collector._measurements_queue.get(timeout=1) == "/tmp/fake_path.ipc"
        assert data_collector._buffer_row_count == 0


def test_add_measurements_auto_flush(data_collector):
    # Test that it auto-flushes when limit is hit
    large_df = pl.DataFrame({"a": range(50000)})
    record_large = DataRecord([], [large_df])

    with patch("metrion.data_collector.write_temp_ipc") as mock_write:
        mock_write.return_value = "/tmp/large.ipc"

        data_collector.add(record_large)

        assert mock_write.called
        assert data_collector._measurements_queue.get(timeout=1) == "/tmp/large.ipc"
