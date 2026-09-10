from unittest.mock import patch

import polars as pl
import pytest

from metrion.data_collector import DataCollector
from metrion.data_record import DataRecord
from metrion.entity.host import Host, HostType
from metrion.util import read_temp_ipc


@pytest.fixture
def data_collector(tmp_path):
    """A collector whose saver process is never started, with a writer that flushes quickly."""
    with patch.object(DataCollector, "_start_collection_process"):
        collector = DataCollector("duckdb", str(tmp_path / "test.db"), flush_interval=0.2)

    yield collector

    collector.ipc_writer.close()
    collector._entities_queue.cancel_join_thread()
    collector._measurements_queue.cancel_join_thread()


def test_add_entities_pushes_to_queue(data_collector):
    host = Host("test-host", HostType.SERVER, "Linux")

    data_collector.add(DataRecord([host], []))

    item = data_collector._entities_queue.get(timeout=1)
    assert len(item) == 1
    assert item[0].id == host.id


def test_measurements_reach_the_queue_as_a_file_path(data_collector):
    """The queue carries the path of an IPC file, never the rows themselves."""
    data_collector.add(DataRecord([], [pl.DataFrame({"a": range(100)})]))

    path = data_collector._measurements_queue.get(timeout=5)

    assert read_temp_ipc(path).height == 100


def test_communication_measurements_are_batched_separately(data_collector):
    """src_endpoint/dst_endpoint frames go into their own batch and their own file."""
    plain = pl.DataFrame({"a": range(5)})
    comm = pl.DataFrame({"id": [1], "src_endpoint": ["a"], "dst_endpoint": ["b"]})

    data_collector.add(DataRecord([], [plain, comm]))

    frames = [read_temp_ipc(data_collector._measurements_queue.get(timeout=5)) for _ in range(2)]

    assert sorted(frame.height for frame in frames) == [1, 5]
    assert sum("src_endpoint" in frame.columns for frame in frames) == 1


def test_add_does_not_accumulate_rows_in_the_collector(data_collector):
    """add() hands everything to the writer, so the collector's own buffer stays empty."""
    data_collector.add(DataRecord([], [pl.DataFrame({"a": range(100)})]))

    assert data_collector._measurement_buffer == []
    assert data_collector._buffer_row_count == 0
