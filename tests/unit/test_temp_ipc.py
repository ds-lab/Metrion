import tempfile
from pathlib import Path

import polars as pl

from metrion import util


def test_ipc_files_go_to_shared_memory_when_it_has_room(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "_SHM_DIR", tmp_path)
    monkeypatch.setattr(util, "_SHM_MIN_FREE_BYTES", 1)

    path = util.write_temp_ipc(pl.DataFrame({"a": [1, 2]}))

    assert Path(path).parent == tmp_path
    assert util.read_temp_ipc(path)["a"].to_list() == [1, 2]
    assert not Path(path).exists()


def test_ipc_files_fall_back_to_the_temp_dir_when_shared_memory_is_small(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "_SHM_DIR", tmp_path)
    monkeypatch.setattr(util, "_SHM_MIN_FREE_BYTES", 1 << 62)

    assert util._temp_dir() == Path(tempfile.gettempdir())


def test_ipc_files_fall_back_to_the_temp_dir_without_shared_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(util, "_SHM_DIR", tmp_path / "missing")

    assert util._temp_dir() == Path(tempfile.gettempdir())
