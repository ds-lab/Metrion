import polars as pl

from metrion.measurement import Measurement
from metrion.util import _add_id_to_measurement_df, _generate_id


def _use_generator(monkeypatch, generator) -> None:
    monkeypatch.setattr(
        Measurement,
        "get_snowflake_generator",
        classmethod(lambda cls: generator),
    )


class _GeneratorWithCapacityGaps:
    """Deterministically simulates temporary Snowflake saturation."""

    def __init__(self, start: int = 1_000_000) -> None:
        self._next_id = start
        self._calls = 0

    def __next__(self) -> int | None:
        self._calls += 1

        if self._calls % 4097 == 0:
            return None

        generated_id = self._next_id
        self._next_id += 1
        return generated_id


def test_generate_id_retries_temporary_none_results(monkeypatch) -> None:
    _use_generator(monkeypatch, iter([101, None, None, 102, None, 103]))

    ids = _generate_id(3)

    assert ids.dtype == pl.Int64
    assert ids.to_list() == [101, 102, 103]
    assert ids.null_count() == 0


def test_generate_id_handles_production_sized_batch(monkeypatch) -> None:
    _use_generator(monkeypatch, _GeneratorWithCapacityGaps())

    requested_ids = 829_736
    ids = _generate_id(requested_ids)

    assert len(ids) == requested_ids
    assert ids.dtype == pl.Int64
    assert ids.null_count() == 0
    assert ids.n_unique() == requested_ids


def test_add_id_repairs_null_typed_id_column(monkeypatch) -> None:
    _use_generator(monkeypatch, iter([201, 202]))

    frame = pl.DataFrame(
        {
            "row": [0, 1],
            "id": [None, None],
        }
    )

    assert frame.schema["id"] == pl.Null

    result = _add_id_to_measurement_df(frame)

    assert result.schema["id"] == pl.Int64
    assert result["id"].to_list() == [201, 202]
    assert result["id"].null_count() == 0


def test_add_id_preserves_existing_ids(monkeypatch) -> None:
    _use_generator(monkeypatch, iter([456]))

    frame = pl.DataFrame(
        {
            "row": [0, 1],
            "id": pl.Series("id", [123, None], dtype=pl.Int64),
        }
    )

    result = _add_id_to_measurement_df(frame).sort("row")

    assert result["id"].to_list() == [123, 456]
    assert result["id"].null_count() == 0


def test_add_id_to_empty_frame_creates_int64_column(monkeypatch) -> None:
    _use_generator(monkeypatch, iter([]))

    frame = pl.DataFrame(
        {
            "value": pl.Series("value", [], dtype=pl.Float64),
        }
    )

    result = _add_id_to_measurement_df(frame)

    assert result.height == 0
    assert result.schema["id"] == pl.Int64
    assert result["id"].null_count() == 0
