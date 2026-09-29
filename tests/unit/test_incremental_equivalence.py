import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import polars as pl
import polars.testing as plt
import pytest

from metrion.config_reader import LiveAttributionConfig
from metrion.data_collector import DataSaver
from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
    DyncamicEnergyAttributionModel,
    GPUEnergyAttributionModel,
)
from metrion.energy_attribution.sql_attribution import get_energy_per_thread
from metrion.energy_attribution.wide_result import ENERGY_COLUMNS, add_up, to_wide

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data"

CUTOFF = 15.0
WINDOW = 10.0
CHUNK_SECONDS = [1.0, 5.0, 20.0]
COMPONENTS = ["cpu", "dram", "gpu"]
PARTS = ["total", "idle", "dynamic"]

COLUMNS = ["logical_entity_id", "value", "metric_id", "physical_entity_id", "start_ts", "end_ts"]
SORT_KEYS = ["start_ts", "end_ts", "logical_entity_id", "physical_entity_id"]


def _chunks(measurement_df: pl.DataFrame, seconds: float):
    """The measurements as they would arrive: batches of `seconds`, oldest first."""
    ordered = measurement_df.sort("end_ts")
    step = dt.timedelta(seconds=seconds)
    start, last = ordered["end_ts"].min(), ordered["end_ts"].max()
    while start <= last:
        batch = ordered.filter((pl.col("end_ts") >= start) & (pl.col("end_ts") < start + step))
        if not batch.is_empty():
            yield batch
        start += step


def _feed(measurement_df, seconds, cpu_df=None, core_df=None, dram_df=None, gpu_df=None) -> list:
    """Run the incremental model over the batches and keep every cycle's result."""
    model = DyncamicEnergyAttributionModel(cutoff=CUTOFF)
    # The fixtures already have the shape update_entites() would build, so they go in directly.
    model._cpu_df = cpu_df if cpu_df is not None else pl.DataFrame()
    model._core_df = core_df if core_df is not None else pl.DataFrame()
    model._dram_df = dram_df if dram_df is not None else pl.DataFrame()
    model._gpu_df = gpu_df if gpu_df is not None else pl.DataFrame()

    results = []
    for batch in _chunks(measurement_df, seconds):
        model.update_measurements(batch)
        results.append(model.attribute_energy())
    # The shutdown drain. Without it every component still holds back the interval whose
    # right edge lies past its slowest stream: at 5-s batches 1.7 % of the CPU and 8.4 % of the
    # DRAM energy in this fixture.
    results.append(model.attribute_energy(force=True))
    return results


def _collect(results: list, component: str, part: str) -> pl.DataFrame:
    """One slot of the attribute_energy result, concatenated over all cycles.

    Parameters
    ----------
    component : str
        "cpu", "dram" or "gpu".
    part : str
        "total", "idle" or "dynamic".
    """
    outer = COMPONENTS.index(component)
    inner = PARTS.index(part)
    frames = [batch[outer][inner] for batch in results if not batch[outer][inner].is_empty()]
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _assert_same(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    plt.assert_frame_equal(
        actual.sort(SORT_KEYS).select(COLUMNS),
        expected.sort(SORT_KEYS).select(COLUMNS),
        check_exact=False,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )


@pytest.fixture(scope="module")
def recorded() -> dict[str, pl.DataFrame]:
    """The recorded run: measurements plus the entities they are resolved against."""
    names = ("measurement", "cpu", "core", "dram", "gpu", "entity")
    return {name: pl.read_parquet(FIXTURE_DIR / f"{name}.parquet") for name in names}


@pytest.fixture(scope="module")
def one_pass(recorded) -> dict[str, tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]]:
    """What the batch models produce over the whole run at once."""
    return {
        "cpu": CPUEnergyAttributionModel().attribute_energy(
            measurement_df=recorded["measurement"], cpu_df=recorded["cpu"], core_df=recorded["core"]
        ),
        "dram": DRAMEnergyAttributionModel().attribute_energy(
            measurement_df=recorded["measurement"], core_df=recorded["core"], dram_df=recorded["dram"]
        ),
        "gpu": GPUEnergyAttributionModel().attribute_gpu_energy(
            measurement_df=recorded["measurement"], gpu_df=recorded["gpu"]
        ),
    }


@pytest.fixture(scope="module")
def chunked(recorded) -> dict[float, list]:
    """One incremental run per batch size, shared by every assertion below."""
    return {
        seconds: _feed(
            recorded["measurement"],
            seconds,
            cpu_df=recorded["cpu"],
            core_df=recorded["core"],
            dram_df=recorded["dram"],
            gpu_df=recorded["gpu"],
        )
        for seconds in CHUNK_SECONDS
    }


def test_the_fixture_is_older_than_the_cutoff(recorded):
    """Otherwise the cutoff caps the window bound and the result depends on the wall clock."""
    newest = recorded["measurement"]["end_ts"].max()
    assert newest < dt.datetime.now(dt.UTC).replace(tzinfo=None) - dt.timedelta(seconds=CUTOFF)


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("part", PARTS)
@pytest.mark.parametrize("seconds", CHUNK_SECONDS)
def test_in_chunks_equals_one_pass(chunked, one_pass, component, part, seconds):
    """The claim S4 rests on: feeding the model in pieces does not move any energy."""
    _assert_same(_collect(chunked[seconds], component, part), one_pass[component][PARTS.index(part)])


@pytest.mark.parametrize("component", COMPONENTS)
@pytest.mark.parametrize("seconds", CHUNK_SECONDS)
def test_no_interval_is_attributed_twice(chunked, component, seconds):
    """An interval may only come out of one cycle, otherwise its energy is counted twice."""
    keys = _collect(chunked[seconds], component, "total").select(SORT_KEYS)

    assert keys.height == keys.unique().height


def test_the_live_stream_matches_what_metrion_analyze_computes(recorded):
    """End to end: what the saver writes during the run is what the SQL of `metrion analyze` computes afterwards."""
    saver = DataSaver(
        None, None, None, "duckdb", "unused", LiveAttributionConfig(enabled=True, cutoff=CUTOFF, window=WINDOW), []
    )
    model = saver.dyn_eam
    model._cpu_df, model._core_df, model._dram_df, model._gpu_df = (
        recorded[name] for name in ("cpu", "core", "dram", "gpu")
    )
    saver._energy_writer = MagicMock()
    for batch in _chunks(recorded["measurement"], 5.0):
        saver._run_live_attribution([batch])
    saver._run_live_attribution([], force=True)
    live = add_up([call.args[0] for call in saver._energy_writer.write.call_args_list])

    con = duckdb.connect()
    for name, frame in recorded.items():
        con.register(name, frame)
    measurement = recorded["measurement"]
    sql = to_wide(get_energy_per_thread(con, measurement["start_ts"].min(), measurement["end_ts"].max()), WINDOW)

    # A thread with nothing on one side counts as 0 J there. Only an absolute tolerance: the two sides
    # add up in a different order and lie 6e-5 J apart, while a slice that arrived late moves up to
    # 2e-2 J, which a relative tolerance hides in a busy window.
    compared = live.join(sql, on=["window_start", "window_end", "logical_entity_id"], how="full", coalesce=True)
    plt.assert_frame_equal(
        compared.select(pl.col(ENERGY_COLUMNS).fill_null(0.0)),
        compared.select(pl.col(f"{column}_right").fill_null(0.0).alias(column) for column in ENERGY_COLUMNS),
        check_exact=False,
        rel_tol=0.0,
        abs_tol=1e-3,
    )
