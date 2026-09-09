import datetime as dt
from pathlib import Path

import polars as pl
import polars.testing as plt
import pytest

from metrion.energy_attribution.energy_attribution import (
    CPUEnergyAttributionModel,
    DRAMEnergyAttributionModel,
    DyncamicEnergyAttributionModel,
    GPUEnergyAttributionModel,
)
from metrion.entity.gpu import GPU
from metrion.metric import GPU_DEVICE_PID_COMPUTE_UTILIZATION, GPU_DEVICE_TOTAL_ENERGY

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "data"

CUTOFF = 15.0
CHUNK_SECONDS = [1.0, 5.0, 20.0]
COMPONENTS = ["cpu", "dram", "gpu"]
PARTS = ["total", "idle", "dynamic"]

COLUMNS = ["logical_entity_id", "value", "metric_id", "physical_entity_id", "start_ts", "end_ts"]
SORT_KEYS = ["start_ts", "end_ts", "logical_entity_id", "physical_entity_id"]

GPU_ID = "GPU-uuid-a100"


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
    # right edge lies past its slowest stream: 4.2 % of the CPU energy in this fixture.
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


def _gpu_entity() -> GPU:
    return GPU(
        manufacturer="NVIDIA",
        model="A100-SXM4-40GB",
        serial_number="SN-A100",
        uuid=GPU_ID,
        pci_id="0000:00:01.0",
        idle_power=50.0,
        max_power=400.0,
        compute_capability_major=8,
        compute_capability_minor=0,
    )


def _gpu_measurements(start: dt.datetime, seconds: float, schema: pl.Schema) -> pl.DataFrame:
    """A GPU stream on the given timeline: counter once a second, utilization every 500 ms.

    measurement.parquet holds no GPU metrics, so the GPU path needs rows of its own. They ride
    on the recorded timeline rather than on a separate one, because the incremental model only
    exercises its bookkeeping when the components run at different speeds: whichever component
    is the slowest of all never needs a lower window bound.
    """
    rows = []
    for step in range(int(seconds) + 1):
        ts = start + dt.timedelta(seconds=step)
        rows.append(
            {
                "metric_id": GPU_DEVICE_TOTAL_ENERGY.name,
                "logical_entity_id": None,
                "value": 1000.0 + 90.0 * step,
                "start_ts": ts,
                "end_ts": ts,
            }
        )
    for step in range(int(seconds * 2)):
        ts = start + dt.timedelta(seconds=0.3 + 0.5 * step)
        for pid, base in (("pid-a", 60.0), ("pid-b", 25.0), ("pid-c", 10.0)):
            rows.append(
                {
                    "metric_id": GPU_DEVICE_PID_COMPUTE_UTILIZATION.name,
                    "logical_entity_id": pid,
                    "value": base + (step % 5) * 3.0,
                    "start_ts": ts,
                    "end_ts": ts,
                }
            )

    return (
        pl.DataFrame(rows, schema_overrides={"logical_entity_id": pl.String})
        .with_columns(id=pl.lit(None, pl.Int64), physical_entity_id=pl.lit(GPU_ID))
        .cast({"value": pl.Float32})
        .select(schema.names())
    )


@pytest.fixture(scope="module")
def recorded() -> dict[str, pl.DataFrame]:
    """The recorded run: measurements plus the entities they are resolved against."""
    frames = {name: pl.read_parquet(FIXTURE_DIR / f"{name}.parquet") for name in ("measurement", "cpu", "core", "dram")}
    measurement = frames["measurement"]
    span = (measurement["end_ts"].max() - measurement["start_ts"].min()).total_seconds()
    frames["measurement"] = pl.concat(
        [measurement, _gpu_measurements(measurement["start_ts"].min(), span, measurement.schema)]
    )
    frames["gpu"] = _gpu_entity()._to_df()
    return frames


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
