"""Unit tests for GPU energy attribution model."""

import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.energy_attribution import (
    DyncamicEnergyAttributionModel,
    GPUEnergyAttributionModel,
)
from metrion.entity.gpu import GPU
from metrion.metric import (
    GPU_DEVICE_PID_DYNAMIC_ENERGY,
    GPU_DEVICE_PID_IDLE_ENERGY,
    GPU_DEVICE_PID_TOTAL_ENERGY,
)


class TestGPUEnergyAttributionModel:
    """Tests for GPUEnergyAttributionModel."""

    @pytest.fixture
    def gpu(self) -> GPU:
        """A100 GPU with 50W idle power."""
        return GPU(
            manufacturer="NVIDIA",
            model="A100-SXM4-40GB",
            serial_number="SN-A100",
            uuid="GPU-uuid-a100",
            pci_id="0000:00:01.0",
            idle_power=50.0,
            max_power=400.0,
            compute_capability_major=8,
            compute_capability_minor=0,
        )

    @pytest.fixture
    def gpu_df(self, gpu: GPU) -> pl.DataFrame:
        """GPU DataFrame for attribution."""
        return gpu._to_df()

    @pytest.fixture
    def base_time(self) -> dt.datetime:
        """Base timestamp for measurements."""
        return dt.datetime(2026, 1, 8, 10, 0, 0)

    @pytest.fixture
    def energy_measurements(self, base_time: dt.datetime) -> pl.DataFrame:
        """Energy counter measurements: 100J consumed over 10 seconds."""
        return pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [1000.0, 1100.0],  # 100J delta
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

    @pytest.fixture
    def utilization_measurements(self, base_time: dt.datetime) -> pl.DataFrame:
        """Process utilization: PID1=60%, PID2=40%."""
        return pl.DataFrame(
            {
                "metric_id": [
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                ],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": ["pid-1234", "pid-5678"],
                "value": [60.0, 40.0],  # 60% and 40% utilization
                "start_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
                "end_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
            }
        )

    def test_empty_measurements_returns_empty(self, gpu_df: pl.DataFrame):
        """Empty measurements return empty results."""
        model = GPUEnergyAttributionModel()
        result = model.attribute_gpu_energy(
            measurement_df=pl.DataFrame(),
            gpu_df=gpu_df,
        )
        assert all(df.is_empty() for df in result)

    def test_empty_gpu_df_returns_empty(
        self, energy_measurements: pl.DataFrame, utilization_measurements: pl.DataFrame
    ):
        """Empty GPU DataFrame returns empty results."""
        model = GPUEnergyAttributionModel()
        measurements = pl.concat([energy_measurements, utilization_measurements])
        result = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=pl.DataFrame(),
        )
        assert all(df.is_empty() for df in result)

    def test_energy_intervals_calculated(
        self,
        gpu_df: pl.DataFrame,
        energy_measurements: pl.DataFrame,
        base_time: dt.datetime,
    ):
        """Energy intervals correctly calculated from counter deltas."""
        model = GPUEnergyAttributionModel()
        intervals = model._calculate_energy_intervals(
            measurement_df=energy_measurements,
            gpu_df=gpu_df,
        )

        assert intervals.height == 1
        assert intervals["energy_j"][0] == 100.0  # 1100 - 1000
        assert intervals["interval_seconds"][0] == 10.0

    def test_idle_energy_capped_at_measured_energy(
        self,
        gpu_df: pl.DataFrame,
        energy_measurements: pl.DataFrame,
    ):
        """Idle energy = idle_power × interval_duration, never more than the GPU used."""
        model = GPUEnergyAttributionModel()
        intervals = model._calculate_energy_intervals(
            measurement_df=energy_measurements,
            gpu_df=gpu_df,
        )

        # 50W idle × 10s would be 500J, but only 100J were measured
        assert intervals["idle_energy_j"][0] == 100.0

    def test_dynamic_energy_calculated(
        self,
        gpu_df: pl.DataFrame,
        energy_measurements: pl.DataFrame,
    ):
        """Dynamic energy = total - idle, and idle never exceeds total."""
        model = GPUEnergyAttributionModel()
        intervals = model._calculate_energy_intervals(
            measurement_df=energy_measurements,
            gpu_df=gpu_df,
        )

        # 500J idle claimed for 100J measured, so the cap leaves nothing dynamic
        assert intervals["dynamic_energy_j"][0] == 0.0

    def test_idle_power_above_measured_energy_keeps_the_sum(
        self,
        gpu_df: pl.DataFrame,
        energy_measurements: pl.DataFrame,
        utilization_measurements: pl.DataFrame,
    ):
        """The attributed energy stays the measured energy even when idle_power is too high."""
        model = GPUEnergyAttributionModel()
        total_df, _, dynamic_df = model.attribute_gpu_energy(
            measurement_df=pl.concat([energy_measurements, utilization_measurements]),
            gpu_df=gpu_df,
        )

        assert (dynamic_df["value"] >= 0.0).all()
        assert total_df["value"].sum() == pytest.approx(100.0)

    def test_attribution_by_utilization_ratio(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Energy attributed proportional to SM utilization."""
        # Create measurements with lower idle power so dynamic > 0
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [1000.0, 2000.0],  # 1000J delta
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        util_df = pl.DataFrame(
            {
                "metric_id": [
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                ],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": ["pid-1234", "pid-5678"],
                "value": [75.0, 25.0],  # 75% and 25% (3:1 ratio)
                "start_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
                "end_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
            }
        )

        measurements = pl.concat([energy_df, util_df])
        model = GPUEnergyAttributionModel()
        total_df, idle_df, dynamic_df = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=gpu_df,
        )

        # 1000J total - 500J idle = 500J dynamic
        # PID1 gets 75% = 375J, PID2 gets 25% = 125J
        pid1_dynamic = dynamic_df.filter(pl.col("logical_entity_id") == "pid-1234")["value"][0]
        pid2_dynamic = dynamic_df.filter(pl.col("logical_entity_id") == "pid-5678")["value"][0]

        assert pid1_dynamic == pytest.approx(375.0, rel=0.01)
        assert pid2_dynamic == pytest.approx(125.0, rel=0.01)

    def test_idle_energy_split_equally(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Idle energy split equally among active processes."""
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [1000.0, 2000.0],
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        util_df = pl.DataFrame(
            {
                "metric_id": [
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                    "GPU_DEVICE_PID_COMPUTE_UTILIZATION",
                ],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": ["pid-1234", "pid-5678"],
                "value": [75.0, 25.0],
                "start_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
                "end_ts": [
                    base_time + dt.timedelta(seconds=5),
                    base_time + dt.timedelta(seconds=5),
                ],
            }
        )

        measurements = pl.concat([energy_df, util_df])
        model = GPUEnergyAttributionModel()
        _, idle_df, _ = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=gpu_df,
        )

        # 500J idle split between 2 processes = 250J each
        pid1_idle = idle_df.filter(pl.col("logical_entity_id") == "pid-1234")["value"][0]
        pid2_idle = idle_df.filter(pl.col("logical_entity_id") == "pid-5678")["value"][0]

        assert pid1_idle == pytest.approx(250.0, rel=0.01)
        assert pid2_idle == pytest.approx(250.0, rel=0.01)

    def test_total_energy_is_sum(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Total energy = dynamic + idle for each process."""
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [1000.0, 2000.0],
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        util_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_PID_COMPUTE_UTILIZATION"],
                "physical_entity_id": ["GPU-uuid-a100"],
                "logical_entity_id": ["pid-1234"],
                "value": [100.0],  # Single process, 100%
                "start_ts": [base_time + dt.timedelta(seconds=5)],
                "end_ts": [base_time + dt.timedelta(seconds=5)],
            }
        )

        measurements = pl.concat([energy_df, util_df])
        model = GPUEnergyAttributionModel()
        total_df, idle_df, dynamic_df = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=gpu_df,
        )

        total_val = total_df["value"][0]
        idle_val = idle_df["value"][0]
        dynamic_val = dynamic_df["value"][0]

        assert total_val == pytest.approx(idle_val + dynamic_val, rel=0.01)

    def test_output_metrics_correct(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Output DataFrames have correct metric IDs."""
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [1000.0, 2000.0],
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        util_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_PID_COMPUTE_UTILIZATION"],
                "physical_entity_id": ["GPU-uuid-a100"],
                "logical_entity_id": ["pid-1234"],
                "value": [100.0],
                "start_ts": [base_time + dt.timedelta(seconds=5)],
                "end_ts": [base_time + dt.timedelta(seconds=5)],
            }
        )

        measurements = pl.concat([energy_df, util_df])
        model = GPUEnergyAttributionModel()
        total_df, idle_df, dynamic_df = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=gpu_df,
        )

        assert total_df["metric_id"][0] == GPU_DEVICE_PID_TOTAL_ENERGY.name
        assert idle_df["metric_id"][0] == GPU_DEVICE_PID_IDLE_ENERGY.name
        assert dynamic_df["metric_id"][0] == GPU_DEVICE_PID_DYNAMIC_ENERGY.name

    def test_wrap_around_handled(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Energy counter wrap-around is handled correctly."""
        # Simulate wrap-around: second value < first value
        # Wrap value is (2**64 - 1) / 1000 ≈ 1.84e16
        wrap_value = (2**64 - 1) / 1000.0
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-uuid-a100", "GPU-uuid-a100"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [wrap_value - 1000.0, 500.0],  # Wrap-around: end near max, start after wrap
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        model = GPUEnergyAttributionModel()
        intervals = model._calculate_energy_intervals(
            measurement_df=energy_df,
            gpu_df=gpu_df,
        )

        # Expected: wrap_value - (wrap_value - 1000) + 500 = 1500J
        assert intervals["energy_j"][0] == pytest.approx(1500.0, rel=0.01)


class TestDynamicModelGPUIntegration:
    """Tests for GPU integration in DyncamicEnergyAttributionModel."""

    @pytest.fixture
    def gpu(self) -> GPU:
        """Test GPU entity."""
        return GPU(
            manufacturer="NVIDIA",
            model="A100",
            serial_number="SN",
            uuid="GPU-test",
            pci_id="0000:00:01.0",
            idle_power=50.0,
            max_power=400.0,
            compute_capability_major=8,
            compute_capability_minor=0,
        )

    def test_update_gpu_df(self, gpu: GPU):
        """GPU entities are added to internal buffer."""
        model = DyncamicEnergyAttributionModel()
        model.update_gpu_df([gpu])

        assert not model._gpu_df.is_empty()
        assert model._gpu_df["id"][0] == gpu.id

    def test_update_entities_includes_gpu(self, gpu: GPU):
        """update_entites correctly handles GPU entities."""
        model = DyncamicEnergyAttributionModel()
        model.update_entites([gpu])

        assert not model._gpu_df.is_empty()
        assert model._gpu_df["id"][0] == gpu.id

    def test_attribute_energy_returns_gpu_result(self, gpu: GPU):
        """attribute_energy returns GPU result tuple."""
        model = DyncamicEnergyAttributionModel()
        model.update_gpu_df([gpu])

        result = model.attribute_energy(force=True)

        # Should return 3 tuples: pkg, dram, gpu
        assert len(result) == 3
        # Each tuple has 3 DataFrames
        assert all(len(r) == 3 for r in result)


class TestCrossBatchContinuity:
    """Tests for cross-batch energy continuity."""

    @pytest.fixture
    def gpu(self) -> GPU:
        """Test GPU entity."""
        return GPU(
            manufacturer="NVIDIA",
            model="A100",
            serial_number="SN",
            uuid="GPU-test",
            pci_id="0000:00:01.0",
            idle_power=50.0,
            max_power=400.0,
            compute_capability_major=8,
            compute_capability_minor=0,
        )

    def test_last_energy_sample_cached(self, gpu: GPU):
        """Last energy sample is cached for cross-batch continuity."""
        model = DyncamicEnergyAttributionModel(cutoff=0.0)
        model.update_gpu_df([gpu])

        base_time = dt.datetime(2026, 1, 8, 10, 0, 0)
        measurements = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-test", "GPU-test"],
                "logical_entity_id": [None, None],
                "value": [1000.0, 2000.0],
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        model.update_measurements(measurements)
        model._get_relevant_measurements(force=True)

        # Check that last energy sample was cached
        assert len(model._last_energy_samples) > 0
        # Cache key is (metric, entity_id)
        key = ("GPU_DEVICE_TOTAL_ENERGY", "GPU-test")
        assert key in model._last_energy_samples

    def test_cached_sample_injected_for_next_batch(self, gpu: GPU):
        """Cached energy sample is injected into buffer for next batch."""
        model = DyncamicEnergyAttributionModel(cutoff=0.0)
        model.update_gpu_df([gpu])

        base_time = dt.datetime(2026, 1, 8, 10, 0, 0)
        batch1 = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-test", "GPU-test"],
                "logical_entity_id": [None, None],
                "value": [1000.0, 2000.0],
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        model.update_measurements(batch1)
        model._get_relevant_measurements(force=True)

        # After first batch, buffer should have cached sample
        buffer_data = model._measurements_df
        assert not buffer_data.is_empty()
        # Cached sample value should be 2000 (last sample)
        gpu_samples = buffer_data.filter(pl.col("metric_id") == "GPU_DEVICE_TOTAL_ENERGY")
        assert gpu_samples["value"][0] == 2000.0


class TestTimeWeightedIdleSplit:
    """Tests for time-weighted GPU idle energy distribution."""

    @pytest.fixture
    def gpu_df(self) -> pl.DataFrame:
        """GPU entity DataFrame."""
        return pl.DataFrame(
            {
                "id": ["GPU-test"],
                "idle_power": [100.0],  # 100W idle
            }
        )

    @pytest.fixture
    def base_time(self) -> dt.datetime:
        """Base timestamp."""
        return dt.datetime(2026, 1, 8, 10, 0, 0)

    def test_idle_energy_weighted_by_presence(self, gpu_df: pl.DataFrame, base_time: dt.datetime):
        """Idle energy is distributed proportional to process presence time."""
        # Energy: 1000J total over 10 seconds, 1000J idle (100W * 10s)
        energy_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_TOTAL_ENERGY", "GPU_DEVICE_TOTAL_ENERGY"],
                "physical_entity_id": ["GPU-test", "GPU-test"],
                "logical_entity_id": pl.Series([None, None], dtype=pl.String),
                "value": [0.0, 2000.0],  # 2000J total
                "start_ts": [base_time, base_time + dt.timedelta(seconds=10)],
                "end_ts": [base_time, base_time + dt.timedelta(seconds=10)],
            }
        )

        # PID1: 3 samples (more presence time)
        # PID2: 1 sample (less presence time)
        # Idle energy should be split 3:1
        util_df = pl.DataFrame(
            {
                "metric_id": ["GPU_DEVICE_PID_COMPUTE_UTILIZATION"] * 4,
                "physical_entity_id": ["GPU-test"] * 4,
                "logical_entity_id": ["pid-1", "pid-1", "pid-1", "pid-2"],
                "value": [25.0, 25.0, 25.0, 25.0],  # Same utilization
                "start_ts": [
                    base_time + dt.timedelta(seconds=2),
                    base_time + dt.timedelta(seconds=4),
                    base_time + dt.timedelta(seconds=6),
                    base_time + dt.timedelta(seconds=5),
                ],
                "end_ts": [
                    base_time + dt.timedelta(seconds=2),
                    base_time + dt.timedelta(seconds=4),
                    base_time + dt.timedelta(seconds=6),
                    base_time + dt.timedelta(seconds=5),
                ],
            }
        )

        measurements = pl.concat([energy_df, util_df])
        model = GPUEnergyAttributionModel()
        _, idle_df, _ = model.attribute_gpu_energy(
            measurement_df=measurements,
            gpu_df=gpu_df,
        )

        # 1000J idle (100W * 10s), split by sample count (3:1)
        pid1_idle = idle_df.filter(pl.col("logical_entity_id") == "pid-1")["value"][0]
        pid2_idle = idle_df.filter(pl.col("logical_entity_id") == "pid-2")["value"][0]

        # PID1 gets 75% of idle (3/4 * 1000 = 750J)
        # PID2 gets 25% of idle (1/4 * 1000 = 250J)
        assert pid1_idle == pytest.approx(750.0, rel=0.01)
        assert pid2_idle == pytest.approx(250.0, rel=0.01)
