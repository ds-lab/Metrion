"""Unit tests for GPU power and energy attribution metrics."""

from metrion.entity.gpu import GPU
from metrion.metric import (
    GPU_DEVICE_PID_DYNAMIC_ENERGY,
    GPU_DEVICE_PID_IDLE_ENERGY,
    GPU_DEVICE_PID_TOTAL_ENERGY,
    GPU_DEVICE_TOTAL_POWER_AVERAGE,
    GPU_DEVICE_TOTAL_POWER_INSTANT,
    Metric,
)


class TestGPUPowerMetrics:
    """Tests for GPU power metric definitions."""

    def test_instant_power_metric_exists(self):
        """GPU_DEVICE_TOTAL_POWER_INSTANT is registered."""
        metric = Metric.get("GPU_DEVICE_TOTAL_POWER_INSTANT")
        assert metric is not None
        assert metric.unit == "W"

    def test_average_power_metric_exists(self):
        """GPU_DEVICE_TOTAL_POWER_AVERAGE is registered."""
        metric = Metric.get("GPU_DEVICE_TOTAL_POWER_AVERAGE")
        assert metric is not None
        assert metric.unit == "W"

    def test_power_metrics_applicable_to_gpu(self):
        """Power metrics should be applicable to GPU entity type."""
        assert GPU_DEVICE_TOTAL_POWER_INSTANT.is_applicable(GPU) is True
        assert GPU_DEVICE_TOTAL_POWER_AVERAGE.is_applicable(GPU) is True


class TestGPUEnergyAttributionMetrics:
    """Tests for GPU per-process energy attribution metrics."""

    def test_pid_total_energy_metric_exists(self):
        """GPU_DEVICE_PID_TOTAL_ENERGY is registered."""
        metric = Metric.get("GPU_DEVICE_PID_TOTAL_ENERGY")
        assert metric is not None
        assert metric.unit == "J"

    def test_pid_idle_energy_metric_exists(self):
        """GPU_DEVICE_PID_IDLE_ENERGY is registered."""
        metric = Metric.get("GPU_DEVICE_PID_IDLE_ENERGY")
        assert metric is not None
        assert metric.unit == "J"

    def test_pid_dynamic_energy_metric_exists(self):
        """GPU_DEVICE_PID_DYNAMIC_ENERGY is registered."""
        metric = Metric.get("GPU_DEVICE_PID_DYNAMIC_ENERGY")
        assert metric is not None
        assert metric.unit == "J"

    def test_energy_metrics_applicable_to_gpu(self):
        """Energy attribution metrics should be applicable to GPU."""
        assert GPU_DEVICE_PID_TOTAL_ENERGY.is_applicable(GPU) is True
        assert GPU_DEVICE_PID_IDLE_ENERGY.is_applicable(GPU) is True
        assert GPU_DEVICE_PID_DYNAMIC_ENERGY.is_applicable(GPU) is True
