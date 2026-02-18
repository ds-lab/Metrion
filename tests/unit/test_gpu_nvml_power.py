"""Unit tests for GPU power collection in NVML data source."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from metrion.entity.gpu import GPU
from metrion.metric import (
    GPU_DEVICE_TOTAL_POWER_AVERAGE,
    GPU_DEVICE_TOTAL_POWER_INSTANT,
)
from metrion.monitoring.data_source.gpu.nvml import NVMLDataSource


class TestGPUPowerCollection:
    """Tests for _read_device_power method in NVMLDataSource."""

    @pytest.fixture
    def gpu_ampere(self) -> GPU:
        """Ampere GPU (CC 8.6) - uses averaged power API."""
        return GPU(
            manufacturer="NVIDIA",
            model="A10",
            serial_number="SN-A10",
            uuid="GPU-uuid-a10",
            pci_id="0000:00:01.0",
            idle_power=25.0,
            max_power=150.0,
            compute_capability_major=8,
            compute_capability_minor=6,
        )

    @pytest.fixture
    def gpu_ga100(self) -> GPU:
        """GA100 (CC 8.0) - uses instantaneous power API."""
        return GPU(
            manufacturer="NVIDIA",
            model="A100-SXM4-40GB",
            serial_number="SN-A100",
            uuid="GPU-uuid-a100",
            pci_id="0000:00:02.0",
            idle_power=50.0,
            max_power=400.0,
            compute_capability_major=8,
            compute_capability_minor=0,
        )

    @pytest.fixture
    def gpu_volta(self) -> GPU:
        """Volta GPU (CC 7.0) - uses instantaneous power API."""
        return GPU(
            manufacturer="NVIDIA",
            model="V100-SXM2-16GB",
            serial_number="SN-V100",
            uuid="GPU-uuid-v100",
            pci_id="0000:00:03.0",
            idle_power=35.0,
            max_power=300.0,
            compute_capability_major=7,
            compute_capability_minor=0,
        )

    @pytest.fixture
    def gpu_hopper(self) -> GPU:
        """Hopper GPU (CC 9.0) - uses averaged power API."""
        return GPU(
            manufacturer="NVIDIA",
            model="H100-SXM5-80GB",
            serial_number="SN-H100",
            uuid="GPU-uuid-h100",
            pci_id="0000:00:04.0",
            idle_power=75.0,
            max_power=700.0,
            compute_capability_major=9,
            compute_capability_minor=0,
        )

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_power_reading_returns_valid_watts(self, mock_pynvml, gpu_ampere: GPU):
        """Power reading returns positive value in Watts."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 75000  # 75W in mW

        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()

        assert len(result.measurements) == 1
        assert result.measurements[0].value == 75.0  # Converted to Watts
        assert result.measurements[0].value > 0

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_mw_to_w_conversion(self, mock_pynvml, gpu_ampere: GPU):
        """Power is correctly converted from milliwatts to watts."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150500  # 150.5W in mW

        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()

        assert result.measurements[0].value == 150.5

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_ampere_emits_average_metric(self, mock_pynvml, gpu_ampere: GPU):
        """Ampere GPU (CC 8.6) emits GPU_DEVICE_TOTAL_POWER_AVERAGE metric."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()

        assert result.measurements[0].metric_id == GPU_DEVICE_TOTAL_POWER_AVERAGE.name

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_hopper_emits_average_metric(self, mock_pynvml, gpu_hopper: GPU):
        """Hopper GPU (CC 9.0) emits GPU_DEVICE_TOTAL_POWER_AVERAGE metric."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 500000

        data_source = NVMLDataSource([gpu_hopper])
        result = data_source._read_device_power()

        assert result.measurements[0].metric_id == GPU_DEVICE_TOTAL_POWER_AVERAGE.name

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_ga100_emits_instant_metric(self, mock_pynvml, gpu_ga100: GPU):
        """GA100 (CC 8.0) emits GPU_DEVICE_TOTAL_POWER_INSTANT metric."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 250000

        data_source = NVMLDataSource([gpu_ga100])
        result = data_source._read_device_power()

        assert result.measurements[0].metric_id == GPU_DEVICE_TOTAL_POWER_INSTANT.name

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_volta_emits_instant_metric(self, mock_pynvml, gpu_volta: GPU):
        """Volta GPU (CC 7.0) emits GPU_DEVICE_TOTAL_POWER_INSTANT metric."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 200000

        data_source = NVMLDataSource([gpu_volta])
        result = data_source._read_device_power()

        assert result.measurements[0].metric_id == GPU_DEVICE_TOTAL_POWER_INSTANT.name

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_measurement_has_correct_entity_id(self, mock_pynvml, gpu_ampere: GPU):
        """Measurement has correct physical entity ID."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()

        assert result.measurements[0].physical_entity_id == gpu_ampere.id
        assert result.measurements[0].logical_entity_id is None

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_measurement_has_timestamp(self, mock_pynvml, gpu_ampere: GPU):
        """Measurement has a valid timestamp."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 100000

        before = datetime.now(UTC)
        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()
        after = datetime.now(UTC)

        assert before <= result.measurements[0].start_ts <= after

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_nvml_error_handled_gracefully(self, mock_pynvml, gpu_ampere: GPU):
        """NVML errors are logged and don't crash the collection."""
        mock_handle = MagicMock()
        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.return_value = mock_handle
        # Set the exception class BEFORE using it
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = Exception("Power reading not supported")

        data_source = NVMLDataSource([gpu_ampere])
        result = data_source._read_device_power()

        # Should return empty measurements on error, not crash
        assert len(result.measurements) == 0

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_multiple_gpus_power_collection(self, mock_pynvml, gpu_ampere: GPU, gpu_volta: GPU):
        """Power collection works for multiple GPUs with different generations."""
        mock_handle_ampere = MagicMock()
        mock_handle_volta = MagicMock()

        mock_pynvml.nvmlInit.return_value = None
        mock_pynvml.nvmlDeviceGetHandleByUUID.side_effect = lambda uuid: (
            mock_handle_ampere if uuid == gpu_ampere.uuid else mock_handle_volta
        )
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = lambda handle: (
            100000 if handle == mock_handle_ampere else 200000
        )

        data_source = NVMLDataSource([gpu_ampere, gpu_volta])
        result = data_source._read_device_power()

        assert len(result.measurements) == 2

        # Find measurements by entity
        ampere_measurement = next(m for m in result.measurements if m.physical_entity_id == gpu_ampere.id)
        volta_measurement = next(m for m in result.measurements if m.physical_entity_id == gpu_volta.id)

        # Ampere should emit AVERAGE, Volta should emit INSTANT
        assert ampere_measurement.metric_id == GPU_DEVICE_TOTAL_POWER_AVERAGE.name
        assert volta_measurement.metric_id == GPU_DEVICE_TOTAL_POWER_INSTANT.name

        # Check values
        assert ampere_measurement.value == 100.0
        assert volta_measurement.value == 200.0


class TestPowerMetricsRegistration:
    """Tests for power metrics registration in NVMLDataSource."""

    def test_power_metrics_in_entity_mapping(self):
        """Power metrics are registered in _entity_metric_mapping."""
        assert GPU_DEVICE_TOTAL_POWER_INSTANT in NVMLDataSource._entity_metric_mapping[GPU]
        assert GPU_DEVICE_TOTAL_POWER_AVERAGE in NVMLDataSource._entity_metric_mapping[GPU]

    @patch("metrion.monitoring.data_source.gpu.nvml.pynvml")
    def test_power_metrics_in_metric_lookup(self, mock_pynvml):
        """Power metrics are registered in _metric_lookup."""
        mock_pynvml.nvmlInit.return_value = None

        gpu = GPU(
            manufacturer="NVIDIA",
            model="Test",
            serial_number="SN",
            uuid="GPU-test",
            pci_id="0000:00:01.0",
            idle_power=25.0,
            max_power=150.0,
            compute_capability_major=8,
            compute_capability_minor=6,
        )

        data_source = NVMLDataSource([gpu])

        assert GPU_DEVICE_TOTAL_POWER_INSTANT in data_source._metric_lookup
        assert GPU_DEVICE_TOTAL_POWER_AVERAGE in data_source._metric_lookup
        assert data_source._metric_lookup[GPU_DEVICE_TOTAL_POWER_INSTANT] == data_source._read_device_power
        assert data_source._metric_lookup[GPU_DEVICE_TOTAL_POWER_AVERAGE] == data_source._read_device_power
