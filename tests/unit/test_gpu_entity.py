"""Unit tests for GPU entity compute capability detection and properties."""

import pytest

from metrion.entity.gpu import GPU


class TestGPUComputeCapability:
    """Tests for GPU compute capability properties."""

    @pytest.fixture
    def gpu_ga100(self) -> GPU:
        """GA100 (A100) GPU: CC 8.0, instantaneous power."""
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
    def gpu_a10(self) -> GPU:
        """A10 GPU: CC 8.6, 1s averaged power."""
        return GPU(
            manufacturer="NVIDIA",
            model="A10",
            serial_number="SN-A10",
            uuid="GPU-uuid-a10",
            pci_id="0000:00:02.0",
            idle_power=25.0,
            max_power=150.0,
            compute_capability_major=8,
            compute_capability_minor=6,
        )

    @pytest.fixture
    def gpu_v100(self) -> GPU:
        """V100 GPU: CC 7.0, instantaneous power."""
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
    def gpu_h100(self) -> GPU:
        """H100 GPU: CC 9.0, 1s averaged power."""
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

    def test_ga100_has_instant_power(self, gpu_ga100: GPU):
        """GA100 (CC 8.0) returns instantaneous power, not averaged."""
        assert gpu_ga100.is_ampere_or_newer is True
        assert gpu_ga100.has_averaged_power_api is False

    def test_a10_has_averaged_power(self, gpu_a10: GPU):
        """A10 (CC 8.6) returns 1s averaged power."""
        assert gpu_a10.is_ampere_or_newer is True
        assert gpu_a10.has_averaged_power_api is True

    def test_v100_has_instant_power(self, gpu_v100: GPU):
        """V100 (CC 7.0) returns instantaneous power."""
        assert gpu_v100.is_ampere_or_newer is False
        assert gpu_v100.has_averaged_power_api is False

    def test_h100_has_averaged_power(self, gpu_h100: GPU):
        """H100 (CC 9.0) returns 1s averaged power."""
        assert gpu_h100.is_ampere_or_newer is True
        assert gpu_h100.has_averaged_power_api is True

    def test_compute_capability_stored(self, gpu_a10: GPU):
        """Verify compute capability fields are stored correctly."""
        assert gpu_a10.compute_capability_major == 8
        assert gpu_a10.compute_capability_minor == 6

    def test_to_df_includes_compute_capability(self, gpu_a10: GPU):
        """Verify DataFrame includes compute capability columns."""
        df = gpu_a10._to_df()
        assert "compute_capability_major" in df.columns
        assert "compute_capability_minor" in df.columns
        assert df["compute_capability_major"][0] == 8
        assert df["compute_capability_minor"][0] == 6


class TestGPUComputeCapabilityEdgeCases:
    """Edge case tests for compute capability handling."""

    def test_default_compute_capability_zero(self):
        """GPU with default compute capability should not have averaged power."""
        gpu = GPU(
            manufacturer="NVIDIA",
            model="Unknown",
            serial_number="SN-UNK",
            uuid="GPU-uuid-unk",
            pci_id="0000:00:05.0",
            idle_power=20.0,
            max_power=100.0,
            # compute_capability defaults to 0
        )
        assert gpu.compute_capability_major == 0
        assert gpu.compute_capability_minor == 0
        assert gpu.is_ampere_or_newer is False
        assert gpu.has_averaged_power_api is False

    @pytest.mark.parametrize(
        "major,minor,expected_averaged",
        [
            (7, 0, False),  # Volta
            (7, 5, False),  # Turing
            (8, 0, False),  # GA100 (A100)
            (8, 6, True),  # Ampere (A40, A10)
            (8, 9, True),  # Ada Lovelace
            (9, 0, True),  # Hopper
        ],
    )
    def test_has_averaged_power_by_generation(self, major: int, minor: int, expected_averaged: bool):
        """Verify has_averaged_power_api for various GPU generations."""
        gpu = GPU(
            manufacturer="NVIDIA",
            model=f"Test-CC-{major}.{minor}",
            serial_number="SN-TEST",
            uuid=f"GPU-uuid-{major}-{minor}",
            pci_id="0000:00:06.0",
            idle_power=30.0,
            max_power=200.0,
            compute_capability_major=major,
            compute_capability_minor=minor,
        )
        assert gpu.has_averaged_power_api is expected_averaged
