"""Unit tests for GPU discovery compute capability parsing."""

from metrion.discovery.system_scanner.bare_metal.linux.gpu import _parse_compute_capability


class TestParseComputeCapability:
    """Tests for _parse_compute_capability function."""

    def test_parse_valid_compute_cap(self):
        """Parse standard compute capability string."""
        major, minor = _parse_compute_capability("8.6")
        assert major == 8
        assert minor == 6

    def test_parse_single_digit_minor(self):
        """Parse CC with single digit minor version."""
        major, minor = _parse_compute_capability("7.0")
        assert major == 7
        assert minor == 0

    def test_parse_hopper(self):
        """Parse Hopper CC 9.0."""
        major, minor = _parse_compute_capability("9.0")
        assert major == 9
        assert minor == 0

    def test_parse_empty_string_returns_zero(self):
        """Empty string returns (0, 0)."""
        major, minor = _parse_compute_capability("")
        assert major == 0
        assert minor == 0

    def test_parse_invalid_format_returns_zero(self):
        """Invalid format returns (0, 0)."""
        major, minor = _parse_compute_capability("invalid")
        assert major == 0
        assert minor == 0

    def test_parse_missing_minor_returns_zero(self):
        """Single number without dot returns (0, 0)."""
        major, minor = _parse_compute_capability("8")
        assert major == 0
        assert minor == 0

    def test_parse_non_numeric_returns_zero(self):
        """Non-numeric parts return (0, 0)."""
        major, minor = _parse_compute_capability("a.b")
        assert major == 0
        assert minor == 0
