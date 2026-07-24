import pathlib

import pytest

from metrion.discovery.system_scanner.bare_metal.linux.util import (
    get_cpu_idle_power_data,
    parse_rapl_idle_log,
)

PACKAGE_HEADER = "=== Package 0 ===\n"


def _write_log(tmp_path: pathlib.Path, body: str) -> pathlib.Path:
    log_path = tmp_path / "rapl-idle.log"
    log_path.write_text(body)
    return log_path


# --- Accepted numeric grammar: digits, digits"."digits, digits","digits ---


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("11.3824", 11.3824),  # 1. existing decimal-point format (regression)
        ("10,0196", 10.0196),  # 2. decimal-comma format
        ("10", 10.0),  # 3. bare integer per accepted grammar (script never emits this today)
        ("0.0000", 0.0),  # 4a. zero, decimal point
        ("0,0000", 0.0),  # 4b. zero, decimal comma
    ],
)
def test_parse_package_total_accepted_formats(tmp_path, token, expected):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + f"Package Total      :  {token} W\n")
    result = parse_rapl_idle_log(log_path)
    assert result[0]["package"] == pytest.approx(expected)


def test_parse_package_total_accepts_real_world_truncated_value(tmp_path):
    """The actual corrupted production value ("10,0000") is syntactically valid
    single-comma-decimal input under the accepted grammar and must parse to 10.0.

    This is intentional: the fractional precision was already lost upstream by
    bash's locale-mangled printf before this value was ever written to the log.
    The parser's job is to correctly interpret the text it is given, not to
    detect or repair historical data corruption -- that is fixed separately by
    pinning LC_NUMERIC=C in freeze-idle-rapl.sh for future measurements.
    """
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :  10,0000 W\n")
    result = parse_rapl_idle_log(log_path)
    assert result[0]["package"] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "token",
    [
        "1,234.56",  # 5a. mixed separators (comma then dot)
        "1.234,56",  # 5b. mixed separators (dot then comma)
        "10,01,96",  # 6a. repeated comma separators
        "10.01.96",  # 6b. repeated dot separators
        "-1.0",  # negative value, not part of accepted grammar
    ],
)
def test_parse_package_total_rejects_malformed_values(tmp_path, token):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + f"Package Total      :  {token} W\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


def test_parse_package_total_rejects_na_value(tmp_path):
    """Unlike CPU Cores/Integrated GPU/DRAM, Package Total has no legitimate
    N/A branch in freeze-idle-rapl.sh -- it is always printed numerically. An
    N/A (or any other non-numeric) value here is a genuine anomaly and must
    raise, never silently fall back to None or 0.0."""
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :      N/A\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


def test_parse_package_total_rejects_empty_value(tmp_path):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :    W\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


def test_parse_package_total_rejects_trailing_malformed_content(tmp_path):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :  10.0196 W extra garbage\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


# --- CPU Cores / Integrated GPU / DRAM: N/A is a legitimate "domain absent"
# signal (per util.py's own documented format) and must keep parsing to None ---


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("CPU Cores", "cpu"),
        ("Integrated GPU", "integrated-gpu"),
        ("DRAM", "dram"),
    ],
)
def test_parse_optional_field_na_preserves_none(tmp_path, label, key):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + f"Package Total      :  11.3824 W\n{label}          : N/A\n")
    result = parse_rapl_idle_log(log_path)
    assert result[0][key] is None


@pytest.mark.parametrize(
    ("label", "key"),
    [
        ("CPU Cores", "cpu"),
        ("Integrated GPU", "integrated-gpu"),
        ("DRAM", "dram"),
    ],
)
def test_parse_optional_field_malformed_value_raises(tmp_path, label, key):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + f"Package Total      :  11.3824 W\n{label}          : 1,234.56 W\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


def test_parse_all_four_fields_mixed_separators(tmp_path):
    """13. Parsing all applicable fields (package, CPU, integrated GPU, DRAM),
    exercising both accepted separators across fields in a single log."""
    log_path = _write_log(
        tmp_path,
        PACKAGE_HEADER
        + "Package Total      :  10,0196 W\n"
        + "CPU Cores          :  8.1234 W\n"
        + "Integrated GPU     :  1,5000 W\n"
        + "DRAM               :  2.3000 W\n",
    )
    result = parse_rapl_idle_log(log_path)
    assert result[0]["package"] == pytest.approx(10.0196)
    assert result[0]["cpu"] == pytest.approx(8.1234)
    assert result[0]["integrated-gpu"] == pytest.approx(1.5)
    assert result[0]["dram"] == pytest.approx(2.3)


def test_parse_entirely_absent_optional_field_stays_none(tmp_path):
    """10. A field whose line never appears at all in the log (as opposed to
    being present with value N/A) must also stay None -- the established
    contract from the initial per-package dict, unaffected by this fix."""
    log_path = _write_log(
        tmp_path,
        PACKAGE_HEADER + "Package Total      :  11.3824 W\nCPU Cores          :  8.1234 W\n",
    )
    result = parse_rapl_idle_log(log_path)
    assert result[0]["integrated-gpu"] is None
    assert result[0]["dram"] is None


# --- File / package level error handling (pre-existing behavior, regression-only) ---


def test_parse_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does-not-exist.log"
    with pytest.raises(FileNotFoundError):
        parse_rapl_idle_log(missing_path)


def test_parse_missing_package_total_data_raises(tmp_path):
    """12. A log with no '=== Package N ===' section at all yields no package
    data whatsoever and must raise, rather than return an empty result."""
    log_path = _write_log(tmp_path, "LOG    : /some/path\nDwell  : 60s\n")
    with pytest.raises(ValueError):
        parse_rapl_idle_log(log_path)


# --- get_cpu_idle_power_data: thin wrapper, confirm it delegates and propagates ---


def test_get_cpu_idle_power_data_delegates_to_parser(tmp_path):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :  11.3824 W\n")
    result = get_cpu_idle_power_data(log_path)
    assert result[0]["package"] == pytest.approx(11.3824)


def test_get_cpu_idle_power_data_propagates_malformed_value(tmp_path):
    log_path = _write_log(tmp_path, PACKAGE_HEADER + "Package Total      :  1,234.56 W\n")
    with pytest.raises(ValueError):
        get_cpu_idle_power_data(log_path)
