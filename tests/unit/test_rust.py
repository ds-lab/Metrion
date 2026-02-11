import polars as pl
import pytest

import metrion.metrion_rust as mmr


def test_prepare_monitoring_data_single_task():
    """Test processing a single task with start and end measurements"""
    # Create input dataframe with the expected format for prepare_monitoring_data
    raw = pl.DataFrame(
        {
            "pid": [1],
            "exec_count": [0],
            "cpu_core": [0],
            "time_start": [1_000_000],  # microseconds
            "aperf_start": [100],
            "mperf_start": [200],
            "unhalted_core_cycles_start": [300],
            "time_end": [2_000_000],  # microseconds
            "aperf_end": [150],
            "mperf_end": [250],
            "unhalted_core_cycles_end": [400],
        },
        schema={
            "pid": pl.UInt32,
            "exec_count": pl.UInt64,
            "cpu_core": pl.UInt32,
            "time_start": pl.UInt64,
            "aperf_start": pl.UInt64,
            "mperf_start": pl.UInt64,
            "unhalted_core_cycles_start": pl.UInt64,
            "time_end": pl.UInt64,
            "aperf_end": pl.UInt64,
            "mperf_end": pl.UInt64,
            "unhalted_core_cycles_end": pl.UInt64,
        },
    )

    # Call the Rust function with ht_factor
    ht_factor = 0.5
    result = mmr.prepare_monitoring_data(raw, ht_factor)

    # Verify output structure
    expected_columns = [
        "pid",
        "start_time",
        "end_time",
        "cpu_core",
        "delta_aperf",
        "delta_mperf",
        "delta_unhalted_core_cycles",
        "execution_time",
        "shared_state",
    ]
    assert result.columns == expected_columns

    # Verify single row output
    assert result.height == 1

    # Check the values
    row = result.row(0)
    assert row[0] == 1  # pid
    assert row[1] == 1_000_000  # start_time
    assert row[2] == 2_000_000  # end_time
    assert row[3] == 0  # cpu_core
    assert row[4] == 50  # delta_aperf (150 - 100)
    assert row[5] == 50  # delta_mperf (250 - 200)
    assert row[6] == 100  # delta_unhalted_core_cycles (400 - 300)
    assert row[7] == 1_000_000.0  # execution_time (2_000_000 - 1_000_000)
    assert row[8] == "alone"  # shared_state


def test_prepare_monitoring_data_empty_input():
    """Test handling of empty input DataFrame"""
    raw = pl.DataFrame(
        schema={
            "pid": pl.UInt32,
            "exec_count": pl.UInt64,
            "cpu_core": pl.UInt32,
            "time_start": pl.UInt64,
            "aperf_start": pl.UInt64,
            "mperf_start": pl.UInt64,
            "unhalted_core_cycles_start": pl.UInt64,
            "time_end": pl.UInt64,
            "aperf_end": pl.UInt64,
            "mperf_end": pl.UInt64,
            "unhalted_core_cycles_end": pl.UInt64,
        }
    )

    result = mmr.prepare_monitoring_data(raw, 0.5)
    assert result.height == 0


def test_prepare_monitoring_data_filtered_zero_values():
    """Test that rows with zero values are filtered out"""
    raw = pl.DataFrame(
        {
            "pid": [1, 2, 3],
            "exec_count": [0, 1, 2],
            "cpu_core": [0, 0, 0],
            "time_start": [1_000_000, 2_000_000, 3_000_000],
            "aperf_start": [0, 100, 200],  # First row has zero aperf_start
            "mperf_start": [200, 0, 300],  # Second row has zero mperf_start
            "unhalted_core_cycles_start": [300, 400, 500],
            "time_end": [2_000_000, 3_000_000, 4_000_000],
            "aperf_end": [150, 150, 250],
            "mperf_end": [250, 250, 350],
            "unhalted_core_cycles_end": [400, 500, 600],
        },
        schema={
            "pid": pl.UInt32,
            "exec_count": pl.UInt64,
            "cpu_core": pl.UInt32,
            "time_start": pl.UInt64,
            "aperf_start": pl.UInt64,
            "mperf_start": pl.UInt64,
            "unhalted_core_cycles_start": pl.UInt64,
            "time_end": pl.UInt64,
            "aperf_end": pl.UInt64,
            "mperf_end": pl.UInt64,
            "unhalted_core_cycles_end": pl.UInt64,
        },
    )

    result = mmr.prepare_monitoring_data(raw, 0.5)

    # Only the third row should remain (pid=3) since first two have zero values
    assert result.height == 1
    assert result["pid"].to_list() == [3]


def test_prepare_monitoring_data_same_time_filtered():
    """Test that rows with same start and end time are filtered out"""
    raw = pl.DataFrame(
        {
            "pid": [1, 2],
            "exec_count": [0, 1],
            "cpu_core": [0, 0],
            "time_start": [1_000_000, 2_000_000],
            "aperf_start": [100, 200],
            "mperf_start": [200, 300],
            "unhalted_core_cycles_start": [300, 400],
            "time_end": [1_000_000, 3_000_000],  # First row has same start/end time
            "aperf_end": [100, 250],
            "mperf_end": [100, 350],
            "unhalted_core_cycles_end": [300, 500],
        },
        schema={
            "pid": pl.UInt32,
            "exec_count": pl.UInt64,
            "cpu_core": pl.UInt32,
            "time_start": pl.UInt64,
            "aperf_start": pl.UInt64,
            "mperf_start": pl.UInt64,
            "unhalted_core_cycles_start": pl.UInt64,
            "time_end": pl.UInt64,
            "aperf_end": pl.UInt64,
            "mperf_end": pl.UInt64,
            "unhalted_core_cycles_end": pl.UInt64,
        },
    )

    result = mmr.prepare_monitoring_data(raw, 0.5)

    # Only the second row should remain (pid=2)
    assert result.height == 1
    assert result["pid"].to_list() == [2]


def test_prepare_monitoring_data_multiple_tasks():
    """Test processing multiple tasks with different timings"""
    raw = pl.DataFrame(
        {
            "pid": [1, 2],
            "exec_count": [0, 0],
            "cpu_core": [0, 0],
            "time_start": [1_000_000, 1_500_000],
            "aperf_start": [100, 200],
            "mperf_start": [200, 300],
            "unhalted_core_cycles_start": [300, 400],
            "time_end": [2_000_000, 2_500_000],
            "aperf_end": [150, 250],
            "mperf_end": [250, 350],
            "unhalted_core_cycles_end": [400, 500],
        },
        schema={
            "pid": pl.UInt32,
            "exec_count": pl.UInt64,
            "cpu_core": pl.UInt32,
            "time_start": pl.UInt64,
            "aperf_start": pl.UInt64,
            "mperf_start": pl.UInt64,
            "unhalted_core_cycles_start": pl.UInt64,
            "time_end": pl.UInt64,
            "aperf_end": pl.UInt64,
            "mperf_end": pl.UInt64,
            "unhalted_core_cycles_end": pl.UInt64,
        },
    )

    result = mmr.prepare_monitoring_data(raw, 1.5)
    with pl.Config(tbl_rows=10, tbl_cols=20):
        print(raw)
        print(result)

    # Should have multiple rows for overlapping tasks
    assert result.height > 0
    assert len(result["pid"].unique()) >= 2  # Should have both PIDs represented


if __name__ == "__main__":
    # allows quick ad-hoc run:  `python test_rust.py`
    pytest.main([__file__])
