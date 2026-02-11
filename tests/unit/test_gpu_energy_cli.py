"""Unit tests for GPU energy attribution SQL queries and CLI integration."""

import datetime

import polars as pl
import pytest

from metrion.energy_attribution.energy_attribution_cli import (
    collapse_time_stamps,
    filter_data_by_cmdline_and_comm,
    load_sql_queries,
)


class TestLoadSQLQueries:
    """Tests for SQL query loading."""

    def test_load_sql_queries_returns_six_queries(self):
        """Verify load_sql_queries returns six query strings."""
        queries = load_sql_queries("sqlite")
        assert len(queries) == 6
        assert all(isinstance(q, str) for q in queries)

    def test_load_sql_queries_gpu_contains_expected_content(self):
        """Verify GPU queries contain expected SQL patterns."""
        queries = load_sql_queries("sqlite")
        gpu_analysis_query = queries[4]
        gpu_group_query = queries[5]

        # GPU analysis query should reference GPU metrics
        assert "GPU_DEVICE_TOTAL_ENERGY" in gpu_analysis_query
        assert "GPU_DEVICE_PID_COMPUTE_UTILIZATION" in gpu_analysis_query
        assert "energy_interval_gpu_tmp" in gpu_analysis_query

        # GPU group query should reference the view
        assert "gpu_energy_per_pid_view" in gpu_group_query

    def test_load_sql_queries_sqlite_strips_markers(self):
        """Verify --sqlite markers are stripped for sqlite dialect."""
        queries = load_sql_queries("sqlite")
        gpu_analysis_query = queries[4]

        # Should have CREATE INDEX without --sqlite prefix
        assert "CREATE INDEX IF NOT EXISTS" in gpu_analysis_query
        # Should not have --sqlite marker
        assert "--sqlite" not in gpu_analysis_query

    def test_load_sql_queries_duckdb_keeps_markers(self):
        """Verify --sqlite markers remain as comments for duckdb."""
        queries = load_sql_queries("duckdb")
        gpu_analysis_query = queries[4]

        # For duckdb, --sqlite lines remain as comments
        assert "--sqlite" in gpu_analysis_query


class TestGPUDataProcessing:
    """Tests for GPU data processing functions."""

    @pytest.fixture
    def sample_gpu_df(self):
        """Create sample GPU energy DataFrame."""
        return pl.DataFrame(
            {
                "interval_start": [
                    datetime.datetime(2026, 1, 8, 10, 0, 0),
                    datetime.datetime(2026, 1, 8, 10, 0, 1),
                    datetime.datetime(2026, 1, 8, 10, 0, 2),
                ],
                "interval_end": [
                    datetime.datetime(2026, 1, 8, 10, 0, 1),
                    datetime.datetime(2026, 1, 8, 10, 0, 2),
                    datetime.datetime(2026, 1, 8, 10, 0, 3),
                ],
                "cmdline": ["python train.py", "python train.py", "python train.py"],
                "executable_name": ["python", "python", "python"],
                "resource_group_name": ["default", "default", "default"],
                "resource_group_id": [1, 1, 1],
                "total_energy_j": [10.0, 15.0, 12.0],
                "dyn_energy_j": [8.0, 12.0, 10.0],
                "idle_energy_j": [2.0, 3.0, 2.0],
            }
        )

    def test_collapse_time_stamps_gpu_data(self, sample_gpu_df):
        """Verify collapse_time_stamps works with GPU data.

        Note: collapse_time_stamps uses a 500ms sentinel, so consecutive
        1-second intervals won't collapse. This test verifies the function
        runs without error and preserves energy values.
        """
        result = collapse_time_stamps(sample_gpu_df)

        assert result is not None
        # With 1s intervals, rows don't collapse (gap > 500ms sentinel)
        assert len(result) == 3
        # Verify total energy is preserved
        assert result["total_energy_j"].sum() == 37.0  # 10 + 15 + 12

    def test_filter_data_by_cmdline_gpu(self, sample_gpu_df):
        """Verify filter_data_by_cmdline_and_comm works with GPU data."""
        result = filter_data_by_cmdline_and_comm(sample_gpu_df, "train")

        assert result is not None
        assert len(result) == 3

    def test_filter_data_by_cmdline_no_match(self, sample_gpu_df):
        """Verify filter returns None when no match found."""
        result = filter_data_by_cmdline_and_comm(sample_gpu_df, "nonexistent")

        assert result is None


class TestGPUSQLQueries:
    """Tests for GPU SQL query structure."""

    def test_gpu_analysis_query_has_wrap_around_handling(self):
        """Verify GPU query handles energy counter wrap-around."""
        queries = load_sql_queries("sqlite")
        gpu_query = queries[4]

        # Should have wrap-around value for 64-bit counter
        assert "18446744073709551615" in gpu_query

    def test_gpu_analysis_query_has_idle_dynamic_split(self):
        """Verify GPU query splits energy into idle and dynamic."""
        queries = load_sql_queries("sqlite")
        gpu_query = queries[4]

        assert "idle_energy_j" in gpu_query
        assert "dynamic_energy_j" in gpu_query
        assert "idle_power" in gpu_query

    def test_gpu_analysis_query_uses_utilization(self):
        """Verify GPU query uses SM utilization for attribution."""
        queries = load_sql_queries("sqlite")
        gpu_query = queries[4]

        assert "util_share" in gpu_query
        assert "time_share" in gpu_query
        assert "total_util" in gpu_query

    def test_gpu_group_query_aggregates_by_cmdline(self):
        """Verify GPU group query aggregates by process attributes."""
        queries = load_sql_queries("sqlite")
        gpu_group_query = queries[5]

        assert "GROUP BY" in gpu_group_query
        assert "cmdline" in gpu_group_query
        assert "resource_group_name" in gpu_group_query
