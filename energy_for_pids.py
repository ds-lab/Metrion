"""
Example: get total energy attribution for a list of PIDs from a monitoring database.

Usage:
    python energy_for_pids.py <db_path> <pid1> [pid2 ...]

Example:
    python energy_for_pids.py monitor_2026_03_11_10_40_01.db 827341 789826
"""

import sys
from pathlib import Path

import polars as pl
from sqlalchemy import text

from metrion.energy_attribution.energy_attribution_cli import (
    close_db_connection,
    connect_to_db,
    get_energy_for_pids,
)


def _get_time_range(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT MIN(start_ts), MAX(start_ts) FROM measurement WHERE metric_id = 'CPU_SOCKET_PKG_ENERGY'")
        ).fetchone()
    return row[0], row[1]


def energy_for_pids(db_path: Path, pids: list[int]) -> pl.DataFrame:
    engine = connect_to_db(db_path, type="duckdb")
    if engine is None:
        raise RuntimeError(f"Could not connect to {db_path}")

    start_time, end_time = _get_time_range(engine)
    df = get_energy_for_pids(engine, pids=pids, start_time=start_time, end_time=end_time)
    close_db_connection(engine)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    db = Path(sys.argv[1])
    pid_list = [int(p) for p in sys.argv[2:]]

    result = energy_for_pids(db, pid_list)
    print(result)
