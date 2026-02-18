import time

import polars as pl
import psutil
from loguru import logger


def _validate_bpf_clock_domain(df: pl.DataFrame, col_name: str = "time_end") -> bool:
    latest_bpf_ns = df.select(pl.col(col_name).max()).item()
    if latest_bpf_ns is None or latest_bpf_ns == 0:
        return False

    boot_epoch_ns = int(psutil.boot_time() * 1e9)
    converted_epoch_ns = boot_epoch_ns + int(latest_bpf_ns)
    now_epoch_ns = time.time_ns()
    delta_seconds = abs(now_epoch_ns - converted_epoch_ns) / 1e9

    if delta_seconds > 60:
        logger.warning(f"BPF clock drift detected (TCP): {delta_seconds:.1f}s between BPF timestamps and wall clock.")

    return True
