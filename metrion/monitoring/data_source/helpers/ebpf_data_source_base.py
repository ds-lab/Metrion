import ctypes as ct
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from multiprocessing import Event, Queue
from typing import Any

import polars as pl
from bcc.table import HashTable
from loguru import logger

from metrion.util import build_cgroup_id_map, write_temp_ipc

# Constants for adaptive interval calculation
_INTERVAL_DECAY_FACTOR: float = 8.0
_MILLISECONDS_TO_SECONDS: float = 1000.0


@dataclass
class EBPFConfig:  # noqa: N801
    """
    Configuration for eBPF-based monitoring.

    Attributes
    ----------
    min_interval_ms : float
        Minimum collection interval in milliseconds.
    max_interval_ms : float
        Maximum collection interval in milliseconds.
    metadata_queue : Queue
        Queue for passing metadata file paths between processes.
    monitoring_queue : Queue
        Queue for passing monitoring data file paths between processes.
    stop_event : Event
        Event to signal monitoring should stop.
    high_usage_threshold_percent : float
        Upper buffer usage percentage threshold. When exceeded, collection
        interval is decreased for more frequent collection.
    low_usage_threshold_percent : float
        Lower buffer usage percentage threshold. When usage falls below this,
        collection interval is increased to reduce overhead.
    map_capacity : int
        Maximum number of entries in the eBPF hash map.
    primary_data_map_name : str
        Name of the primary data map in the eBPF program.
    secondary_data_map_name : str
        Name of the secondary data map in the eBPF program.
    metadata_map_name : str
        Name of the metadata map in the eBPF program.
    buffer_selector_map_name : str
        Name of the buffer selector map in the eBPF program.
    inflight_map_name : str
        Name of the map tracking in-flight writes in the eBPF program.
    metadata_schema : dict
        Schema for the metadata DataFrame.
    raw_data_schema : dict
        Schema for the raw monitoring data DataFrame.
    """

    min_interval_ms: float
    max_interval_ms: float
    metadata_queue: Queue = field(default_factory=Queue)
    monitoring_queue: Queue = field(default_factory=Queue)
    stop_event: Event = field(default_factory=Event)  # type: ignore
    high_usage_threshold_percent: float = 75.0
    low_usage_threshold_percent: float = 20.0
    map_capacity: int = 2**16
    primary_data_map_name: str = "monitoring_data_map_1"
    secondary_data_map_name: str = "monitoring_data_map_2"
    pid_monitoring_data_buf_len: str = "__pid_monitoring_data_buf_len__"
    metadata_map_name: str = "monitoring_metadata_map"
    buffer_selector_map_name: str = "buffer"
    inflight_map_name: str = "inflight"
    metadata_schema: dict = field(default_factory=dict)
    raw_data_schema: dict = field(default_factory=dict)


@dataclass
class BufferUsageStats:
    """
    Statistics about eBPF hash map buffer usage.

    Attributes
    ----------
    active_buffer_index : int
        Index of the currently active buffer (0 or 1).
    entry_count : int
        Number of entries currently in the active buffer.
    usage_percent : float
        Percentage of buffer capacity currently in use (0-100).
    """

    active_buffer_index: int
    entry_count: int
    usage_percent: float


class EBPFDataSource:  # noqa: N801
    """
    Base class for eBPF-based data sources with adaptive monitoring capabilities.

    This class provides infrastructure for collecting data from eBPF programs
    using a double-buffering strategy to minimize data loss during collection.

    Parameters
    ----------
    config : EBPFConfig
        Configuration object for the eBPF data source.

    Attributes
    ----------
    config : EBPFConfig
        The configuration for this data source.
    _bpf : BPF
        The BPF object for interacting with the eBPF program.
        Must be set by subclasses.
    _is_monitoring : bool
        Flag indicating whether monitoring is currently active.
    _collection_interval_ms : float
        Current collection interval in milliseconds.
    """

    SENT_MON: tuple[str, str] = ("__EOF__", "monitoring")
    SENT_META: tuple[str, str] = ("__EOF__", "pidmeta")

    # Type hint for BPF object - set by subclasses
    _bpf: "BPF"  # type: ignore  # noqa: F821

    def __init__(self, config: EBPFConfig) -> None:
        self.config = config
        self._is_monitoring: bool = False
        self._collection_interval_ms: float = config.max_interval_ms

    def _get_buffer_usage_stats(self) -> BufferUsageStats:
        """
        Calculate current usage statistics for the active eBPF hash map buffer.

        Returns
        -------
        BufferUsageStats
            Statistics including active buffer index, entry count, and usage percentage.
        """
        buffer_index = self._bpf[self.config.buffer_selector_map_name][0].value

        if buffer_index == 0:
            active_map_len = self._bpf[self.config.pid_monitoring_data_buf_len][0].value
        else:
            active_map_len = self._bpf[self.config.pid_monitoring_data_buf_len][1].value

        usage_percent = (active_map_len / self.config.map_capacity) * 100.0

        return BufferUsageStats(
            active_buffer_index=buffer_index,
            entry_count=active_map_len,
            usage_percent=usage_percent,
        )

    def _swap_buffers_and_get_data(self) -> tuple[Any, ...]:
        """
        Atomically swap the active buffer and retrieve data from the previous buffer.

        This implements a double-buffering strategy where we flip to the alternate
        buffer before reading, ensuring the eBPF program can continue writing
        without blocking.

        Returns
        -------
        tuple[HashTable, int]
            A tuple containing the hash table with collected data and the
            previous buffer index.
        """
        previous_buffer_index = self._bpf[self.config.buffer_selector_map_name][0].value

        if previous_buffer_index == 0:
            data_map = self._bpf[self.config.primary_data_map_name]
            self._bpf[self.config.pid_monitoring_data_buf_len][0] = ct.c_uint(0)
            new_buffer_index = ct.c_uint(1)
        else:
            data_map = self._bpf[self.config.secondary_data_map_name]
            self._bpf[self.config.pid_monitoring_data_buf_len][1] = ct.c_uint(0)
            new_buffer_index = ct.c_uint(0)

        # Atomically switch to the new buffer
        self._bpf.get_table(self.config.buffer_selector_map_name)[ct.c_uint(0)] = new_buffer_index

        return data_map, previous_buffer_index

    def _process_raw_monitoring_data(self, raw_data: Sequence[HashTable], buffer_index: int) -> pl.DataFrame:
        """
        Process raw monitoring data retrieved from the eBPF hash map.

        Parameters
        ----------
        raw_data : HashTable
            The raw monitoring data from the eBPF hash map.
        buffer_index : int
            The buffer index from which the data was retrieved.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing the processed monitoring data.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_raw_monitoring_data")

    def _process_raw_metadata(self, raw_metadata: HashTable) -> pl.DataFrame:
        """
        Process raw metadata retrieved from the eBPF hash map.

        Parameters
        ----------
        raw_metadata : HashTable
            The raw metadata from the eBPF hash map.

        Returns
        -------
        pl.DataFrame
            A Polars DataFrame containing the processed metadata.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _process_raw_metadata")

    def _collect_and_enqueue_data(self) -> None:
        """
        Collect data from eBPF maps and enqueue for processing.

        Retrieves both monitoring data and metadata from the kernel,
        processes them into DataFrames, and places the results in
        their respective queues for downstream consumption.

        Notes
        -----
        Data is written to temporary IPC files and the file paths are
        enqueued for consumption by other processes. Empty DataFrames
        are not enqueued.
        """
        raw_monitoring_data, buffer_index = self._swap_buffers_and_get_data()

        monitoring_df = self._process_raw_monitoring_data(raw_monitoring_data, buffer_index)
        if len(monitoring_df) > 0:
            ipc_path = write_temp_ipc(monitoring_df)
            self.config.monitoring_queue.put_nowait(ipc_path)

        raw_metadata = self._bpf[self.config.metadata_map_name]
        metadata_df = self._process_raw_metadata(raw_metadata)
        if len(metadata_df) > 0:
            ipc_path = write_temp_ipc(metadata_df)
            self.config.metadata_queue.put_nowait(ipc_path)

    def _calculate_adaptive_interval(self, usage_percent: float) -> float:
        """
        Calculate an adaptive collection interval based on buffer usage.

        Uses an exponential decay function to determine the optimal collection
        interval. Higher buffer usage results in more frequent collection to
        prevent data loss.

        Parameters
        ----------
        usage_percent : float
            Current buffer usage as a percentage (0-100).

        Returns
        -------
        float
            The calculated collection interval in milliseconds, clamped to
            the configured min/max bounds.
        """
        decay_rate = _INTERVAL_DECAY_FACTOR * usage_percent / 100.0
        calculated_interval = self.config.max_interval_ms * math.exp(-decay_rate)

        return max(
            self.config.min_interval_ms,
            min(calculated_interval, self.config.max_interval_ms),
        )

    def _should_adjust_interval(self, usage_percent: float) -> bool:
        """
        Determine whether the collection interval should be adjusted.

        Interval adjustment is triggered when buffer usage exceeds the
        configured high threshold or falls below the low usage threshold.

        Parameters
        ----------
        usage_percent : float
            Current buffer usage as a percentage (0-100).

        Returns
        -------
        bool
            True if the interval should be adjusted, False otherwise.
        """
        return (
            usage_percent > self.config.high_usage_threshold_percent
            or usage_percent < self.config.low_usage_threshold_percent
        )

    def _update_collection_interval(self, usage_stats: BufferUsageStats) -> None:
        """
        Update the collection interval based on current buffer usage.

        If the usage falls outside the configured thresholds, recalculates
        the collection interval using an exponential decay function and
        logs the change for debugging purposes.

        Parameters
        ----------
        usage_stats : BufferUsageStats
            Current buffer usage statistics.

        Notes
        -----
        No action is taken if usage is within the normal range
        (between low and high thresholds).
        """
        if not self._should_adjust_interval(usage_stats.usage_percent):
            return

        previous_interval_ms = self._collection_interval_ms
        new_interval_ms = self._calculate_adaptive_interval(usage_stats.usage_percent)

        # Only log significant interval changes (>10% change)
        if abs(new_interval_ms - previous_interval_ms) / previous_interval_ms > 0.1:
            logger.trace(
                f"Adaptive interval: {previous_interval_ms / 1000:.3f}s → {new_interval_ms / 1000:.3f}s "
                f"(buffer: {usage_stats.usage_percent:.1f}%)"
            )

        self._collection_interval_ms = new_interval_ms

    def _perform_collection_cycle(self) -> None:
        """
        Execute a single data collection cycle.

        Collects data from eBPF maps and adjusts the collection interval
        based on current buffer usage.

        Notes
        -----
        This method performs two operations in sequence:

        1. Collects and enqueues data from the eBPF maps
        2. Checks buffer usage and updates the collection interval if needed
        """
        self._collect_and_enqueue_data()

        usage_stats = self._get_buffer_usage_stats()
        self._update_collection_interval(usage_stats)

    def _calculate_sleep_duration(self, next_collection_time: float) -> float:
        """
        Calculate how long to sleep until the next collection.

        Parameters
        ----------
        next_collection_time : float
            Timestamp of the next scheduled collection.

        Returns
        -------
        float
            Duration to sleep in seconds, with a minimum to prevent CPU spinning.
        """
        min_sleep_sec = self.config.min_interval_ms / _MILLISECONDS_TO_SECONDS
        time_until_next = next_collection_time - time.time()
        return max(min_sleep_sec, time_until_next)

    def start_adaptive_monitoring(self) -> None:
        """
        Start adaptive monitoring that dynamically adjusts collection intervals.

        The monitoring system periodically collects data from eBPF hash maps
        and automatically adjusts the collection frequency based on buffer
        usage to prevent overflow while optimizing performance.

        The adaptive algorithm uses exponential decay to calculate intervals:
        - High buffer usage triggers more frequent collection
        - Low buffer usage allows longer intervals to reduce overhead

        Notes
        -----
        This method blocks until the stop event is set. The monitoring loop:

        1. Checks if collection is due based on the current interval
        2. Collects data from eBPF maps
        3. Adjusts the interval based on buffer usage
        4. Sleeps until the next collection time

        Call `stop_monitoring()` to gracefully terminate the monitoring loop.
        """
        if self._is_monitoring:
            return

        self._is_monitoring = True
        next_collection_time = time.time()

        try:
            while not self.config.stop_event.is_set():
                current_time = time.time()

                if current_time >= next_collection_time:
                    self._perform_collection_cycle()

                    next_collection_time = current_time + 0.2

                sleep_duration = self._calculate_sleep_duration(next_collection_time)
                time.sleep(sleep_duration)
        finally:
            self.stop_monitoring()

    def _wait_for_inflight_writes(self) -> None:
        """
        Wait for all in-flight eBPF writes to complete.

        Polls the inflight map until both buffers report zero pending writes.

        Raises
        ------
        Exception
            Logs error and continues if unable to read inflight map.
        """
        try:
            inflight_map = self._bpf[self.config.inflight_map_name]
            while inflight_map[0].value > 0 or inflight_map[1].value > 0:
                time.sleep(0.05)
        except Exception as e:
            logger.error(f"Error while waiting for in-flight writes: {e}")

    def _has_pending_buffer_data(self) -> bool:
        """
        Check if either data buffer has pending entries.

        Returns
        -------
        bool
            True if either primary or secondary buffer contains data.
        """
        primary_count = self._bpf[self.config.pid_monitoring_data_buf_len][0].value
        secondary_count = self._bpf[self.config.pid_monitoring_data_buf_len][1].value
        return primary_count > 0 or secondary_count > 0

    def _drain_buffer(self, map_name: str, buffer_index: int) -> bool:
        """
        Drain data from a single buffer and enqueue for processing.

        Parameters
        ----------
        map_name : str
            Name of the eBPF map to drain.
        buffer_index : int
            Index of the buffer (0 or 1).

        Returns
        -------
        bool
            True if data was drained, False if buffer was empty.
        """
        raw_data = self._bpf[map_name]
        df = self._process_raw_monitoring_data(raw_data, buffer_index)

        if len(df) > 0:
            ipc_path = write_temp_ipc(df)
            self.config.monitoring_queue.put_nowait(ipc_path)
            return True
        return False

    def _drain_all_buffers(self) -> None:
        """
        Drain all remaining data from both eBPF data buffers.

        Continues draining until both buffers are empty, with a small
        sleep between iterations to avoid busy-waiting.
        """
        while self._has_pending_buffer_data():
            primary_drained = self._drain_buffer(self.config.primary_data_map_name, 0)
            secondary_drained = self._drain_buffer(self.config.secondary_data_map_name, 1)

            if not primary_drained and not secondary_drained:
                time.sleep(0.02)

    def _flush_final_metadata(self) -> None:
        """
        Capture and enqueue the final metadata snapshot.
        """
        raw_metadata = self._bpf[self.config.metadata_map_name]
        metadata_df = self._process_raw_metadata(raw_metadata)

        if len(metadata_df) > 0:
            ipc_path = write_temp_ipc(metadata_df)
            self.config.metadata_queue.put_nowait(ipc_path)

    def _close_queues(self) -> None:
        """
        Send sentinel values and close all queues.

        Signals downstream consumers that no more data will be sent.
        """
        self.config.monitoring_queue.put(self.SENT_MON)
        self.config.monitoring_queue.close()
        self.config.monitoring_queue.join_thread()

        self.config.metadata_queue.put(self.SENT_META)
        self.config.metadata_queue.close()
        self.config.metadata_queue.join_thread()

    def _flush_all_pending(self) -> None:
        """
        Flush any remaining data from eBPF maps before stopping.

        Ensures that all collected data is processed and enqueued
        before terminating the monitoring session.

        Notes
        -----
        This method performs the following sequence:

        1. Waits for in-flight writes to complete in both buffers
        2. Drains any remaining data from both buffers
        3. Captures a final metadata snapshot
        4. Sends sentinel values and closes the queues
        """
        logger.debug("Flushing all pending data")
        self._wait_for_inflight_writes()
        logger.debug("Draining all buffers")
        self._drain_all_buffers()
        logger.debug("Flushing final metadata")
        self._flush_final_metadata()
        logger.debug("Closing queues")
        self._close_queues()
        logger.debug("Flushing all pending data completed")

    def stop_monitoring(self) -> None:
        """
        Stop the adaptive monitoring process.

        Sets the monitoring flag to False, signals the eBPF program to stop
        accepting new data, and flushes any pending data from the buffers.

        Notes
        -----
        This method is safe to call multiple times; subsequent calls after
        the first will be no-ops.
        """
        if not self._is_monitoring:
            return

        self._is_monitoring = False
        self._bpf.get_table(self.config.buffer_selector_map_name)[ct.c_uint(0)] = ct.c_uint(2)
        self._flush_all_pending()

    @classmethod
    def raw_pid_metadata_add_cgroup_name(cls, raw_pid_metadata_df: pl.DataFrame, schema: dict) -> pl.DataFrame:
        """Add cgroup name to the raw pid metadata dataframe."""
        if len(raw_pid_metadata_df) == 0:
            return pl.DataFrame(
                schema=schema,
                orient="row",
            )
        cgroup_id_map = build_cgroup_id_map()

        def map_cgroup_id(cgroup) -> str:
            """Map cgroup_id to cgroup_name using the provided mapping."""
            return cgroup_id_map.get(cgroup, "unknown")

        out_df = raw_pid_metadata_df.with_columns(
            pl.col("cgroup_id").map_elements(map_cgroup_id, return_dtype=pl.String).alias("cgroup_name")
        )

        return out_df
