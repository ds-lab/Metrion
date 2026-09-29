from __future__ import annotations

import ctypes
import multiprocessing as mp
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import polars as pl
from loguru import logger

from metrion.util import IpcAsyncWriter

# Drops are checked on every poll iteration; without a throttle a busy ring buffer would
# produce a warning every few milliseconds.
_DROP_WARNING_INTERVAL_S = 10.0

# Pause between two drains of the ring buffer. Blocking until data arrives instead wakes the
# process for the first event after every drain, and for sched_switch the process going back
# to sleep is such an event, so it would never stay asleep.
_POLL_INTERVAL_S = 0.005


@dataclass(slots=True)
class RingBufferConfig:
    buffer_flush_threshold: int
    buffer_flush_time_threshold: float

    # How long the IpcAsyncWriter inside this process holds rows before writing a file. Named
    # apart from buffer_flush_time_threshold above on purpose: that one is the step before,
    # from the event buffer into the writer.
    writer_flush_s: float = 2.0

    output_queue: mp.Queue = field(default_factory=mp.Queue)
    stop_event: mp.Event = field(default_factory=mp.Event)  # ty:ignore[invalid-type-form]

    # Sentinel is put into the queue on shutdown to signal the consumer.
    sentinel: Any = "SENTINEL"
    df_schema: dict[str, Any] = field(default_factory=dict)


class EbpfRingBufferProcess(mp.Process, ABC):
    """
    Generic base for an eBPF ring-buffer producer that:
    - runs in its own Process
    - buffers events into a list[dict]
    - flushes to a Path via a writer function and puts that Path onto a queue
    - emits a sentinel on shutdown
    - enforces at most one live instance per subclass (in the parent interpreter)
    """

    _singletons: ClassVar[dict[type[EbpfRingBufferProcess], EbpfRingBufferProcess]] = {}
    _singletons_lock: ClassVar[threading.Lock] = threading.Lock()

    # Appended to the overflow warning, for producers whose buffer size can be configured.
    _overflow_hint: ClassVar[str] = ""

    def __init__(self, config: RingBufferConfig, *, name: str | None = None, daemon: bool = True) -> None:
        super().__init__(name=name or self.__class__.__name__, daemon=daemon)

        self._ring_config = config

        # ring events buffer (dict mode - legacy)
        self._buffer: list[dict[str, Any]] = []
        # raw bytes buffer (fast mode - subclasses can enable)
        self._raw_buffer: list[Any] = []
        self._use_raw_bytes: bool = False  # subclasses set to True for fast path

        self._buffer_flush_threshold = int(config.buffer_flush_threshold)
        self._buffer_flush_time_threshold = float(config.buffer_flush_time_threshold)
        self._writer_flush_s = float(config.writer_flush_s)
        self._last_flush_time = time.time()

        # queue + lifecycle
        self._queue: mp.Queue = config.output_queue
        self._sentinel = config.sentinel
        self._stop_event: mp.Event = config.stop_event  # ty:ignore[invalid-type-form]
        self._flush_requested = False

        # df schema
        self._df_schema: dict[str, Any] = config.df_schema

        # missing events tracking
        self._missing_events = 0
        self._total_events = 0
        self._last_drop_warning = 0.0

        # created in child process
        self._bpf = None

        # ipc writer
        self._ipc_writer: None | IpcAsyncWriter = None

    # -----------------------------
    # Singleton process management
    # -----------------------------
    @classmethod
    def start_singleton(cls, config: RingBufferConfig, *, daemon: bool = True) -> EbpfRingBufferProcess:
        with cls._singletons_lock:
            existing = cls._singletons.get(cls)
            if existing is not None and existing.is_alive():
                return existing

            inst = cls(config=config, daemon=daemon)
            inst.start()
            cls._singletons[cls] = inst
            return inst

    @classmethod
    def get_singleton(cls) -> EbpfRingBufferProcess | None:
        with cls._singletons_lock:
            inst = cls._singletons.get(cls)
            if inst is not None and inst.is_alive():
                return inst
            return None

    @classmethod
    def stop_singleton(cls) -> None:
        with cls._singletons_lock:
            inst = cls._singletons.get(cls)
        if inst is None:
            return
        inst.stop()
        inst.join(timeout=10)

        with cls._singletons_lock:
            # remove if it's ours and not alive anymore
            cur = cls._singletons.get(cls)
            if cur is inst and not inst.is_alive():
                cls._singletons.pop(cls, None)

    def stop(self) -> None:
        self._stop_event.set()

    # -----------------------------
    # Process entrypoint
    # -----------------------------
    def run(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self._ipc_writer = IpcAsyncWriter(
            self._queue, max_rows=self._buffer_flush_threshold, max_age_s=self._writer_flush_s
        )
        try:
            self._bpf = self._create_bpf()
            self._setup_perf_events()
            self._attach_ringbuf()
            self._poll_loop()
        finally:
            self._stop_monitoring()

    def _poll_loop(self) -> None:
        assert self._bpf is not None
        wait_s = _POLL_INTERVAL_S
        while not self._stop_event.wait(wait_s):
            self._consume()
            # A pass that ended on a full buffer (see _handle_ring_event) left events behind, so
            # the next one starts right after the flush instead of one interval later.
            wait_s = 0 if self._flush_requested else _POLL_INTERVAL_S
            if self._flush_requested:
                self._flush_requested = False
                self._flush_to_queue()

            self._check_missing_events()

    def _consume(self) -> None:
        assert self._bpf is not None
        self._bpf.ring_buffer_consume()

    def _buffered_events(self) -> int:
        return len(self._buffer) + len(self._raw_buffer)

    # -----------------------------
    # Hooks subclasses must provide
    # -----------------------------
    @abstractmethod
    def _create_bpf(self):
        raise NotImplementedError

    @abstractmethod
    def _attach_ringbuf(self) -> None:
        raise NotImplementedError

    def _setup_perf_events(self) -> None:
        return

    @abstractmethod
    def _event_to_record(self, event: Any) -> dict[str, Any]:
        """
        Convert one ring-buffer event object into a record dict for buffering.
        """
        raise NotImplementedError

    def _get_drop_count(self) -> int:
        """
        Optional hook: override if your drops map is not 'rb_drops' or uses different semantics.
        """
        assert self._bpf is not None
        drop_map = self._bpf["rb_drops"]
        k = drop_map.Key(0)
        v = drop_map[k]
        try:
            return int(sum(v))  # per-cpu array
        except TypeError:
            return int(v.value)  # scalar

    # -----------------------------
    # Ringbuf callback + flushing
    # -----------------------------
    def _time_flush_due(self) -> bool:
        now = time.time()
        return (now - self._last_flush_time) >= self._buffer_flush_time_threshold

    def _request_flush_if_needed(self) -> None:
        buffer_len = len(self._raw_buffer) if self._use_raw_bytes else len(self._buffer)
        if buffer_len >= self._buffer_flush_threshold:
            self._flush_requested = True
        elif self._time_flush_due():
            self._flush_requested = True

    def _handle_ring_event(self, ctx, data, size) -> int:
        """
        This is the callback you pass to open_ring_buffer.
        """
        assert self._bpf is not None
        if self._use_raw_bytes:
            # Fast path: just copy raw bytes, defer parsing to flush
            raw = ctypes.string_at(data, size)
            self._raw_buffer.append(raw)
        else:
            # Legacy path: parse each event to dict
            e = self._read_ring_event(data)
            self._buffer.append(self._event_to_record(e))
        self._request_flush_if_needed()

        # ring_buffer_consume() reads for as long as events arrive. Arriving faster than this
        # process takes them, it would never return and nothing would be flushed; the buffer would
        # grow until the OOM killer ends the process. A negative return ends the pass.
        return -1 if self._flush_requested else 0

    def _read_ring_event(self, data):
        assert self._bpf is not None
        return self._bpf["events"].event(data)

    def _flush_to_queue(self) -> None:
        if self._use_raw_bytes:
            self._flush_raw_to_queue()
        else:
            self._flush_dict_to_queue()

    def _flush_dict_to_queue(self) -> None:
        """Legacy path: flush dict records to queue."""
        if not self._buffer:
            self._last_flush_time = time.time()
            return

        # Snapshot then clear first to keep latency low and prevent re-writing old rows
        records = self._buffer
        self._buffer = []
        self._last_flush_time = time.time()

        df = pl.DataFrame(records, schema=self._df_schema, strict=True)
        self._total_events += df.height
        self._ipc_writer.enqueue(df)

    def _flush_raw_to_queue(self) -> None:
        """Fast path: batch-parse raw bytes and flush to queue.

        Subclasses using raw bytes mode must override this method to implement
        efficient batch parsing (e.g., using NumPy structured arrays).
        """
        if not self._raw_buffer:
            self._last_flush_time = time.time()
            return

        # Snapshot then clear
        raw_events = self._raw_buffer
        self._raw_buffer = []
        self._last_flush_time = time.time()

        # Subclasses must implement _parse_raw_events to convert to DataFrame
        df = self._parse_raw_events(raw_events)
        self._total_events += df.height
        self._ipc_writer.enqueue(df)

    def _parse_raw_events(self, raw_events: list[Any]) -> pl.DataFrame:
        """Convert raw bytes to DataFrame. Override in subclass for raw bytes mode."""
        raise NotImplementedError("Subclasses using raw bytes mode must implement _parse_raw_events")

    # -----------------------------
    # Missing events tracking
    # -----------------------------
    def _check_missing_events(self) -> None:
        missing_events = self._get_drop_count()
        if missing_events <= self._missing_events:
            return

        delta = missing_events - self._missing_events
        self._missing_events = missing_events
        logger.debug(f"[{self.name}] Missed events: {delta} (Total missing events {missing_events})")

        now = time.time()
        if now - self._last_drop_warning < _DROP_WARNING_INTERVAL_S:
            return
        self._last_drop_warning = now

        # _total_events only advances on flush, so the events still sitting in the buffer have to
        # be counted here as well - otherwise an early drop reads as a 100 % loss rate.
        buffered = self._buffered_events()
        seen = self._total_events + buffered + missing_events
        logger.warning(
            f"[{self.name}] Ring buffer overflow: {missing_events} of {seen} events dropped by the "
            f"kernel ({100.0 * missing_events / seen:.2f} %). Dropped events cannot be recovered.{self._overflow_hint}"
        )

    # -----------------------------
    # Shutdown
    # -----------------------------
    def _stop_monitoring(self) -> None:
        """
        Flush remaining events, send sentinel, and close queue from producer side.
        """
        try:
            self._flush_to_queue()
            self._ipc_writer.close()
        except Exception:
            # best-effort shutdown
            pass

        # Outside the try above on purpose: this is the only place a run reports its losses, so a
        # failed flush must not swallow it.
        logger.info(
            f"[{self.name}] Events processed: {self._total_events}, dropped in the kernel: {self._missing_events}"
        )

        try:
            self._queue.put(self._sentinel)
        except Exception:
            pass

        try:
            self._queue.close()
            self._queue.join_thread()
        except Exception:
            pass
