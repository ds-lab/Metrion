"""Benchmark: the per-event Python cost of the eBPF ring-buffer consumer.

`EbpfRingBufferProcess._handle_ring_event` is invoked by BCC **once per
sched_switch event**. On a busy 64-core machine that is 10^4..10^6 events per
second. This measures what that costs before any of METRION's own work happens,
and compares it with the two realistic alternatives:

  * batched consumption (one Python call per N events)
  * in-kernel pre-aggregation (fewer events reach user space at all)

It also does the ring-buffer bandwidth arithmetic for the current 328-byte
`sched_event_v2`, whose 192 metadata bytes are zero on ~98% of events.

No BPF or root needed: the callback body is replayed against a ctypes buffer of
the same size, which is what dominates the cost.

Run:
    python bench/bench_ringbuf_callback.py --events 500000 --rate 50000
"""

from __future__ import annotations

import argparse
import ctypes
import time

from _common import Bench

EVENT_SIZE_CURRENT = 328  # struct sched_event_v2
EVENT_SIZE_DATA_ONLY = 136  # monitoring payload without the inline metadata block
RINGBUF_BYTES = 32 * 1024 * 1024  # ebpf.ringbuf_mib default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=500_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rate", type=int, default=50_000, help="assumed events/s for the projection")
    parser.add_argument("--batch", type=int, default=256, help="events per batched callback")
    args = parser.parse_args()

    n = args.events
    backing = ctypes.create_string_buffer(EVENT_SIZE_CURRENT)
    ptr = ctypes.addressof(backing)

    bench = Bench("eBPF ring-buffer consumption in Python", rows=n, repeats=args.repeats)

    def current() -> None:
        """_handle_ring_event with _use_raw_bytes=True, called once per event."""
        raw_buffer: list[bytes] = []
        threshold = 10_000
        last_flush = time.time()
        flush_requested = False
        for _ in range(n):
            raw = ctypes.string_at(ptr, EVENT_SIZE_CURRENT)
            raw_buffer.append(raw)
            # _request_flush_if_needed
            if len(raw_buffer) >= threshold:
                flush_requested = True
            elif (time.time() - last_flush) >= 1.0:
                flush_requested = True
            if flush_requested:
                flush_requested = False
                raw_buffer = []
                last_flush = time.time()

    def without_time_check() -> None:
        """Same, but the wall-clock check moved out of the per-event path."""
        raw_buffer: list[bytes] = []
        threshold = 10_000
        append = raw_buffer.append
        string_at = ctypes.string_at
        for _ in range(n):
            append(string_at(ptr, EVENT_SIZE_CURRENT))
            if len(raw_buffer) >= threshold:
                raw_buffer = []
                append = raw_buffer.append

    def batched() -> None:
        """One Python call per `--batch` events (libbpf ring_buffer__consume style)."""
        chunk = ctypes.create_string_buffer(EVENT_SIZE_CURRENT * args.batch)
        chunk_ptr = ctypes.addressof(chunk)
        raw_buffer: list[bytes] = []
        for _ in range(0, n, args.batch):
            raw_buffer.append(ctypes.string_at(chunk_ptr, EVENT_SIZE_CURRENT * args.batch))
            if len(raw_buffer) * args.batch >= 10_000:
                raw_buffer = []

    def batched_small_event() -> None:
        """Batched consumption of the proposed 136-byte data-only event."""
        chunk = ctypes.create_string_buffer(EVENT_SIZE_DATA_ONLY * args.batch)
        chunk_ptr = ctypes.addressof(chunk)
        raw_buffer: list[bytes] = []
        for _ in range(0, n, args.batch):
            raw_buffer.append(ctypes.string_at(chunk_ptr, EVENT_SIZE_DATA_ONLY * args.batch))
            if len(raw_buffer) * args.batch >= 10_000:
                raw_buffer = []

    bench.run("current: one Python callback per event", current)
    bench.run("cheap fix: hoist time.time() out of the loop", without_time_check)
    bench.run(f"batched: one call per {args.batch} events", batched)
    bench.run("batched + 136B data-only event", batched_small_event)
    bench.report()

    per_event_us = bench.results[0].best / n * 1e6
    print("=== projection ===")
    print("NOTE: this is a LOWER BOUND. It replays the body of _handle_ring_event but not")
    print("      BCC's ctypes CFUNCTYPE trampoline, which adds roughly 1-3 us per event on")
    print("      top. Measure the real thing with the perf recipe in bench/README.md.")
    print(f"per-event callback cost:              {per_event_us:8.3f} us")
    for rate in (args.rate, args.rate * 4, args.rate * 20):
        share = per_event_us * rate / 1e6 * 100
        print(f"at {rate:>9,} events/s:              {share:8.2f} % of one core, callback only")
    print()

    print("=== ring-buffer bandwidth (ebpf.ringbuf_mib: 32) ===")
    for size, label in ((EVENT_SIZE_CURRENT, "current 328 B event"), (EVENT_SIZE_DATA_ONLY, "proposed 136 B event")):
        capacity = RINGBUF_BYTES // size
        for rate in (args.rate, args.rate * 4):
            fill_ms = capacity / rate * 1000
            mib_s = size * rate / 1024 / 1024
            print(
                f"{label:<22} capacity {capacity:>9,} events | "
                f"at {rate:>8,}/s: {mib_s:6.1f} MiB/s, buffer full in {fill_ms:6.1f} ms"
            )
    print()
    print("The poll loop drains the buffer every 5 ms (_POLL_INTERVAL_S), plus")
    print("however long the previous drain took. Compare that with 'buffer full in' above:")
    print("if the fill time is close to or below the poll period, events are dropped.")


if __name__ == "__main__":
    main()
