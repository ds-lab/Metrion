from __future__ import annotations

import ctypes as ct
import multiprocessing as mp
import os
import queue as q
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, override

import polars as pl
import psutil
from loguru import logger

from metrion.entity.logical_entity import LogicalEntity
from metrion.metrion_rust import BpfProgram  # ty:ignore[unresolved-import]
from metrion.monitoring.data_source.cpu.ring_buff_base import EbpfRingBufferProcess, RingBufferConfig
from metrion.monitoring.data_source.cpu.utils import (
    get_cgroup_v2_path,
    get_pid_starttime_ns,
    read_starttime_from_stat_file_in_ns,
)
from metrion.queue_util import drain_remaining, drain_until_sentinel
from metrion.util import boot_epoch_s, build_cgroup_id_map, read_temp_ipc


# struct task_event_t in logical_entity_registry.bpf.c
class TaskEvent(ct.Structure):
    _fields_ = [
        ("event_type", ct.c_uint32),
        ("tid", ct.c_uint32),
        ("tgid", ct.c_uint32),
        ("tid_start_ns", ct.c_uint64),
        ("tgid_start_ns", ct.c_uint64),
        ("parent_tid", ct.c_uint32),
        ("parent_tgid", ct.c_uint32),
        ("parent_tid_start_ns", ct.c_uint64),
        ("parent_tgid_start_ns", ct.c_uint64),
        ("uid", ct.c_uint32),
        ("gid", ct.c_uint32),
        ("cgroup_id", ct.c_uint64),
        ("old_tid", ct.c_uint32),
        ("__pad", ct.c_uint32),
        ("comm", ct.c_char * 16),
        ("cmdline", ct.c_char * 128),
    ]


class LogicalEntityRegistryBpfProcess(EbpfRingBufferProcess):
    @override
    def _create_bpf(self):
        return BpfProgram("logical_entity_registry")

    @override
    def _attach_ringbuf(self) -> None:
        assert self._bpf is not None
        self._bpf.open_ring_buffer("events", self._handle_ring_event)

    @override
    def _read_ring_event(self, data: bytes) -> TaskEvent:
        return TaskEvent.from_buffer_copy(data)

    @override
    def _event_to_record(self, event: Any) -> dict[str, Any]:
        return {
            "tid": int(event.tid),
            "tgid": int(event.tgid),
            "tid_start_time": int(event.tid_start_ns),
            "tgid_start_time": int(event.tgid_start_ns),
            "parent_tid": int(event.parent_tid),
            "parent_tgid": int(event.parent_tgid),
            "parent_tid_start_time": int(event.parent_tid_start_ns),
            "parent_tgid_start_time": int(event.parent_tgid_start_ns),
            "event_type": int(event.event_type),
            "uid": int(event.uid),
            "gid": int(event.gid),
            "cgroup_id": int(event.cgroup_id),
            "old_tid": int(event.old_tid),  # PID before migration
            "comm": event.comm.decode("utf-8", errors="replace"),
            "cmdline": event.cmdline.decode("utf-8", errors="replace").strip(),
        }


class LogicalEntityRegistry:
    """
    Registry for logical entities (processes/threads).
    """

    _instance = None
    _lock = threading.Lock()

    # Event type constants matching BPF program
    EVENT_TYPE_DATA_ONLY = 0
    EVENT_TYPE_WITH_METADATA = 1

    # An exited thread's last measurements are looked up until ebpf_flush plus the slowest source
    # interval after the exit (6 s with the shipped config), longer when the queues back up.
    _EXIT_GRACE_S = 60.0

    def __new__(cls, *args, **kwargs):
        # Thread-safe singleton construction
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._entities: dict[tuple[int, int], LogicalEntity] = {}  # (tid, start_time_ns) -> Entity
        self._starts_by_tid: dict[int, list[int]] = {}  # tid -> start_time_ns of each entity with it, oldest first
        # (processed at, key (tid, start_time_ns)) of exited threads, oldest first; dropped after _EXIT_GRACE_S
        self._exited: deque[tuple[float, tuple[int, int]]] = deque()
        # ids of the dropped entities, handed out by get_forgotten_ids
        self._forgotten: list[str] = []
        # Keys added or changed since the last get_new_entities(); the /proc scan below fills it too.
        self._pending: set[tuple[int, int]] = set()
        # The data source threads and the orchestrator all call into this one registry.
        self._state_lock = threading.RLock()

        self._queue: mp.Queue = mp.Queue()
        self._stop_event: mp.Event = mp.Event()  # ty:ignore[invalid-type-form]
        # update() runs below in __init__ already, so this has to exist by then
        self._producer_finished = False

        self._config = RingBufferConfig(
            buffer_flush_threshold=100,
            buffer_flush_time_threshold=0.5,
            output_queue=self._queue,
            stop_event=self._stop_event,
            df_schema={
                "tid": pl.UInt64,
                "tgid": pl.UInt64,
                "tid_start_time": pl.UInt64,
                "tgid_start_time": pl.UInt64,
                "parent_tid": pl.UInt64,
                "parent_tgid": pl.UInt64,
                "parent_tid_start_time": pl.UInt64,
                "parent_tgid_start_time": pl.UInt64,
                "event_type": pl.UInt64,
                "uid": pl.UInt64,
                "gid": pl.UInt64,
                "cgroup_id": pl.UInt64,
                "old_tid": pl.UInt64,  # PID before migration (for EXEC events)
                "comm": pl.String,
                "cmdline": pl.String,  # Command line from BPF
            },
        )

        # Boot time for converting ticks to ns
        self._boot_time_s = boot_epoch_s()
        self._boot_time_dt = datetime.fromtimestamp(self._boot_time_s, tz=UTC)
        self._clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

        # REORDERED: Scan /proc FIRST, before BPF starts
        # This eliminates the race window where BPF sees tasks before /proc
        logger.info("Scanning /proc before starting BPF...")
        self._scan_proc()
        self._cgroup_id_map = build_cgroup_id_map()
        logger.info(f"Initial scan found {len(self._entities)} entities")

        # NOW start BPF process (after /proc scan)
        logger.info("Starting BPF process...")
        self._bpf_process = LogicalEntityRegistryBpfProcess.start_singleton(self._config)

        # Process any events that arrived during startup
        logger.info("Processing initial BPF events...")
        self.update()  # Drain queue once

        # debug
        self._no_match_count = 0

    @classmethod
    def get_instance(cls) -> LogicalEntityRegistry:
        return cls()

    def stop(self) -> list[LogicalEntity]:
        with self._state_lock:
            self._bpf_process.stop()

            if self._producer_finished:
                # update() already consumed the sentinel; only in-flight items are left.
                df_paths = drain_remaining(self._queue, name="registry")
            else:
                df_paths, seen = drain_until_sentinel(
                    self._queue, self._config.sentinel, deadline_s=20.0, name="registry"
                )
                if not seen:
                    logger.warning("LogicalEntityRegistry: sentinel not received, continuing anyway")

            for item in df_paths:
                df = read_temp_ipc(item)
                self._process_events(df)

            self._bpf_process.join(timeout=10)
            if self._bpf_process.is_alive():
                logger.warning("Registry BPF process did not exit, killing it")
                self._bpf_process.kill()
                self._bpf_process.join(timeout=5)

            return self.get_new_entities()

    def update(self):
        """Consume events from queue and update registry."""
        with self._state_lock:
            df_paths = []
            max_iterations = 1000  # Prevent infinite loop

            for _ in range(max_iterations):
                try:
                    # Take what is there; the next call picks up what arrives later.
                    item = self._queue.get_nowait()
                except q.Empty:
                    # Queue is empty, done draining
                    break
                if item == self._config.sentinel:
                    self._producer_finished = True
                    break
                df_paths.append(item)

            if len(df_paths) == 0:
                return

            dfs = []
            for item in df_paths:
                df = read_temp_ipc(item)
                dfs.append(df)

            # In arrival order: an EXEC or RENAME changes the entity its NEW created.
            total_df = pl.concat(dfs, how="vertical")
            self._process_events(total_df)

    def _process_events(self, df: pl.DataFrame):
        # EVENT_TYPES: 1=NEW, 2=EXEC, 3=RENAME, 4=EXIT
        for row in df.iter_rows(named=True):
            event_type = row["event_type"]

            if event_type == 1:  # NEW
                self._add_entity(row)
            elif event_type == 2:  # EXEC
                self._update_exec(row)
            elif event_type == 3:  # RENAME
                self._update_rename(row)
            elif event_type == 4:  # EXIT
                self._handle_exit(row)

    def _scan_proc(self):
        """Scan /proc to populate initial entities."""
        logger.info("Scanning /proc for initial entities...")
        access_denied = 0
        no_such = 0
        thread_access_denied = 0
        thread_no_such = 0
        for proc in psutil.process_iter(["pid"]):
            try:
                # Process itself (main thread)
                self._add_psutil_process(proc)

                # Threads
                try:
                    for thread in proc.threads():
                        self._add_psutil_thread(proc, thread)
                except psutil.NoSuchProcess:
                    thread_no_such += 1
                    continue
                except psutil.AccessDenied:
                    thread_access_denied += 1
                    continue

            except psutil.NoSuchProcess:
                no_such += 1
                continue
            except psutil.AccessDenied:
                access_denied += 1
                continue
        logger.info(
            f"Initial scan complete. Registered {len(self._entities)} entities. AccessDenied={access_denied} "
            f"NoSuchProcess={no_such} ThreadAccessDenied={thread_access_denied} ThreadNoSuchProcess={thread_no_such}"
        )

    def _resolve_parent_start_times(
        self, parent_pid: int, parent_tid: int, fallback_parent_pid_start_ns: int, fallback_parent_tid_start_ns: int
    ) -> tuple[int, int]:
        """
        Resolve parent start times by looking up existing entities in the registry.

        Strategy:
        1. First check if the exact (pid, fallback_start_ns) key exists - if so, use it directly
        2. Otherwise, try to find the latest entity for that pid and use its start time
        3. Fall back to the provided estimate if nothing found

        Parameters
        ----------
        parent_pid : int
            Parent process ID (tgid)
        parent_tid : int
            Parent thread ID
        fallback_parent_pid_start_ns : int
            Estimated parent process start time (ns since boot) if not found
        fallback_parent_tid_start_ns : int
            Estimated parent thread start time (ns since boot) if not found

        Returns
        -------
        tuple[int, int]
            (resolved_parent_pid_start_ns, resolved_parent_tid_start_ns)
        """
        resolved_parent_pid_start_ns = fallback_parent_pid_start_ns
        resolved_parent_tid_start_ns = fallback_parent_tid_start_ns

        # Try to resolve parent process
        if parent_pid > 0:
            # First: check if exact key exists
            exact_key = (parent_pid, fallback_parent_pid_start_ns)
            if exact_key in self._entities:
                # Exact match found, use the fallback (it's correct)
                resolved_parent_pid_start_ns = fallback_parent_pid_start_ns
            else:
                # No exact match, try latest entity for this pid
                parent_entity = self._get_latest_entity(parent_pid)
                if parent_entity is not None:
                    parent_start_dt = parent_entity.start_time
                    if parent_start_dt is not None:
                        delta = parent_start_dt - self._boot_time_dt
                        resolved_parent_pid_start_ns = int(delta.total_seconds() * 1_000_000_000)

        # Try to resolve parent thread (may be different from parent_pid for threads)
        if parent_tid > 0 and parent_tid != parent_pid:
            # First: check if exact key exists
            exact_key = (parent_tid, fallback_parent_tid_start_ns)
            if exact_key in self._entities:
                resolved_parent_tid_start_ns = fallback_parent_tid_start_ns
            else:
                parent_tid_entity = self._get_latest_entity(parent_tid)
                if parent_tid_entity is not None:
                    parent_tid_start_dt = parent_tid_entity.start_time
                    if parent_tid_start_dt is not None:
                        delta = parent_tid_start_dt - self._boot_time_dt
                        resolved_parent_tid_start_ns = int(delta.total_seconds() * 1_000_000_000)
        else:
            # parent_tid == parent_pid, use the same resolved value
            resolved_parent_tid_start_ns = resolved_parent_pid_start_ns

        return resolved_parent_pid_start_ns, resolved_parent_tid_start_ns

    def _add_psutil_process(self, proc: psutil.Process):
        pid = proc.pid
        try:
            ppid = proc.ppid()
        except Exception:
            ppid = 0

        # start time: read /proc stat directly (robust + works for kernel threads)
        try:
            start_ns = get_pid_starttime_ns(pid)
            start_dt = self._ns_to_datetime(start_ns)
        except Exception:
            # if we can't get a start time, don't poison the registry with 0
            return

        # optional fields: best-effort
        try:
            name = proc.name()
        except Exception:
            name = "unknown"

        try:
            cmdline = " ".join(proc.cmdline())
        except Exception:
            cmdline = ""

        try:
            u = proc.uids().real
        except Exception:
            u = -1

        try:
            g = proc.gids().real
        except Exception:
            g = -1

        try:
            cgroup = get_cgroup_v2_path(pid)
        except Exception:
            cgroup = "unknown"

        # parent start times: best-effort
        try:
            ppid_start_ns = get_pid_starttime_ns(ppid) if ppid > 0 else 0
        except Exception:
            ppid_start_ns = 0

        # Resolve parent start times from registry if available
        fallback_ppid_start_ns = ppid_start_ns
        resolved_parent_pid_ns, resolved_parent_tid_ns = self._resolve_parent_start_times(
            parent_pid=ppid,
            parent_tid=ppid,
            fallback_parent_pid_start_ns=fallback_ppid_start_ns,
            fallback_parent_tid_start_ns=fallback_ppid_start_ns,
        )

        logical_entity = LogicalEntity(
            native_thread_id=pid,
            native_pid=pid,
            native_parent_pid=ppid,
            native_parent_thread_id=ppid,
            executable_name=name,
            cmdline=cmdline,
            start_time=start_dt,
            pid_start_time=self._ns_to_datetime(start_ns),
            parent_pid_start_time=self._ns_to_datetime(resolved_parent_pid_ns)
            if resolved_parent_pid_ns
            else self._boot_time_dt,
            parent_tid_start_time=self._ns_to_datetime(resolved_parent_tid_ns)
            if resolved_parent_tid_ns
            else self._boot_time_dt,
            resource_group_id=None,
            resource_group_name=cgroup,
            user_id=u,
            group_id=g,
        )

        self._store_entity(logical_entity, pid, start_ns)

    def _add_psutil_thread(self, p: psutil.Process, t: Any):
        tid = t.id
        stat_file = Path(f"/proc/{p.pid}/task/{tid}/stat")
        start_time_ns = read_starttime_from_stat_file_in_ns(stat_file)
        start_ts = self._ns_to_datetime(start_time_ns)

        # Threads share cgroup, uid, gid with process usually
        cgroup_path = get_cgroup_v2_path(p.pid)
        pid_starttime_ns = get_pid_starttime_ns(p.pid)
        ppid = p.ppid()
        try:
            ppid_start_ns = get_pid_starttime_ns(ppid) if ppid > 0 else 0
        except Exception:
            ppid_start_ns = 0

        # Resolve parent start times from registry if available
        resolved_parent_pid_ns, resolved_parent_tid_ns = self._resolve_parent_start_times(
            parent_pid=ppid,
            parent_tid=ppid,
            fallback_parent_pid_start_ns=ppid_start_ns,
            fallback_parent_tid_start_ns=ppid_start_ns,
        )

        entity = LogicalEntity(
            native_thread_id=tid,
            native_pid=p.pid,
            native_parent_pid=ppid,
            native_parent_thread_id=ppid,
            executable_name=p.name(),  # Threads share name usually
            cmdline=" ".join(p.cmdline()),
            start_time=start_ts,
            pid_start_time=self._ns_to_datetime(pid_starttime_ns),
            parent_tid_start_time=self._ns_to_datetime(resolved_parent_tid_ns)
            if resolved_parent_tid_ns
            else self._boot_time_dt,
            parent_pid_start_time=self._ns_to_datetime(resolved_parent_pid_ns)
            if resolved_parent_pid_ns
            else self._boot_time_dt,
            resource_group_id=None,
            resource_group_name=cgroup_path,
            user_id=p.uids().real,
            group_id=p.gids().real,
        )

        # Store
        self._store_entity(entity, tid, start_time_ns)

    def _add_entity(self, row: dict[str, Any]) -> None:
        # Use cmdline from BPF event, fall back to /proc only if empty
        cmdline = row.get("cmdline", "").strip()
        if not cmdline or cmdline == row["comm"]:
            # BPF cmdline is empty or same as comm (fallback case), try /proc
            cmdline = self._get_cmdline(row["tgid"])
        # Final fallback to comm if still empty
        if not cmdline:
            cmdline = row["comm"]

        # Resolve parent start times from registry if available
        resolved_parent_pid_ns, resolved_parent_tid_ns = self._resolve_parent_start_times(
            parent_pid=row["parent_tgid"],
            parent_tid=row["parent_tid"],
            fallback_parent_pid_start_ns=row["parent_tgid_start_time"],
            fallback_parent_tid_start_ns=row["parent_tid_start_time"],
        )

        entity = LogicalEntity(
            native_thread_id=row["tid"],
            native_pid=row["tgid"],
            native_parent_pid=row["parent_tgid"],
            native_parent_thread_id=row["parent_tid"],
            executable_name=row["comm"],
            cmdline=cmdline,
            start_time=self._ns_to_datetime(row["tid_start_time"]),
            pid_start_time=self._ns_to_datetime(row["tgid_start_time"]),
            parent_tid_start_time=self._ns_to_datetime(resolved_parent_tid_ns),
            parent_pid_start_time=self._ns_to_datetime(resolved_parent_pid_ns),
            resource_group_id=row.get("cgroup_id"),
            resource_group_name=self._cgroup_name(row["cgroup_id"]),
            user_id=row.get("uid"),
            group_id=row.get("gid"),
        )

        self._store_entity(entity, row["tid"], row["tid_start_time"])

    def _update_exec(self, row: dict[str, Any]) -> None:
        """
        EXEC: Update existing entity or migrate from old_tid.
        Handles PID migration during exec by looking up old_tid if new tid not found.
        """
        tid = row["tid"]
        old_tid = row["old_tid"]
        tid_start_ns = row["tid_start_time"]

        # Try to find entity by new tid first
        key = (tid, tid_start_ns)
        entity = self._entities.get(key)

        # If not found and old_tid differs, look up by old_tid and migrate
        if entity is None and old_tid != 0 and old_tid != tid:
            old_key = (old_tid, tid_start_ns)
            entity = self._entities.get(old_key)

            if entity is not None:
                # Migrate entity to new tid key
                self._drop_entity(old_key)
                entity._native_thread_id = tid
                entity._thread_id = entity._generate_thread_id()
                self._store_entity(entity, tid, tid_start_ns)

        # If still not found, create new entity
        if entity is None:
            self._add_entity(row)
            return

        # Update metadata (comm, cmdline may have changed during exec)
        entity.executable_name = row["comm"]
        # Use cmdline from BPF event, fall back to /proc only if empty
        cmdline = row.get("cmdline", "").strip()
        if not cmdline or cmdline == row["comm"]:
            cmdline = self._get_cmdline(entity.native_pid)
        if cmdline:
            entity.cmdline = cmdline

        # Ensure entity is stored under correct key
        self._entities[key] = entity
        self._pending.add(key)

    def _update_rename(self, row: dict[str, Any]) -> None:
        """
        RENAME: update comm only (cmdline does NOT change). Identify entity via (tid, tid_start_time) key.
        """
        tid = row["tid"]
        tid_start_ns = row["tid_start_time"]

        # Prefer exact match to avoid PID reuse confusion
        key = (tid, tid_start_ns)
        entity = self._entities.get(key)

        # Fallback: latest entity for tid
        if entity is None:
            return

        # Update comm/executable name
        comm = row["comm"]
        entity.executable_name = comm

        # Use cmdline from BPF event, fall back to /proc only if empty
        cmdline = row.get("cmdline", "").strip()
        if not cmdline or cmdline == row["comm"]:
            cmdline = self._get_cmdline(entity.native_pid)
        if cmdline:
            entity.cmdline = cmdline

        if key is not None:
            self._entities[key] = entity
            self._pending.add(key)

    def _handle_exit(self, row):
        # Only one thread can hold a tid at a time, so every entity with this tid is dead now. That also
        # covers the threads from the /proc scan, whose EXIT carries start 0 because BPF never saw them start.
        now = time.monotonic()
        for start_ns in self._starts_by_tid.get(row["tid"], []):
            self._exited.append((now, (row["tid"], start_ns)))

    def _get_cmdline(self, pid: int) -> str:
        try:
            cmdline_file = Path(f"/proc/{pid}/cmdline")
            with cmdline_file.open() as f:
                content = f.read()
                # cmdline is null-separated
                res = content.replace("\0", " ").strip()
                return res
        except Exception:
            return ""

    def _cgroup_name(self, cgroup_id: int) -> str:
        """Path of the cgroup; the map is rebuilt only for an id it does not know yet."""
        if cgroup_id not in self._cgroup_id_map:
            self._cgroup_id_map = build_cgroup_id_map()
            # Still missing after a fresh walk: a later walk will not find it either.
            self._cgroup_id_map.setdefault(cgroup_id, "unknown")
        return self._cgroup_id_map[cgroup_id]

    def _get_task_starttime_ns(self, tid: int, tgid: int) -> int | None:
        """
        Best-effort lookup of a thread's start time in ns since boot.
        Prefer /proc/<tgid>/task/<tid>/stat, fall back to /proc/<tid>/stat.
        """
        stat_paths = [Path(f"/proc/{tgid}/task/{tid}/stat")]
        if tid != tgid:
            stat_paths.append(Path(f"/proc/{tid}/stat"))

        for stat_path in stat_paths:
            try:
                return read_starttime_from_stat_file_in_ns(stat_path)
            except FileNotFoundError:
                continue
            except Exception:
                return None
        return None

    def _store_entity(self, entity: LogicalEntity, tid: int, start_ns: int):
        key = (tid, start_ns)
        self._entities[key] = entity
        self._pending.add(key)
        starts = self._starts_by_tid.setdefault(tid, [])
        # The /proc scan stores every main thread twice, as process and as thread.
        if start_ns not in starts:
            starts.append(start_ns)

    def _drop_entity(self, key: tuple[int, int]) -> LogicalEntity | None:
        tid, start_ns = key
        entity = self._entities.pop(key, None)
        starts = self._starts_by_tid.get(tid, [])
        if start_ns in starts:
            starts.remove(start_ns)
        if not starts:
            self._starts_by_tid.pop(tid, None)
        return entity

    def _forget_exited(self) -> None:
        """Drop the threads that exited more than _EXIT_GRACE_S ago."""
        horizon = time.monotonic() - self._EXIT_GRACE_S
        while self._exited and self._exited[0][0] < horizon:
            _, key = self._exited.popleft()
            entity = self._drop_entity(key)
            if entity is not None:
                self._forgotten.append(entity.id)

    def _get_latest_entity(self, tid: int) -> LogicalEntity | None:
        starts = self._starts_by_tid.get(tid)
        if not starts:
            return None
        return self._entities[(tid, starts[-1])]

    def _find_key(self, pid: int, start_ns: int) -> tuple[int, int] | None:
        """Key of this thread: the exact start, else the nearest start of the pid within six clock ticks."""
        if (pid, start_ns) in self._entities:
            return (pid, start_ns)
        tolerance_ns = 6 * (1_000_000_000 // self._clk_tck)
        # serach for 190
        # 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400
        # -90, -70, -50, -30, -10, 10,   30,  50,  70,  90, 110, 130, 150, 170, 190, 210
        #
        nearest = min(self._starts_by_tid.get(pid, []), key=lambda start: abs(start - start_ns), default=None)
        if nearest is None or abs(nearest - start_ns) > tolerance_ns:
            return None
        return (pid, nearest)

    def _ns_to_datetime(self, ns: int) -> datetime:
        # ns is since boot
        return self._boot_time_dt + timedelta(seconds=ns / 1e9)

    def _ticks_to_datetime(self, ticks: int) -> datetime:
        return self._ns_to_datetime(int(ticks * 1e9 / self._clk_tck))

    def get_logical_entity(
        self,
        pid: int,
        start_time: int | datetime,
    ) -> LogicalEntity | None:
        """
        Lookup by canonical identity (pid/tid, start_ns_since_boot).

        - If start_time is datetime: interpret as UTC wall-clock and convert to ns since boot.
        - If start_time is int: interpret as ns since boot.

        Lookup strategy:
        1) Exact: (pid, start_ns)
        2) Fuzzy: the nearest start of that pid within six clock ticks
        A start_time before boot falls back to the latest entity for pid.
        """
        with self._state_lock:
            # ---- normalize start_time -> ns since boot ----
            if isinstance(start_time, datetime):
                dt = start_time
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                else:
                    dt = dt.astimezone(UTC)

                delta = dt - self._boot_time_dt
                start_ns = int(delta.total_seconds() * 1_000_000_000)
            else:
                start_ns = int(start_time)

            if start_ns < 0:
                logger.warning(f"Invalid start_time {start_time} for pid {pid}, using latest entity")
                return self._get_latest_entity(pid)

            key = self._find_key(pid, start_ns)
            if key is None:
                self._no_match_count += 1
                return None
            return self._entities[key]

    def get_all_entities_df(self) -> pl.DataFrame:
        """Return all entities as a DataFrame."""

        with self._state_lock:
            data = []
            for entity in self._entities.values():
                data.append(entity._to_df())

            return pl.concat([df for df in data], how="vertical")

    def _add_entity_from_bpf_metadata(self, pid: int, pid_start: int, meta: dict[str, Any]) -> LogicalEntity:
        # Resolve parent start times from registry if available
        resolved_parent_pid_ns, resolved_parent_tid_ns = self._resolve_parent_start_times(
            parent_pid=meta["parent_tgid"],
            parent_tid=meta["parent_tid"],
            fallback_parent_pid_start_ns=meta["start_time_parent_tgid"],
            fallback_parent_tid_start_ns=meta["start_time_parent"],
        )

        le = LogicalEntity(
            native_thread_id=pid,
            native_pid=meta["tgid"],
            native_parent_pid=meta["parent_tgid"],
            native_parent_thread_id=meta["parent_tid"],
            executable_name=meta["comm"],
            cmdline=meta["cmdline"] or meta["comm"],
            start_time=self._ns_to_datetime(pid_start),
            pid_start_time=self._ns_to_datetime(meta["start_time_tgid"]),
            parent_tid_start_time=self._ns_to_datetime(resolved_parent_tid_ns),
            parent_pid_start_time=self._ns_to_datetime(resolved_parent_pid_ns),
            resource_group_id=meta.get("cgroup_id"),
            user_id=meta.get("uid"),
            group_id=meta.get("gid"),
        )

        self._store_entity(le, pid, pid_start)
        return le

    def _register_entities_from_inline_metadata(self, df: pl.DataFrame) -> int:
        """
        Register logical entities from inline metadata embedded in ring buffer events.

        This is the key to zero-loss matching: events with event_type=1 contain
        full process metadata. We create LogicalEntity objects inline before
        attempting to match.

        Returns the count of entities created from inline metadata.
        """
        with self._state_lock:
            # Filter to events with inline metadata
            metadata_events = df.filter(pl.col("event_type") == self.EVENT_TYPE_WITH_METADATA)

            if metadata_events.is_empty():
                return 0

            # Get unique (pid, pid_start) combinations with metadata
            unique_metadata = metadata_events.select(
                [
                    "pid",
                    "pid_start",
                    "tgid",
                    "tgid_start",
                    "parent_tid",
                    "parent_tgid",
                    "parent_tid_start",
                    "parent_tgid_start",
                    "cgroup_id",
                    "uid",
                    "gid",
                    "comm",
                    "cmdline",
                ]
            ).unique(subset=["pid", "pid_start"])

            created_count = 0
            for row in unique_metadata.iter_rows(named=True):
                pid = row["pid"]
                pid_start = row["pid_start"]

                # Skip if entity already exists in registry
                existing = self.get_logical_entity(pid=pid, start_time=pid_start)
                if existing is not None:
                    continue

                # Build metadata dict and register entity
                meta = {
                    "tgid": row["tgid"],
                    "start_time": pid_start,
                    "start_time_tgid": row["tgid_start"],
                    "parent_tid": row["parent_tid"],
                    "parent_tgid": row["parent_tgid"],
                    "start_time_parent": row["parent_tid_start"],
                    "start_time_parent_tgid": row["parent_tgid_start"],
                    "cgroup_id": row["cgroup_id"],
                    "uid": row["uid"],
                    "gid": row["gid"],
                    "comm": row["comm"] or "",
                    "cmdline": row["cmdline"] or row["comm"] or "",
                }
                self._add_entity_from_bpf_metadata(pid, pid_start, meta)
                created_count += 1

            return created_count

    def update_entity_metadata_from_df(self, df: pl.DataFrame) -> int:
        """
        Update existing entities with better metadata from monitoring dataframes.

        This is called by sched_switch and tcp_monitor to enrich entities that
        may have been created with incomplete metadata (e.g., from inline metadata
        where comm/cmdline was empty or from /proc scan of short-lived processes).

        The DataFrame is expected to have at minimum:
        - pid: int (thread ID)
        - pid_start: int (start time in ns since boot)

        And optionally for enrichment:
        - comm: str (process command name from BPF)
        - cmdline: str (command line from BPF)
        - tgid: int (process group ID)
        - tgid_start: int (process group start time)
        - cgroup_id: int
        - uid: int
        - gid: int

        Returns the count of entities that were updated with new metadata.
        """
        with self._state_lock:
            if df.is_empty():
                return 0

            # Check what columns are available for enrichment
            available_cols = set(df.columns)
            has_comm = "comm" in available_cols
            has_cmdline = "cmdline" in available_cols

            if not has_comm and not has_cmdline:
                return 0  # Nothing to enrich with

            # Get unique (pid, pid_start) combinations that have metadata to offer
            select_cols = ["pid", "pid_start"]
            if has_comm:
                select_cols.append("comm")
            if has_cmdline:
                select_cols.append("cmdline")

            # Filter to rows with non-empty comm or cmdline
            filter_expr = pl.lit(False)
            if has_comm:
                filter_expr = filter_expr | (pl.col("comm").is_not_null() & (pl.col("comm") != ""))
            if has_cmdline:
                filter_expr = filter_expr | (pl.col("cmdline").is_not_null() & (pl.col("cmdline") != ""))

            enrichment_df = df.select(select_cols).filter(filter_expr).unique(subset=["pid", "pid_start"])

            if enrichment_df.is_empty():
                return 0

            updated_count = 0

            for row in enrichment_df.iter_rows(named=True):
                pid = row["pid"]
                pid_start = row["pid_start"]
                comm = row.get("comm", "") or ""
                cmdline = row.get("cmdline", "") or ""

                # Skip if no enrichment data
                if not comm and not cmdline:
                    continue

                # Find the entity
                key = self._find_key(pid, pid_start)
                if key is None:
                    continue
                entity = self._entities[key]

                # Check if entity needs enrichment
                needs_update = False
                current_name = entity.executable_name or ""
                current_cmdline = entity.cmdline or ""

                # Update executable_name if current is empty/None and we have comm
                if comm and (not current_name or current_name == "unknown"):
                    entity.executable_name = comm
                    needs_update = True

                # Update cmdline if current is empty/None/same-as-comm and we have better cmdline
                if cmdline and cmdline != comm:
                    # We have a real cmdline (not just comm)
                    if not current_cmdline or current_cmdline == current_name or current_cmdline == "unknown":
                        entity.cmdline = cmdline
                        needs_update = True
                elif comm and (not current_cmdline or current_cmdline == "unknown"):
                    # At least use comm as cmdline fallback
                    entity.cmdline = comm
                    needs_update = True

                if needs_update:
                    self._pending.add(key)
                    updated_count += 1

            return updated_count

    def get_new_entities(self) -> list[LogicalEntity]:
        """Return all entities that have been added or updated since the last call.

        Afterwards drops the threads that exited more than _EXIT_GRACE_S ago, so every thread is
        reported before it is forgotten.
        """
        with self._state_lock:
            pending, self._pending = self._pending, set()
            # An EXEC that moves an entity to a new tid drops its old key.
            new_entities = [self._entities[key] for key in pending if key in self._entities]
            self._forget_exited()
            return new_entities

    def get_forgotten_ids(self) -> list[str]:
        """Return the ids of the entities dropped since the last call."""
        with self._state_lock:
            forgotten, self._forgotten = self._forgotten, []
            return forgotten
