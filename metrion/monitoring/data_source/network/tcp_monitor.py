"""TCP network traffic monitor using eBPF ring buffer.

This module provides a TCP/UDP traffic monitor that uses eBPF ring buffer
to capture network events at the kernel level, collecting per-process
traffic data with IPv4 and IPv6 support.
"""

import ipaddress
import multiprocessing as mp
import socket
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, override

import polars as pl
import psutil
from bcc import BPF  # ty:ignore[unresolved-import]
from loguru import logger

from metrion.data_record import DataRecord
from metrion.entity.communication_endpoint import CommunicationEndpoint, NetworkAddress, NetworkTransport
from metrion.entity.entity import Entity
from metrion.entity.logical_entity_registry import LogicalEntityRegistry
from metrion.entity.network_interface import NetworkInterface, NetworkInterfaceRegistry
from metrion.metric import NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED, NETWORK_HOST_PID_TOTAL_BYTES_SENT, Metric
from metrion.monitoring.data_source.cpu.ring_buff_base import EbpfRingBufferProcess, RingBufferConfig
from metrion.monitoring.data_source.data_source_base import DataSource
from metrion.monitoring.data_source.util import _validate_bpf_clock_domain
from metrion.util import check_entity_type, read_temp_ipc


def _ipv4_to_string(addr: int) -> str:
    """Convert a 32-bit IPv4 address to dotted decimal string."""
    try:
        return socket.inet_ntoa(addr.to_bytes(4, byteorder="little"))
    except (OSError, OverflowError):
        return "0.0.0.0"


def _ipv6_to_string(addr_bytes: bytes) -> str:
    """Convert 16 bytes to an IPv6 address string."""
    try:
        ipv6 = ipaddress.IPv6Address(addr_bytes)
        if ipv6.ipv4_mapped is not None:
            return str(ipv6.ipv4_mapped)
        return ipv6.exploded if ipv6.compressed == "::" else ipv6.compressed
    except (OSError, ValueError):
        return "::"


# DataFrame schema for TCP events
TCP_EVENT_SCHEMA = {
    # Core event data
    "pid": pl.UInt32,
    "tgid": pl.UInt32,
    "cgroup_id": pl.UInt64,
    "pid_start": pl.UInt64,  # ns since boot
    "tgid_start": pl.UInt64,  # ns since boot
    "time_ns": pl.UInt64,  # ns since boot
    "socket_id": pl.UInt64,
    "bytes": pl.UInt32,
    "ifindex": pl.UInt32,
    "family": pl.UInt16,
    "src_addr": pl.String,
    "dst_addr": pl.String,
    "src_port": pl.UInt16,
    "dst_port": pl.UInt16,
    "direction": pl.UInt8,
    "transport_type": pl.UInt8,
    # Event type flag for inline metadata
    "event_type": pl.UInt8,
    # Inline metadata fields (nullable for data-only events)
    "parent_tid": pl.UInt32,
    "parent_tgid": pl.UInt32,
    "parent_tid_start": pl.UInt64,
    "parent_tgid_start": pl.UInt64,
    "uid": pl.UInt32,
    "gid": pl.UInt32,
    "comm": pl.String,
    "cmdline": pl.String,
}


class TCPMonitorBpfProcess(EbpfRingBufferProcess):
    """eBPF ring buffer process for TCP/UDP traffic monitoring with inline metadata."""

    _boot_time: datetime = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)

    @override
    def _create_bpf(self) -> BPF:
        """Load and compile the eBPF v2 program with inline metadata support."""
        base_dir = Path(__file__).resolve().parent
        # Use v2 BPF program with inline metadata for zero-loss entity matching
        c_file_path = base_dir / "tcp_monitor.c"

        bpf = BPF(src_file=str(c_file_path))

        bpf.attach_kprobe(event="tcp_cleanup_rbuf", fn_name="trace_tcp_cleanup_rbuf")
        bpf.attach_kprobe(event="tcp_sendmsg", fn_name="trace_tcp_sendmsg")
        bpf.attach_kprobe(event="udp_sendmsg", fn_name="trace_udp_sendmsg")
        bpf.attach_kprobe(event="udp_recvmsg", fn_name="trace_udp_recvmsg")

        return bpf

    @override
    def _attach_ringbuf(self) -> None:
        """Attach ring buffer callback."""
        assert self._bpf is not None
        self._bpf["events"].open_ring_buffer(self._handle_ring_event)

    @override
    def _event_to_record(self, event: Any) -> dict[str, Any]:
        """Convert eBPF event to dictionary for DataFrame with inline metadata."""
        family = int(event.family)
        if family == socket.AF_INET:
            src_addr = _ipv4_to_string(event.saddr)
            dst_addr = _ipv4_to_string(event.daddr)
        else:
            src_addr = _ipv6_to_string(bytes(event.ipv6_rcv_saddr))
            dst_addr = _ipv6_to_string(bytes(event.ipv6_daddr))

        record = {
            "pid": int(event.pid),
            "tgid": int(event.tgid),
            "cgroup_id": int(event.cgroup_id),
            "pid_start": int(event.pid_start),
            "tgid_start": int(event.tgid_start),
            "time_ns": int(event.time_ns),
            "socket_id": int(event.socket_id),
            "bytes": int(event.bytes),
            "ifindex": int(event.ifindex),
            "family": family,
            "src_addr": src_addr,
            "dst_addr": dst_addr,
            "src_port": int(event.sport),
            "dst_port": int(event.dport),
            "direction": int(event.direction),
            "transport_type": int(event.transport_type),
            "event_type": int(event.event_type),
        }

        # Include inline metadata when event_type indicates it's present
        if event.event_type == LogicalEntityRegistry.EVENT_TYPE_WITH_METADATA:
            record.update(
                {
                    "parent_tid": int(event.parent_tid),
                    "parent_tgid": int(event.parent_tgid),
                    "parent_tid_start": int(event.parent_tid_start),
                    "parent_tgid_start": int(event.parent_tgid_start),
                    "uid": int(event.uid),
                    "gid": int(event.gid),
                    "comm": event.comm.decode("utf-8", errors="replace").rstrip("\x00"),
                    "cmdline": event.cmdline.decode("utf-8", errors="replace").rstrip("\x00").strip(),
                }
            )
        else:
            record.update(
                {
                    "parent_tid": None,
                    "parent_tgid": None,
                    "parent_tid_start": None,
                    "parent_tgid_start": None,
                    "uid": None,
                    "gid": None,
                    "comm": None,
                    "cmdline": None,
                }
            )

        return record


class TCPDataSource(DataSource):
    """Data source for TCP/UDP network traffic monitoring.

    Uses an eBPF ring buffer process to collect network traffic metrics
    per-process with IPv4/IPv6 support.
    """

    _entity_metric_mapping = {
        NetworkInterface: {NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED, NETWORK_HOST_PID_TOTAL_BYTES_SENT}
    }

    _instance = None
    _boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=UTC)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        entities: Sequence[NetworkInterface],
        interval: float | None = None,
    ) -> None:
        if hasattr(self, "_initialized"):
            return

        super().__init__(entities, interval=interval)
        check_entity_type(entities, expected_type=NetworkInterface)

        self._logical_entity_registry = LogicalEntityRegistry.get_instance()

        # Caches for measurements and entities
        self._cache_received: list[pl.DataFrame] = []
        self._cache_send: list[pl.DataFrame] = []
        self._cache_entities: list[Entity] = []
        self._cache_communication_endpoints: list[CommunicationEndpoint] = []

        # Ring buffer config
        self._ring_config = RingBufferConfig(
            buffer_flush_threshold=10000,
            buffer_flush_time_threshold=1.0,
            output_queue=mp.Queue(),
            stop_event=mp.Event(),
            sentinel="TCP_SENTINEL",
            df_schema=TCP_EVENT_SCHEMA,
        )

        # Start the eBPF ring buffer process
        self._bpf_process = TCPMonitorBpfProcess.start_singleton(self._ring_config, daemon=True)

        self._boot_time_s = psutil.boot_time()
        self._clock_validated = False
        self._initialized = True

    def _get_raw_events(self) -> pl.DataFrame:
        """Get raw events from the ring buffer queue."""
        frames = []
        while True:
            try:
                item = self._ring_config.output_queue.get(block=True, timeout=0.1)
                if item == self._ring_config.sentinel:
                    break
                # item is a path to IPC file
                df = read_temp_ipc(item)
                frames.append(df)
            except Exception:
                break

        if not frames:
            return pl.DataFrame(schema=TCP_EVENT_SCHEMA)

        return pl.concat(frames, how="vertical")

    def _add_logical_entity_id_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add logical entity ID using inline metadata + LogicalEntityRegistry.

        Uses a two-tier matching strategy:
        1. INLINE METADATA: Extract metadata from events with event_type=1 and
           register entities before matching (zero-loss path).
        2. REGISTRY LOOKUP: Check the LogicalEntityRegistry for known entities.
        """
        # Step 1: Register entities from inline metadata (zero-loss path)
        self._logical_entity_registry._register_entities_from_inline_metadata(df)

        # Step 1b: Update existing entities with better metadata (enrichment path)
        self._logical_entity_registry.update_entity_metadata_from_df(df)

        # Step 2: Update registry (may have entities from sched_switch or task_newtask)
        self._logical_entity_registry.update()

        # Step 3: Match all (pid, pid_start) combinations to entity IDs
        pids_start_times = df.select([pl.col("pid"), pl.col("pid_start")]).unique().rows()

        entity_id_list = []
        pid_list = []
        start_time_list = []
        unmatched = []

        for pid, start_time_ns in pids_start_times:
            entity = self._logical_entity_registry.get_logical_entity(pid=pid, start_time=start_time_ns)
            if entity is not None:
                entity_id_list.append(entity.id)
                pid_list.append(pid)
                start_time_list.append(start_time_ns)
                self._cache_entities.append(entity)
            else:
                unmatched.append((pid, start_time_ns))

        if unmatched:
            logger.warning(f"TCP: Failed to match {len(unmatched)} entities. Examples: {unmatched[:5]}")

        if not entity_id_list:
            return df.with_columns(pl.lit(None).alias("logical_entity_id").cast(pl.String))

        entity_df = pl.DataFrame(
            {
                "pid": pid_list,
                "pid_start": start_time_list,
                "logical_entity_id": entity_id_list,
            }
        )

        result = df.join(entity_df, on=["pid", "pid_start"], how="left")

        return result

    def _add_physical_entity_id(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add physical entity ID (NIC) based on ifindex."""
        return df.with_columns(
            pl.col("ifindex")
            .map_elements(lambda x: NetworkInterfaceRegistry().get_id_by_ifindex(x), return_dtype=pl.String)
            .alias("physical_entity_id")
        ).filter(pl.col("physical_entity_id").is_not_null())

    def _create_communication_endpoints(self, df: pl.DataFrame) -> tuple[pl.DataFrame, list[CommunicationEndpoint]]:
        """Create communication endpoints from traffic data."""
        new_endpoints: dict[str, CommunicationEndpoint] = {}

        src_endpoint_ids = []
        dst_endpoint_ids = []

        for row in df.iter_rows(named=True):
            transport = NetworkTransport.TCP if row["transport_type"] == 0 else NetworkTransport.UDP

            # Source endpoint (local)
            src_addr = NetworkAddress(
                ip_address=row["src_addr"],
                port=row["src_port"],
                transport=transport,
            )
            src_ep = CommunicationEndpoint(network_address=src_addr)
            if src_ep.id not in new_endpoints:
                new_endpoints[src_ep.id] = src_ep
            src_endpoint_ids.append(src_ep.id)

            # Destination endpoint (remote)
            dst_addr = NetworkAddress(
                ip_address=row["dst_addr"],
                port=row["dst_port"],
                transport=transport,
            )
            dst_ep = CommunicationEndpoint(network_address=dst_addr)
            if dst_ep.id not in new_endpoints:
                new_endpoints[dst_ep.id] = dst_ep
            dst_endpoint_ids.append(dst_ep.id)

        df = df.with_columns(
            pl.Series("src_endpoint", src_endpoint_ids),
            pl.Series("dst_endpoint", dst_endpoint_ids),
        )

        return df, list(new_endpoints.values())

    def _create_measurement_df(self, df: pl.DataFrame, metric_id: str) -> pl.DataFrame:
        """Create measurement DataFrame from traffic data."""
        res = df.select(
            [
                pl.lit(metric_id).alias("metric_id"),
                pl.col("physical_entity_id"),
                pl.col("logical_entity_id"),
                pl.col("bytes").alias("value"),
                (pl.lit(self._boot_time) + pl.col("time_ns").cast(pl.Duration("ns"))).alias("start_ts"),
                (pl.lit(self._boot_time) + pl.col("time_ns").cast(pl.Duration("ns"))).alias("end_ts"),
                pl.col("src_endpoint"),
                pl.col("dst_endpoint"),
            ]
        )

        return res

    def _validate_bpf_clock_domain(self, df: pl.DataFrame) -> None:
        """One-time check: compare converted BPF timestamp against wall clock."""
        if self._clock_validated or df.is_empty():
            return
        res = _validate_bpf_clock_domain(df, col_name="time_ns")
        self._clock_validated = True if res else False

    def _build_cache(self) -> None:
        """Build cache from eBPF ring buffer data."""
        raw_df = self._get_raw_events()

        if raw_df.is_empty():
            return

        self._validate_bpf_clock_domain(raw_df)

        # Filter valid start times and add logical entity IDs
        valid_df = raw_df.filter(pl.col("pid_start") > 0)
        if valid_df.is_empty():
            return

        valid_df = self._add_logical_entity_id_to_df(valid_df)
        valid_df = valid_df.filter(pl.col("logical_entity_id").is_not_null())

        if valid_df.is_empty():
            return

        # Split by direction
        send_df = valid_df.filter(pl.col("direction") == 1)
        recv_df = valid_df.filter(pl.col("direction") == 0)

        # Add physical entity IDs (filter out rows without valid NIC)
        send_df = self._add_physical_entity_id(send_df)
        recv_df = self._add_physical_entity_id(recv_df)

        # Create communication endpoints
        if not send_df.is_empty():
            send_df, send_endpoints = self._create_communication_endpoints(send_df)
            self._cache_communication_endpoints.extend(send_endpoints)
            self._cache_send.append(self._create_measurement_df(send_df, NETWORK_HOST_PID_TOTAL_BYTES_SENT.name))

        if not recv_df.is_empty():
            recv_df, recv_endpoints = self._create_communication_endpoints(recv_df)
            self._cache_communication_endpoints.extend(recv_endpoints)
            self._cache_received.append(
                self._create_measurement_df(recv_df, NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED.name)
            )

    @override
    def get_data(self, metrics: list) -> DataRecord:
        """Get collected data for the specified metrics."""
        self._build_cache()

        measurements = []
        for metric in metrics:
            if metric.id == NETWORK_HOST_PID_TOTAL_BYTES_RECEIVED.id:
                measurements.extend(self._cache_received)
            elif metric.id == NETWORK_HOST_PID_TOTAL_BYTES_SENT.id:
                measurements.extend(self._cache_send)

        entities = self._cache_entities + self._cache_communication_endpoints

        # Clear caches
        self._cache_received = []
        self._cache_send = []
        self._cache_entities = []
        self._cache_communication_endpoints = []

        return DataRecord(
            measurements=measurements,
            entities=entities,
        )

    @override
    def clean_up(self) -> None:
        """Clean up resources."""
        pass

    @override
    def is_applicable(self, entity: Entity) -> bool:
        """Check if the entity is applicable to this data source."""
        return isinstance(entity, NetworkInterface)

    @override
    def stop(self, metrics: list[Metric]) -> DataRecord:
        """Stop monitoring and gather final data."""
        TCPMonitorBpfProcess.stop_singleton()

        # Drain remaining events
        self._build_cache()

        return self.get_data(metrics)
