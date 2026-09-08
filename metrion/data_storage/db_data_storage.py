from collections.abc import Sequence
from contextlib import contextmanager

import polars as pl
from duckdb_engine import insert
from loguru import logger
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import registry, sessionmaker
from sqlalchemy.types import String as StringDB
from sqlalchemy.types import TypeDecorator
from tqdm import tqdm

from metrion.data_record import DataRecord
from metrion.data_storage.i_data_storage import DataStorageI
from metrion.entity.communication_endpoint import CommunicationEndpoint, NetworkTransport
from metrion.entity.core import Core
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.entity import Entity
from metrion.entity.gpu import GPU
from metrion.entity.host import Host as HostEntity
from metrion.entity.host import HostType
from metrion.entity.logical_entity import LogicalEntity
from metrion.entity.metadata import Metadata
from metrion.entity.network_interface import NetworkInterface
from metrion.measurement import CommunicationMeasurement, Measurement
from metrion.metric import Metric
from metrion.util import _add_id_to_measurement_df


def prefer_text(new_expr, old_expr):
    return func.coalesce(func.nullif(new_expr, ""), old_expr)


class CommaSeparatedIntegers(TypeDecorator):
    impl = StringDB
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Handle single integer case
        if isinstance(value, int):
            return str(value)
        # Convert list of ints to comma-separated string.
        return ",".join(str(item) for item in value)

    def process_result_value(self, value, dialect):
        if value is None or value == "":
            return []
        # Convert comma-separated string back to a list of ints.
        return [int(item) for item in value.split(",")]


class SQLDatabaseStorage(DataStorageI):
    def __init__(self, db_url: str):
        self._count = 0
        self._db_url = db_url
        # Create a new SQLAlchemy engine.
        self.engine = create_engine(db_url)
        # Create a configured "Session" class.
        self.Session = sessionmaker(bind=self.engine)
        # Create all tables for the provided models.
        self.registry = registry()
        self._create_tables()
        self.save_all_metrics()

    @contextmanager
    def session_scope(self):
        session = self.Session()
        try:
            # set memory limit
            # session.execute(text("SET memory_limit = '8GB';"))
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _create_tables(self):
        ###############
        #    Entity   #
        ###############

        self._entity_table = Table(
            "entity",
            self.registry.metadata,
            Column("id", String, primary_key=True),
            Column("parent_id", String, nullable=True),
            Column("entity_type", String(50)),
        )

        # Measurement
        self._measurement_table = Table(
            "measurement",
            self.registry.metadata,
            Column("id", BigInteger, primary_key=True, autoincrement=False),
            Column("value", Float),
            Column("metric_id", String),
            Column("physical_entity_id", String),
            Column("logical_entity_id", String),
            Column("start_ts", DateTime),
            Column("end_ts", DateTime),
        )

        # Communication Measurement
        self._communication_measurement_table = Table(
            "communication_measurement",
            self.registry.metadata,
            Column("measurement_id", BigInteger, primary_key=True, autoincrement=False),
            Column("src_endpoint", String, nullable=False),
            Column("dst_endpoint", String, nullable=False),
        )

        # Communication Endpoint
        self._communication_endpoint_table = Table(
            "communication_endpoint",
            self.registry.metadata,
            Column("id", String, primary_key=True),
            Column("ip_address", String, nullable=False),
            Column("port", BigInteger, nullable=False),
            Column("transport", String, nullable=False),
        )

        # Network Interface
        self._network_interface_table = Table(
            "network_interface",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("name", String, nullable=False),
            Column("kind", String, nullable=False),
            Column("mac_address", String, nullable=True),
        )

        self._network_addresses_table = Table(
            "network_addresses",
            self.registry.metadata,
            Column("id", String, primary_key=True, nullable=False),
            Column("interface_id", String, ForeignKey("network_interface.id"), nullable=False),
            Column("address", String, nullable=False),
            Column("family", String, nullable=False),
            Column("prefix_length", BigInteger, nullable=False),
        )

        # Metadata Table
        self._metadata_table = Table(
            "metadata",
            self.registry.metadata,
            Column("id", String, primary_key=True),
            Column("reference_id", String, nullable=False),
            Column("metadata_type", String, nullable=False),
            Column("metadata_value", String, nullable=False),
        )

        # Metric

        self._metric_table = Table(
            "metric",
            self.registry.metadata,
            Column("name", String, primary_key=True),
            Column("unit", String),
            Column("description", String),
        )

        # CPU
        self._cpu_table = Table(
            "cpu",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("manufacturer", String),
            Column("model", String),
            Column("physical_cores", BigInteger),
            Column("logical_cores", BigInteger),
            Column("package_id", BigInteger),
            Column("cache", BigInteger),
            Column("hyperthreading", Boolean),
            Column("core_ids", CommaSeparatedIntegers),
            Column("idle_power_package", Float),
            Column("max_power_package", Float),
        )

        # Core
        self._core_table = Table(
            "core",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("core_id", BigInteger, nullable=False),
            Column("package_id", BigInteger, nullable=False),
            Column("thread_siblings", CommaSeparatedIntegers, nullable=False),
            Column("mhz", BigInteger, nullable=False),
        )

        # DRAM

        self._dram_table = Table(
            "dram",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("package_id", BigInteger, nullable=False),
            Column("idle_power", Float),
            Column("max_power", Float),
            Column("total_memory", Float),
        )

        # GPU

        self._gpu_table = Table(
            "gpu",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("manufacturer", String),
            Column("model", String),
            Column("serial_number", String),
            Column("uuid", String),
            Column("pci_id", String),
            Column("idle_power", Float),
            Column("max_power", Float),
            Column("compute_capability_major", Integer),
            Column("compute_capability_minor", Integer),
        )

        # Host
        self._host_table = Table(
            "host",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("name", String),
            Column("host_type", Enum(HostType, name="host_type_enum")),
            Column("OS", String),
        )

        # Logical Entity

        self._logical_entity_table = Table(
            "logical_entity",
            self.registry.metadata,
            Column("id", String, ForeignKey("entity.id"), primary_key=True, nullable=False),
            Column("native_thread_id", BigInteger),
            Column("native_pid", BigInteger),
            Column("pid_id", String),
            Column("native_parent_tid", BigInteger),
            Column("parent_tid_id", String),
            Column("native_parent_pid", BigInteger),
            Column("parent_pid_id", String),
            Column("executable_name", String),
            Column("cmdline", String),
            Column("start_time", DateTime),
            Column("pid_start_time", DateTime),
            Column("parent_tid_start_time", DateTime),
            Column("parent_pid_start_time", DateTime),
            Column("resource_group_id", BigInteger),
            Column("resource_group_name", String),
            Column("user_id", BigInteger),
            Column("group_id", BigInteger),
        )

        self.registry.metadata.create_all(self.engine)

    def _entity_row(self, entity: Entity) -> dict:
        return {
            "id": entity.id,
            "parent_id": entity.parent_id,
            "entity_type": entity.__class__.__name__,
        }

    def _cpu_row(self, cpu: CPU) -> dict:
        return {
            "id": cpu.id,
            "manufacturer": cpu.manufacturer,
            "model": cpu.model,
            "physical_cores": cpu.physical_cores,
            "logical_cores": cpu.logical_cores,
            "package_id": cpu.package_id,
            "cache": cpu.cache,
            "hyperthreading": cpu.hyperthreading,
            "core_ids": cpu.get_core_numbers(),
            "idle_power_package": cpu.idle_power_package,
            "max_power_package": cpu.max_power_package,
        }

    def _core_row(self, core: Core) -> dict:
        return {
            "id": core.id,
            "core_id": core.core_id,
            "package_id": core.package_id,
            "thread_siblings": core.thread_siblings,
            "mhz": core.mhz,
        }

    def _dram_row(self, dram: DRAM) -> dict:
        return {
            "id": dram.id,
            "total_memory": dram.total_memory,
            "idle_power": dram.idle_power,
            "package_id": dram.cpu.package_id,
            "max_power": dram.max_power,
        }

    def _gpu_row(self, gpu: GPU) -> dict:
        return {
            "id": gpu.id,
            "manufacturer": gpu.manufacturer,
            "model": gpu.model,
            "serial_number": gpu.serial_number,
            "uuid": gpu.uuid,
            "pci_id": gpu.pci_id,
            "idle_power": gpu.idle_power,
            "max_power": gpu.max_power,
            "compute_capability_major": gpu.compute_capability_major,
            "compute_capability_minor": gpu.compute_capability_minor,
        }

    def _host_row(self, host: HostEntity) -> dict:
        return {
            "id": host.id,
            "name": host.name,
            "host_type": host.host_type,
            "OS": host.os,
        }

    def _network_interface_row(self, interface: NetworkInterface) -> tuple[dict, list[dict]]:
        network_interface_row = {
            "id": interface.id,
            "name": interface.name,
            "kind": interface.kind.value,
            "mac_address": interface.mac_address,
        }
        addresses_rows = [
            {
                "id": addr.id,
                "interface_id": interface.id,
                "family": addr.family.value,
                "address": addr.address,
                "prefix_length": addr.prefix_length,
            }
            for addr in interface.addresses
        ]
        return network_interface_row, addresses_rows

    def _communication_endpoint_row(self, endpoint: CommunicationEndpoint) -> dict:
        transport = endpoint._network_address.transport
        transport_str = transport.value if isinstance(transport, NetworkTransport) else transport

        endpoint_row = {
            "id": endpoint.id,
            "ip_address": endpoint._network_address.ip_address,
            "port": endpoint._network_address.port,
            "transport": transport_str,
        }

        return endpoint_row

    def _logical_entity_row(self, logical_entity: LogicalEntity) -> tuple[dict, dict, dict, dict]:
        logical_entity_row = {
            "id": logical_entity.id,
            "native_thread_id": logical_entity.native_thread_id,
            "native_pid": logical_entity.native_pid,
            "pid_id": logical_entity.pid_id,
            "native_parent_tid": logical_entity.native_parent_thread_id,
            "parent_tid_id": logical_entity.parent_tid_id,
            "native_parent_pid": logical_entity.native_parent_pid,
            "parent_pid_id": logical_entity.parent_pid_id,
            "executable_name": logical_entity.executable_name,
            "cmdline": logical_entity.cmdline,
            "start_time": logical_entity.start_time,
            "pid_start_time": logical_entity.pid_start_time,
            "parent_pid_start_time": logical_entity.parent_pid_start_time,
            "resource_group_id": logical_entity.resource_group_id,
            "resource_group_name": logical_entity.resource_group_name,
            "user_id": logical_entity.user_id,
            "group_id": logical_entity.group_id,
        }

        pid_row = {
            "id": logical_entity.pid_id,
            "native_thread_id": logical_entity.native_pid,
            "native_pid": logical_entity.native_pid,
            "pid_id": logical_entity.pid_id,
            "native_parent_tid": None,
            "parent_tid_id": None,
            "native_parent_pid": None,
            "parent_pid_id": None,
            "executable_name": None,
            "cmdline": None,
            "start_time": logical_entity.pid_start_time,
            "pid_start_time": logical_entity.pid_start_time,
            "parent_pid_start_time": None,
            "resource_group_id": None,
            "resource_group_name": None,
            "user_id": None,
            "group_id": None,
        }

        parent_pid_row = {
            "id": logical_entity.parent_pid_id,
            "native_thread_id": logical_entity.native_parent_pid,
            "native_pid": logical_entity.native_parent_pid,
            "pid_id": logical_entity.parent_pid_id,
            "native_parent_tid": None,
            "parent_tid_id": None,
            "native_parent_pid": None,
            "parent_pid_id": None,
            "executable_name": None,
            "cmdline": None,
            "start_time": logical_entity.parent_pid_start_time,
            "pid_start_time": logical_entity.parent_pid_start_time,
            "parent_pid_start_time": None,
            "resource_group_id": None,
            "resource_group_name": None,
            "user_id": None,
            "group_id": None,
        }

        parent_tid_row = {
            "id": logical_entity.parent_tid_id,
            "native_thread_id": logical_entity.native_parent_thread_id,
            "native_pid": None,
            "pid_id": None,
            "native_parent_tid": None,
            "parent_tid_id": None,
            "native_parent_pid": None,
            "parent_pid_id": None,
            "executable_name": None,
            "cmdline": None,
            "start_time": None,
            "pid_start_time": logical_entity.parent_tid_start_time,
            "parent_pid_start_time": None,
            "resource_group_id": None,
            "resource_group_name": None,
            "user_id": None,
            "group_id": None,
        }

        return (logical_entity_row, pid_row, parent_pid_row, parent_tid_row)

    def _metadata_rows(self, metadata: list[Metadata], default_reference_id: str | None = None) -> list[dict]:
        rows: list[dict] = []
        for md in metadata:
            reference_id = md.reference_id or default_reference_id
            if reference_id is None:
                continue
            rows.append(
                {
                    "id": md.id,
                    "reference_id": reference_id,
                    "metadata_type": md.metadata_type,
                    "metadata_value": str(md.metadata_value),
                }
            )
        return rows

    def _entity_upsert_stmt(self):
        stmt = insert(self._entity_table)
        return stmt.on_conflict_do_nothing(index_elements=[self._entity_table.c.id])

    def _cpu_upsert_stmt(self):
        stmt = insert(self._cpu_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._cpu_table.c.id],
            set_={
                "manufacturer": prefer_text(stmt.excluded.manufacturer, self._cpu_table.c.manufacturer),
                "model": prefer_text(stmt.excluded.model, self._cpu_table.c.model),
                "package_id": func.coalesce(stmt.excluded.package_id, self._cpu_table.c.package_id),
                "cache": func.coalesce(stmt.excluded.cache, self._cpu_table.c.cache),
                "physical_cores": func.coalesce(stmt.excluded.physical_cores, self._cpu_table.c.physical_cores),
                "logical_cores": func.coalesce(stmt.excluded.logical_cores, self._cpu_table.c.logical_cores),
                "hyperthreading": func.coalesce(stmt.excluded.hyperthreading, self._cpu_table.c.hyperthreading),
                "idle_power_package": func.coalesce(
                    stmt.excluded.idle_power_package, self._cpu_table.c.idle_power_package
                ),
                "max_power_package": func.coalesce(
                    stmt.excluded.max_power_package, self._cpu_table.c.max_power_package
                ),
                "core_ids": func.coalesce(stmt.excluded.core_ids, self._cpu_table.c.core_ids),
            },
        )

    def _core_upsert_stmt(self):
        stmt = insert(self._core_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._core_table.c.id],
            set_={
                "core_id": func.coalesce(stmt.excluded.core_id, self._core_table.c.core_id),
                "package_id": func.coalesce(stmt.excluded.package_id, self._core_table.c.package_id),
                "thread_siblings": func.coalesce(
                    func.nullif(stmt.excluded.thread_siblings, ""), self._core_table.c.thread_siblings
                ),
                "mhz": func.coalesce(stmt.excluded.mhz, self._core_table.c.mhz),
            },
        )

    def _network_interface_upsert_stmt(self):
        stmt = insert(self._network_interface_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._network_interface_table.c.id],
            set_={
                "name": prefer_text(stmt.excluded.name, self._network_interface_table.c.name),
                "kind": prefer_text(stmt.excluded.kind, self._network_interface_table.c.kind),
                "mac_address": prefer_text(stmt.excluded.mac_address, self._network_interface_table.c.mac_address),
            },
        )

    def _network_address_upsert_stmt(self):
        stmt = insert(self._network_addresses_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._network_addresses_table.c.id],
            set_={
                "interface_id": func.coalesce(stmt.excluded.interface_id, self._network_addresses_table.c.interface_id),
                "address": prefer_text(stmt.excluded.address, self._network_addresses_table.c.address),
                "family": prefer_text(stmt.excluded.family, self._network_addresses_table.c.family),
                "prefix_length": func.coalesce(
                    stmt.excluded.prefix_length, self._network_addresses_table.c.prefix_length
                ),
            },
        )

    def _dram_upsert_stmt(self):
        stmt = insert(self._dram_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._dram_table.c.id],
            set_={
                "total_memory": func.coalesce(stmt.excluded.total_memory, self._dram_table.c.total_memory),
                "idle_power": func.coalesce(stmt.excluded.idle_power, self._dram_table.c.idle_power),
                "max_power": func.coalesce(stmt.excluded.max_power, self._dram_table.c.max_power),
                "package_id": func.coalesce(stmt.excluded.package_id, self._dram_table.c.package_id),
            },
        )

    def _gpu_upsert_stmt(self):
        stmt = insert(self._gpu_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._gpu_table.c.id],
            set_={
                "manufacturer": func.coalesce(
                    func.nullif(stmt.excluded.manufacturer, ""), self._gpu_table.c.manufacturer
                ),
                "model": func.coalesce(func.nullif(stmt.excluded.model, ""), self._gpu_table.c.model),
                "serial_number": func.coalesce(
                    func.nullif(stmt.excluded.serial_number, ""), self._gpu_table.c.serial_number
                ),
                "uuid": func.coalesce(func.nullif(stmt.excluded.uuid, ""), self._gpu_table.c.uuid),
                "pci_id": func.coalesce(func.nullif(stmt.excluded.pci_id, ""), self._gpu_table.c.pci_id),
                "idle_power": func.coalesce(stmt.excluded.idle_power, self._gpu_table.c.idle_power),
                "max_power": func.coalesce(stmt.excluded.max_power, self._gpu_table.c.max_power),
                "compute_capability_major": func.coalesce(
                    stmt.excluded.compute_capability_major, self._gpu_table.c.compute_capability_major
                ),
                "compute_capability_minor": func.coalesce(
                    stmt.excluded.compute_capability_minor, self._gpu_table.c.compute_capability_minor
                ),
            },
        )

    def _host_upsert_stmt(self):
        stmt = insert(self._host_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._host_table.c.id],
            set_={
                "name": func.coalesce(func.nullif(stmt.excluded.name, ""), self._host_table.c.name),
                "host_type": func.coalesce(func.nullif(stmt.excluded.host_type, ""), self._host_table.c.host_type),
                "OS": func.coalesce(func.nullif(stmt.excluded.OS, ""), self._host_table.c.OS),
            },
        )

    def _communication_endpoint_upsert_stmt(self):
        stmt = insert(self._communication_endpoint_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._communication_endpoint_table.c.id],
            set_={
                "ip_address": func.coalesce(
                    func.nullif(stmt.excluded.ip_address, ""), self._communication_endpoint_table.c.ip_address
                ),
                "port": func.coalesce(stmt.excluded.port, self._communication_endpoint_table.c.port),
                "transport": func.coalesce(
                    func.nullif(stmt.excluded.transport, ""), self._communication_endpoint_table.c.transport
                ),
            },
        )

    def _metadata_upsert_stmt(self):
        stmt = insert(self._metadata_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._metadata_table.c.id],
            set_={
                "reference_id": func.coalesce(stmt.excluded.reference_id, self._metadata_table.c.reference_id),
                "metadata_type": prefer_text(stmt.excluded.metadata_type, self._metadata_table.c.metadata_type),
                "metadata_value": prefer_text(stmt.excluded.metadata_value, self._metadata_table.c.metadata_value),
            },
        )

    def _logical_entity_upsert_stmt(self):
        stmt = insert(self._logical_entity_table)
        return stmt.on_conflict_do_update(
            index_elements=[self._logical_entity_table.c.id],
            set_={
                "native_thread_id": func.coalesce(
                    stmt.excluded.native_thread_id, self._logical_entity_table.c.native_thread_id
                ),
                "native_pid": func.coalesce(stmt.excluded.native_pid, self._logical_entity_table.c.native_pid),
                "pid_id": func.coalesce(func.nullif(stmt.excluded.pid_id, ""), self._logical_entity_table.c.pid_id),
                "native_parent_pid": func.coalesce(
                    stmt.excluded.native_parent_pid, self._logical_entity_table.c.native_parent_pid
                ),
                "parent_pid_id": func.coalesce(
                    func.nullif(stmt.excluded.parent_pid_id, ""), self._logical_entity_table.c.parent_pid_id
                ),
                "parent_tid_id": func.coalesce(
                    func.nullif(stmt.excluded.parent_tid_id, ""), self._logical_entity_table.c.parent_tid_id
                ),
                "executable_name": func.coalesce(
                    func.nullif(stmt.excluded.executable_name, ""), self._logical_entity_table.c.executable_name
                ),
                "cmdline": func.coalesce(func.nullif(stmt.excluded.cmdline, ""), self._logical_entity_table.c.cmdline),
                "start_time": func.coalesce(stmt.excluded.start_time, self._logical_entity_table.c.start_time),
                "pid_start_time": func.coalesce(
                    stmt.excluded.pid_start_time, self._logical_entity_table.c.pid_start_time
                ),
                "parent_pid_start_time": func.coalesce(
                    stmt.excluded.parent_pid_start_time, self._logical_entity_table.c.parent_pid_start_time
                ),
                "resource_group_id": func.coalesce(
                    stmt.excluded.resource_group_id, self._logical_entity_table.c.resource_group_id
                ),
                "resource_group_name": func.coalesce(
                    func.nullif(stmt.excluded.resource_group_name, ""), self._logical_entity_table.c.resource_group_name
                ),
                "user_id": func.coalesce(stmt.excluded.user_id, self._logical_entity_table.c.user_id),
                "group_id": func.coalesce(stmt.excluded.group_id, self._logical_entity_table.c.group_id),
            },
        )

    def _execute_stmt(self, stmt, rows):
        if len(rows) == 0:
            return
        with self.session_scope() as session:
            session.execute(stmt, rows)

    def _execute_entities_with_parents(self, rows):
        rows = list(rows)
        if not rows:
            return

        stmt = self._entity_upsert_stmt()
        self._execute_stmt(stmt, rows)

    def _save_entity(self, entity: Entity):
        upsert = self._entity_upsert_stmt()
        self._execute_stmt(upsert, self._entity_row(entity))

    def _save_core(self, core: Core):
        self._save_entity(core)

        upsert = self._core_upsert_stmt()
        self._execute_stmt(upsert, self._core_row(core))

    def _save_measurement(self, measurement: Measurement):
        stmt = self._measurement_table.insert().values(
            id=measurement.id,
            value=measurement.value,
            metric_id=measurement.metric_id,
            physical_entity_id=measurement.physical_entity_id,
            logical_entity_id=measurement.logical_entity_id,
            start_ts=measurement.start_ts,
            end_ts=measurement.end_ts,
        )
        self._execute_stmt(stmt)

    def _save_communication_measurement(self, comm_measurement: CommunicationMeasurement):
        self._save_measurement(comm_measurement)
        stmt = self._communication_measurement_table.insert().values(
            measurement_id=comm_measurement.id,
            src_endpoint=comm_measurement.src_endpoint,
            dst_endpoint=comm_measurement.dst_endpoint,
        )
        self._execute_stmt(stmt)

    def _save_metric(self, metric: Metric):
        stmt = self._metric_table.insert().values(name=metric.name, unit=metric.unit, description=metric.description)
        with self.session_scope() as session:
            session.execute(stmt)

    def _merge_row(self, old_row: dict, new_row: dict):
        merged_row = old_row.copy()
        for column, value in new_row.items():
            if value in (None, "", []):
                continue
            if old_row.get(column) in (None, "", []):
                merged_row[column] = value
        return merged_row

    def save_entities(self, entities: Sequence[Entity], progressbar: bool = False):
        entity_rows = {}
        cpu_rows = {}
        core_rows = {}
        dram_rows = {}
        gpu_rows = {}
        host_rows = {}
        communication_endpoint_rows = {}
        logical_entity_rows = {}
        network_interface_rows = {}
        network_addresses = []

        entity_iterator = tqdm(entities, desc="Saving entities") if progressbar else entities

        for e in entity_iterator:
            entity_rows[e.id] = self._entity_row(e)

            if isinstance(e, CPU):
                cpu_row = self._cpu_row(e)
                if e.id in cpu_rows:
                    cpu_rows[e.id] = self._merge_row(cpu_rows[e.id], cpu_row)
                else:
                    cpu_rows[e.id] = cpu_row
                for core in e.cores:
                    core_entity_row = self._entity_row(core)
                    if core.id in entity_rows:
                        entity_rows[core.id] = self._merge_row(entity_rows[core.id], core_entity_row)
                    else:
                        entity_rows[core.id] = core_entity_row
                    core_row = self._core_row(core)
                    if core.id in core_rows:
                        core_rows[core.id] = self._merge_row(core_rows[core.id], core_row)
                    else:
                        core_rows[core.id] = core_row

            if isinstance(e, Core):
                core_row = self._core_row(e)
                if e.id in core_rows:
                    core_rows[e.id] = self._merge_row(core_rows[e.id], core_row)
                else:
                    core_rows[e.id] = core_row
            elif isinstance(e, DRAM):
                dram_row = self._dram_row(e)
                if e.id in dram_rows:
                    dram_rows[e.id] = self._merge_row(dram_rows[e.id], dram_row)
                else:
                    dram_rows[e.id] = dram_row
            elif isinstance(e, GPU):
                gpu_row = self._gpu_row(e)
                if e.id in gpu_rows:
                    gpu_rows[e.id] = self._merge_row(gpu_rows[e.id], gpu_row)
                else:
                    gpu_rows[e.id] = gpu_row
            elif isinstance(e, NetworkInterface):
                ni_row, addr_rows = self._network_interface_row(e)
                network_interface_rows[e.id] = ni_row
                for addr_row in addr_rows:
                    network_addresses.append(addr_row)
            elif isinstance(e, HostEntity):
                host_row = self._host_row(e)
                if e.id in host_rows:
                    host_rows[e.id] = self._merge_row(host_rows[e.id], host_row)
                else:
                    host_rows[e.id] = host_row
            elif isinstance(e, LogicalEntity):
                le_row = self._logical_entity_row(e)

                for p in le_row:
                    if p["id"] is None or p["id"] == "":
                        continue
                    if p["id"] not in entity_rows:
                        entity_rows[p["id"]] = {
                            "id": p["id"],
                            "parent_id": None,
                            "entity_type": "LogicalEntity",
                        }
                    if p["id"] in logical_entity_rows:
                        logical_entity_rows[p["id"]] = self._merge_row(logical_entity_rows[p["id"]], p)
                    else:
                        logical_entity_rows[p["id"]] = p
            elif isinstance(e, CommunicationEndpoint):
                endpoint_row = self._communication_endpoint_row(e)
                communication_endpoint_rows[e.id] = endpoint_row

        self._execute_entities_with_parents(entity_rows.values())
        self._execute_stmt(self._cpu_upsert_stmt(), list(cpu_rows.values()))
        self._execute_stmt(self._core_upsert_stmt(), list(core_rows.values()))
        self._execute_stmt(self._dram_upsert_stmt(), list(dram_rows.values()))
        self._execute_stmt(self._gpu_upsert_stmt(), list(gpu_rows.values()))
        self._execute_stmt(self._host_upsert_stmt(), list(host_rows.values()))
        self._execute_stmt(self._network_interface_upsert_stmt(), list(network_interface_rows.values()))
        self._execute_stmt(self._network_address_upsert_stmt(), network_addresses)
        self._execute_stmt(self._logical_entity_upsert_stmt(), list(logical_entity_rows.values()))
        self._execute_stmt(self._communication_endpoint_upsert_stmt(), list(communication_endpoint_rows.values()))

    def save_metadata(self, metadata: Metadata):
        md_rows = self._metadata_rows(metadata)
        self._execute_stmt(self._metadata_upsert_stmt(), md_rows)

    def save_all_metrics(self):
        all_metrics = Metric.get_all_metrics()
        stmt = select(self._metric_table.c.name)
        with self.session_scope() as session:
            result = set(session.execute(stmt).scalars().all())

        missing_metrics = set(all_metrics) - result

        for metric in missing_metrics:
            missing_metric = Metric.get(metric)
            if missing_metric is not None:
                self._save_metric(missing_metric)

    def _save_measurement_objects(
        self, measurements: Sequence[Measurement], comm_measurements: Sequence[CommunicationMeasurement]
    ):
        # Build rows for measurement table (includes comm measurements too)
        normal_measurements_rows = [
            dict(
                id=m.id,
                value=m.value,
                metric_id=m.metric_id,
                physical_entity_id=m.physical_entity_id,
                logical_entity_id=m.logical_entity_id,
                start_ts=m.start_ts,
                end_ts=m.end_ts,
            )
            for m in (measurements + comm_measurements)
        ]

        # Build rows for communication_measurement table
        comm_measurements_rows = [
            dict(
                measurement_id=m.id,
                src_endpoint=m.src_endpoint,
                dst_endpoint=m.dst_endpoint,
            )
            for m in comm_measurements
        ]

        # Bulk insert using context manager (single transaction)
        with self.session_scope() as session:
            if normal_measurements_rows:
                session.execute(self._measurement_table.insert(), normal_measurements_rows)

            if comm_measurements_rows:
                session.execute(self._communication_measurement_table.insert(), comm_measurements_rows)

    def _save_measurement_df(self, measurement_df: pl.DataFrame):
        with self.engine.begin() as sa_conn:
            # DuckDBConnection (DB-API connection). With duckdb-engine, this is usually the DuckDBConnection.
            ddb_conn = sa_conn.connection
            # 1) Insert measurement rows (measurement_dfs)
            self._count += measurement_df.filter(pl.col("metric_id") == "CPU_CORE_PID_WORK").height
            measurement_df = measurement_df.with_columns(
                pl.col("start_ts").dt.convert_time_zone("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%.6f").alias("start_ts"),
                pl.col("end_ts").dt.convert_time_zone("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%.6f").alias("end_ts"),
            )
            ddb_conn.register("tmp_df", measurement_df)
            try:
                ddb_conn.execute("""
                    INSERT INTO measurement
                        (id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts)
                    SELECT
                        id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts
                    FROM tmp_df
                """)
            finally:
                ddb_conn.unregister("tmp_df")

        logger.debug(f"Saved work measurements: {self._count}")

    def _save_comm_measurement_df(self, comm_df: pl.DataFrame):
        with self.engine.begin() as sa_conn:
            ddb_conn = sa_conn.connection
            comm_df = comm_df.with_columns(
                pl.col("start_ts").dt.convert_time_zone("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%.6f").alias("start_ts"),
                pl.col("end_ts").dt.convert_time_zone("UTC").dt.strftime("%Y-%m-%d %H:%M:%S%.6f").alias("end_ts"),
            )
            comm_meas_arrow = comm_df.select(
                [
                    "id",
                    "value",
                    "metric_id",
                    "physical_entity_id",
                    "logical_entity_id",
                    "start_ts",
                    "end_ts",
                ]
            ).to_arrow(compat_level=pl.CompatLevel.newest())
            comm_info_arrow = comm_df.select(["id", "src_endpoint", "dst_endpoint"]).to_arrow(
                compat_level=pl.CompatLevel.newest()
            )

            # Register once; run both inserts; always unregister.
            ddb_conn.register("tmp_comm_info_df", comm_info_arrow)
            try:
                # Insert into measurement using a registered view with only needed cols
                # Easiest is to register the selected df under a second name.
                ddb_conn.register("tmp_comm_meas_df", comm_meas_arrow)
                try:
                    ddb_conn.execute("""
                        INSERT INTO measurement
                            (id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts)
                        SELECT
                            id, value, metric_id, physical_entity_id, logical_entity_id, start_ts, end_ts
                        FROM tmp_comm_meas_df
                    """)
                finally:
                    ddb_conn.unregister("tmp_comm_meas_df")

                # Insert into communication_measurement
                ddb_conn.execute("""
                    INSERT INTO communication_measurement
                        (measurement_id, src_endpoint, dst_endpoint)
                    SELECT
                        id, src_endpoint, dst_endpoint
                    FROM tmp_comm_info_df
                """)
            finally:
                ddb_conn.unregister("tmp_comm_info_df")

            logger.debug(f"Saved communication measurements: {comm_df.height}")

    def save_measurements(self, measurements: Sequence[Measurement | pl.DataFrame | str], progressbar: bool = False):
        # split measurements
        measurement_dfs = []
        measurement_comm_dfs = []
        measurement_objs = []
        measurement_comm_objs = []
        measurement_paths = []
        for measurement in measurements:
            # allways new
            if isinstance(measurement, pl.DataFrame):
                measurement_df = _add_id_to_measurement_df(measurement)
                if "src_endpoint" in measurement.columns and "dst_endpoint" in measurement.columns:
                    measurement_comm_dfs.append(measurement_df)
                else:
                    measurement_dfs.append(measurement_df)
            elif isinstance(measurement, CommunicationMeasurement):
                measurement_comm_objs.append(measurement)
            elif isinstance(measurement, Measurement):
                measurement_objs.append(measurement)
            elif isinstance(measurement, str):
                measurement_paths.append(measurement)
            else:
                logger.warning(f"Unknown measurement type: {type(measurement)}")

        self._save_measurement_objects(measurement_objs, measurement_comm_objs)

        if len(measurement_dfs) > 0:
            measurement_df = pl.concat(measurement_dfs, how="vertical")
            self._save_measurement_df(measurement_df)
        if len(measurement_comm_dfs) > 0:
            comm_df = pl.concat(measurement_comm_dfs, how="vertical")
            self._save_comm_measurement_df(comm_df)

    def save_data_record(self, data_record: DataRecord, progressbar: bool = False):
        """
        Save a DataRecord to the storage.
        :param data_record: The DataRecord to save.
        """
        if len(data_record.entities) > 0:
            self.save_entities(data_record.entities, progressbar=progressbar)
        if len(data_record.measurements) > 0:
            self.save_measurements(data_record.measurements, progressbar=progressbar)
        if len(data_record.metadata) > 0:
            self.save_metadata(data_record.metadata)

    def close(self):
        """
        Close the database connection.
        """
        self.engine.dispose()
