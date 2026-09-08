from collections.abc import Sequence
from contextlib import contextmanager

import polars as pl
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
from metrion.util import _add_id_to_measurement_df, naive_utc_timestamps


def comma_separated(value) -> str | None:
    """Render ints the way a CommaSeparatedIntegers column holds them."""
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)

    return ",".join(str(item) for item in value)


class CommaSeparatedIntegers(TypeDecorator):
    impl = StringDB
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return comma_separated(value)

    def process_result_value(self, value, dialect):
        if value is None or value == "":
            return []
        # Convert comma-separated string back to a list of ints.
        return [int(item) for item in value.split(",")]


class SQLDatabaseStorage(DataStorageI):
    def __init__(self, db_url: str):
        self._count = 0
        self._db_url = db_url
        self._upsert_statements: dict[tuple[str, tuple[str, ...]], str] = {}
        self._varchar_columns: set[tuple[str, str]] = set()
        # Create a new SQLAlchemy engine.
        self.engine = create_engine(db_url)
        with self.engine.begin() as connection:
            # A timestamp that still carries a time zone is stored in the session time zone, so
            # without this the machine's locale decides what lands in the table.
            connection.exec_driver_sql("SET GLOBAL TimeZone='UTC'")
            # DuckDB keeps the insertion order of a table by default, which costs time and space
            # on an append-only write path that nothing reads in order.
            connection.exec_driver_sql("SET GLOBAL preserve_insertion_order=false")
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
            # No primary key: the ids are snowflakes, nothing joins on them, and the index
            # costs a fifth of the file.
            Column("id", BigInteger),
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
            "core_ids": comma_separated(cpu.get_core_numbers()),
            "idle_power_package": cpu.idle_power_package,
            "max_power_package": cpu.max_power_package,
        }

    def _core_row(self, core: Core) -> dict:
        return {
            "id": core.id,
            "core_id": core.core_id,
            "package_id": core.package_id,
            "thread_siblings": comma_separated(core.thread_siblings),
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
            "host_type": host.host_type.name,
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

    def _upsert_sql(self, table: str, columns: tuple[str, ...]) -> str:
        """
        Build the upsert for one batch shape: a later value only overrides a column it fills.

        Parameters
        ----------
        table : str
            Target table. It must have `id` as its primary key.
        columns : tuple of str
            The columns the batch carries, in the order the staging view has them.
        """
        if (table, columns) in self._upsert_statements:
            return self._upsert_statements[(table, columns)]

        if not self._varchar_columns:
            with self.engine.connect() as connection:
                self._varchar_columns = {
                    (row[0], row[1])
                    for row in connection.exec_driver_sql(
                        "SELECT table_name, column_name FROM duckdb_columns() WHERE data_type = 'VARCHAR'"
                    ).all()
                }

        names = ", ".join(f'"{column}"' for column in columns)
        updates = []
        for column in columns:
            if column == "id":
                continue
            value = f'excluded."{column}"'
            if (table, column) in self._varchar_columns:
                # An empty string carries as little as NULL, and neither may erase what is there.
                value = f"nullif({value}, '')"
            updates.append(f'"{column}" = coalesce({value}, {table}."{column}")')

        statement = (
            f"INSERT INTO {table} ({names}) SELECT {names} FROM staging "
            f"ON CONFLICT (id) DO UPDATE SET {', '.join(updates)}"
        )
        self._upsert_statements[(table, columns)] = statement

        return statement

    def _upsert(self, table: str, rows) -> None:
        """
        Write a batch of rows into a table in one statement.

        Parameters
        ----------
        table : str
            Target table.
        rows : Iterable[dict]
            Rows to write, all with the same keys. DuckDB keeps the first hit on a row and
            silently drops the rest, so of two rows with one id only the last is kept.
        """
        rows = list(rows)
        if not rows:
            return

        staging = pl.DataFrame(rows, infer_schema_length=None).unique(subset=["id"], keep="last")
        statement = self._upsert_sql(table, tuple(staging.columns))

        with self.engine.begin() as sa_conn:
            connection = sa_conn.connection
            connection.register("staging", staging)
            try:
                connection.execute(statement)
            finally:
                connection.unregister("staging")

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

        self._upsert("entity", entity_rows.values())
        self._upsert("cpu", cpu_rows.values())
        self._upsert("core", core_rows.values())
        self._upsert("dram", dram_rows.values())
        self._upsert("gpu", gpu_rows.values())
        self._upsert("host", host_rows.values())
        self._upsert("network_interface", network_interface_rows.values())
        self._upsert("network_addresses", network_addresses)
        self._upsert("logical_entity", logical_entity_rows.values())
        self._upsert("communication_endpoint", communication_endpoint_rows.values())

    def save_metadata(self, metadata: Metadata):
        self._upsert("metadata", self._metadata_rows(metadata))

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
            measurement_df = naive_utc_timestamps(measurement_df)
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
            comm_df = naive_utc_timestamps(comm_df)
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
