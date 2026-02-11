from datetime import datetime

import polars as pl
import pytest
from sqlalchemy import select

from metrion.data_record import DataRecord
from metrion.data_storage.db_data_storage import SQLDatabaseStorage
from metrion.entity.communication_endpoint import CommunicationEndpoint, NetworkAddress
from metrion.entity.core import Core
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.gpu import GPU
from metrion.entity.host import Host, HostType
from metrion.entity.logical_entity import LogicalEntity
from metrion.entity.metadata import Metadata
from metrion.measurement import Measurement
from metrion.metric import CPU_CORE_PID_TIME, CPU_SOCKET_PKG_ENERGY


@pytest.fixture(params=["sqlite", "duckdb"])
def storage(request, tmp_path):
    backend = request.param
    if backend == "duckdb":
        db_url = f"duckdb:///{tmp_path / 'test_duck.db'}"
    else:
        db_url = f"sqlite:///{tmp_path / 'test.db'}"

    storage = SQLDatabaseStorage(db_url)
    try:
        yield storage
    finally:
        storage.close()


def test_host_persisted_with_measurement(storage):
    host = Host("test-host", HostType.SERVER, "Linux")
    measurement = Measurement(
        value=1.23,
        metric_id=CPU_SOCKET_PKG_ENERGY.name,
        physical_entity_id=host.id,
    )
    record = DataRecord([host], [measurement])

    storage.save_data_record(record)

    with storage.Session() as session:
        host_row = session.execute(
            select(
                storage._host_table.c.id,
                storage._host_table.c.name,
                storage._host_table.c.host_type,
                storage._host_table.c.OS,
            )
        ).one()
        entity_row = session.execute(
            select(storage._entity_table.c.entity_type, storage._entity_table.c.parent_id).where(
                storage._entity_table.c.id == host.id
            )
        ).one()
        measurement_row = session.execute(
            select(
                storage._measurement_table.c.value,
                storage._measurement_table.c.metric_id,
                storage._measurement_table.c.physical_entity_id,
            )
        ).one()

    assert host_row.id == host.id
    assert host_row.name == host.name
    assert host_row.host_type == HostType.SERVER
    assert host_row.OS == host.os
    assert entity_row.entity_type == "Host"
    assert entity_row.parent_id is None
    assert measurement_row.metric_id == measurement.metric_id
    assert measurement_row.physical_entity_id == host.id
    assert measurement_row.value == pytest.approx(measurement.value)


def test_cpu_and_core_persisted(storage):
    host = Host("parent-host", HostType.SERVER, "Linux")
    cpu = CPU(
        manufacturer="Acme",
        model="X1",
        physical_cores=2,
        logical_cores=4,
        package_id=0,
        cache=8,
        hyperthreading=True,
        topology={0: 0},
        cores=[],
        idle_power_package=7.5,
        max_power_package=45.0,
        parent_id=host.id,
    )
    core = Core(core_id=0, package_id=0, thread_siblings=[0], mhz=3200, parent_id=cpu.id)
    cpu.cores = [core]

    record = DataRecord([host, cpu, core], [])
    storage.save_data_record(record)

    with storage.Session() as session:
        cpu_row = session.execute(
            select(
                storage._cpu_table.c.id,
                storage._cpu_table.c.manufacturer,
                storage._cpu_table.c.hyperthreading,
                storage._cpu_table.c.core_ids,
                storage._cpu_table.c.idle_power_package,
                storage._cpu_table.c.max_power_package,
            )
        ).one()
        core_row = session.execute(
            select(
                storage._core_table.c.id,
                storage._core_table.c.core_id,
                storage._core_table.c.thread_siblings,
            )
        ).one()

    assert cpu_row.id == cpu.id
    assert cpu_row.manufacturer == cpu.manufacturer
    assert cpu_row.hyperthreading is True
    assert cpu_row.core_ids == [core.core_id]
    assert cpu_row.idle_power_package == pytest.approx(cpu.idle_power_package)
    assert cpu_row.max_power_package == pytest.approx(cpu.max_power_package)
    assert core_row.id == core.id
    assert core_row.core_id == core.core_id
    assert core_row.thread_siblings == core.thread_siblings


def test_dram_persisted(storage):
    host = Host("host-with-dram", HostType.SERVER, "Linux")
    cpu = CPU(
        manufacturer="Acme",
        model="Y1",
        physical_cores=4,
        logical_cores=8,
        package_id=1,
        cache=16,
        hyperthreading=False,
        topology={},
        cores=[],
        idle_power_package=6.0,
        max_power_package=40.0,
        parent_id=host.id,
    )
    dram = DRAM(total_memory=64.0, cpu=cpu, idle_power=2.5, max_power=25.0, parent_id=host.id)

    record = DataRecord([host, cpu, dram], [])
    storage.save_data_record(record)

    with storage.Session() as session:
        dram_row = session.execute(
            select(
                storage._dram_table.c.id,
                storage._dram_table.c.total_memory,
                storage._dram_table.c.package_id,
                storage._dram_table.c.idle_power,
                storage._dram_table.c.max_power,
            )
        ).one()

    assert dram_row.id == dram.id
    assert dram_row.total_memory == pytest.approx(dram.total_memory)
    assert dram_row.package_id == cpu.package_id
    assert dram_row.idle_power == pytest.approx(dram.idle_power)
    assert dram_row.max_power == pytest.approx(dram.max_power)


def test_gpu_persisted(storage):
    host = Host("host-with-gpu", HostType.SERVER, "Linux")
    gpu = GPU(
        manufacturer="AcmeGPU",
        model="Z9",
        serial_number="SN123",
        uuid="uuid-123",
        pci_id="0000:00:01.0",
        idle_power=12.0,
        max_power=150.0,
        compute_capability_major=8,
        compute_capability_minor=6,
        parent_id=host.id,
    )

    record = DataRecord([host, gpu], [])
    storage.save_data_record(record)

    with storage.Session() as session:
        gpu_row = session.execute(
            select(
                storage._gpu_table.c.id,
                storage._gpu_table.c.model,
                storage._gpu_table.c.pci_id,
                storage._gpu_table.c.manufacturer,
                storage._gpu_table.c.idle_power,
                storage._gpu_table.c.max_power,
            )
        ).one()

    assert gpu_row.id == gpu.id
    assert gpu_row.model == gpu.model
    assert gpu_row.pci_id == gpu.pci_id
    assert gpu_row.manufacturer == gpu.manufacturer
    assert gpu_row.idle_power == pytest.approx(gpu.idle_power)
    assert gpu_row.max_power == pytest.approx(gpu.max_power)


def test_metadata_persisted_for_endpoint(storage):
    endpoint = CommunicationEndpoint(NetworkAddress("127.0.0.1", 8080))
    md = Metadata(reference_id=endpoint.id, metadata_type="net.dns", metadata_value="example.local")

    record = DataRecord([endpoint], [], metadata=[md])
    storage.save_data_record(record)

    with storage.Session() as session:
        metadata_row = session.execute(
            select(
                storage._metadata_table.c.reference_id,
                storage._metadata_table.c.metadata_type,
                storage._metadata_table.c.metadata_value,
            )
        ).one()

    assert metadata_row.reference_id == endpoint.id
    assert metadata_row.metadata_type == "net.dns"
    assert metadata_row.metadata_value == "example.local"


def test_logical_entity_upsert_enriches_metadata(storage):
    now = datetime.now()
    base_entity = LogicalEntity(
        native_thread_id=101,
        native_pid=101,
        native_parent_pid=1,
        native_parent_thread_id=0,
        executable_name="proc",
        cmdline="/usr/bin/proc",
        start_time=now,
        pid_start_time=now,
        parent_pid_start_time=now,
        parent_tid_start_time=now,
    )

    enriched_entity = LogicalEntity(
        native_thread_id=101,
        native_pid=101,
        native_parent_pid=1,
        native_parent_thread_id=0,
        executable_name="proc",
        cmdline="/usr/bin/proc",
        start_time=now,
        pid_start_time=now,
        parent_pid_start_time=now,
        parent_tid_start_time=now,
        resource_group_id=42,
        resource_group_name="/user.slice",
        user_id=1000,
        group_id=1000,
    )

    storage.save_data_record(DataRecord([base_entity], []))
    storage.save_data_record(DataRecord([enriched_entity], []))

    with storage.Session() as session:
        logical_entity_row = session.execute(
            select(
                storage._logical_entity_table.c.native_pid,
                storage._logical_entity_table.c.native_parent_pid,
                storage._logical_entity_table.c.resource_group_id,
                storage._logical_entity_table.c.user_id,
                storage._logical_entity_table.c.group_id,
                storage._logical_entity_table.c.cmdline,
                storage._logical_entity_table.c.executable_name,
            ).where(storage._logical_entity_table.c.id == base_entity.id)
        ).one()

    assert logical_entity_row.native_pid == enriched_entity.native_pid
    assert logical_entity_row.native_parent_pid == enriched_entity.native_parent_pid
    assert logical_entity_row.resource_group_id == enriched_entity.resource_group_id
    assert logical_entity_row.user_id == enriched_entity.user_id
    assert logical_entity_row.group_id == enriched_entity.group_id
    assert logical_entity_row.cmdline == enriched_entity.cmdline
    assert logical_entity_row.executable_name == enriched_entity.executable_name


def test_measurement_df_without_id_persisted(storage):
    backend = storage.engine.dialect.name
    driver_module = "adbc_driver_sqlite" if backend == "sqlite" else "adbc_driver_duckdb"
    pytest.importorskip(driver_module)

    host = Host("df-host", HostType.SERVER, "Linux")
    now = datetime.now()
    measurement_df = pl.DataFrame(
        {
            "value": [2.5],
            "metric_id": [CPU_CORE_PID_TIME.name],
            "physical_entity_id": [host.id],
            "logical_entity_id": [None],
            "start_ts": [now],
            "end_ts": [now],
        }
    )
    record = DataRecord([host], [measurement_df])

    storage.save_data_record(record)

    with storage.Session() as session:
        rows = session.execute(
            select(
                storage._measurement_table.c.metric_id,
                storage._measurement_table.c.physical_entity_id,
                storage._measurement_table.c.value,
            )
        ).all()
        print(rows)

    assert len(rows) == 1
    row = rows[0]
    assert row.metric_id == CPU_CORE_PID_TIME.name
    assert row.physical_entity_id == host.id
    assert row.value == pytest.approx(2.5)


def test_measurement_df_with_id_persisted(storage):
    backend = storage.engine.dialect.name
    driver_module = "adbc_driver_sqlite" if backend == "sqlite" else "adbc_driver_duckdb"
    pytest.importorskip(driver_module)

    host = Host("df-host", HostType.SERVER, "Linux")
    now = datetime.now()
    measurement = Measurement(
        value=1.45,
        metric_id=CPU_CORE_PID_TIME.name,
        physical_entity_id=host.id,
        start_ts=now,
        end_ts=now,
    )
    measurement_df = measurement.to_df()

    record = DataRecord([host], [measurement_df])

    storage.save_data_record(record)

    with storage.Session() as session:
        rows = session.execute(
            select(
                storage._measurement_table.c.id,
                storage._measurement_table.c.metric_id,
                storage._measurement_table.c.physical_entity_id,
                storage._measurement_table.c.value,
            )
        ).all()
        print(rows)

    assert len(rows) == 1
    row = rows[0]
    assert row.metric_id == CPU_CORE_PID_TIME.name
    assert row.physical_entity_id == host.id
    assert row.value == pytest.approx(1.45)
    assert row.id == measurement.id
