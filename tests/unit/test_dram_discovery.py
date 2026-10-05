import pytest

from metrion.config_reader.config_schema import PowerConfig
from metrion.discovery.hardware_detection import HardwareDetection
from metrion.discovery.system_scanner.bare_metal.linux import dram as dram_scanner
from metrion.discovery.system_scanner.bare_metal.linux.dram import get_memory_per_package
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.host import Host, HostType


def _sysfs(tmp_path, nodes, cpu_packages):
    """Build a fake /sys from {node_id: (cpulist, memory_mb)} and {cpu_id: package_id}."""
    root = tmp_path / "sys"

    for node_id, (cpulist, memory_mb) in nodes.items():
        node_path = root / "devices/system/node" / f"node{node_id}"
        node_path.mkdir(parents=True)
        (node_path / "cpulist").write_text(f"{cpulist}\n")
        (node_path / "meminfo").write_text(
            f"Node {node_id} MemTotal:       {int(memory_mb * 1024)} kB\nNode {node_id} MemFree:        1024 kB\n"
        )

    for cpu_id, package_id in cpu_packages.items():
        topology = root / "devices/system/cpu" / f"cpu{cpu_id}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package_id}\n")

    return root


def test_two_memory_controllers_land_on_one_socket(tmp_path):
    """Regression: DIMMs named Controller0-/Controller1-ChannelA-DIMM0 were read as socket 2."""
    root = _sysfs(tmp_path, {0: ("0-15", 32000.0)}, dict.fromkeys(range(16), 0))

    assert get_memory_per_package(root, tmp_path / "proc_meminfo") == {0: 32000.0}


def test_sub_numa_clustering_sums_the_nodes_of_a_socket(tmp_path):
    """Intel SNC and AMD NPS split one socket into several nodes; the memory belongs to the socket."""
    root = _sysfs(
        tmp_path,
        {0: ("0-7", 16000.0), 1: ("8-15", 16000.0)},
        dict.fromkeys(range(16), 0),
    )

    assert get_memory_per_package(root, tmp_path / "proc_meminfo") == {0: 32000.0}


def test_two_sockets_stay_apart(tmp_path):
    root = _sysfs(
        tmp_path,
        {0: ("0-7", 16000.0), 1: ("8-15", 48000.0)},
        {**dict.fromkeys(range(8), 0), **dict.fromkeys(range(8, 16), 1)},
    )

    assert get_memory_per_package(root, tmp_path / "proc_meminfo") == {0: 16000.0, 1: 48000.0}


def test_memory_only_node_is_skipped(tmp_path):
    """A node without CPUs is CXL or persistent memory: no socket owns it and RAPL does not meter it."""
    root = _sysfs(tmp_path, {0: ("0-7", 16000.0), 1: ("", 128000.0)}, dict.fromkeys(range(8), 0))

    assert get_memory_per_package(root, tmp_path / "proc_meminfo") == {0: 16000.0}


def test_no_numa_nodes_falls_back_to_the_machine_total(tmp_path):
    """Regression: a kernel without CONFIG_NUMA must not leave the host without any DRAM entity."""
    proc_meminfo = tmp_path / "proc_meminfo"
    proc_meminfo.write_text("MemTotal:       16384000 kB\nMemFree:         1024 kB\n")

    assert get_memory_per_package(tmp_path / "sys", proc_meminfo) == {0: 16000.0}


def test_every_package_id_comes_from_the_cpu_topology(tmp_path):
    """The invariant the DMI locator heuristic broke: sockets are read, never derived."""
    root = _sysfs(
        tmp_path,
        {0: ("0-3", 8000.0), 1: ("4-7", 8000.0)},
        {**dict.fromkeys(range(4), 0), **dict.fromkeys(range(4, 8), 3)},
    )

    assert set(get_memory_per_package(root, tmp_path / "proc_meminfo")) == {0, 3}


def test_a_node_the_kernel_gives_no_package_for_is_skipped(tmp_path):
    root = _sysfs(tmp_path, {0: ("0-3", 8000.0), 1: ("4-7", 8000.0)}, dict.fromkeys(range(4), 0))

    assert get_memory_per_package(root, tmp_path / "proc_meminfo") == {0: 8000.0}


class _StubConfig:
    def get_subcomponent_power_config(self, component_type, subcomponent_id):
        return PowerConfig("auto", "auto")


def _cpu(package_id):
    return CPU(
        manufacturer="ACME",
        model="Test",
        physical_cores=8,
        logical_cores=16,
        package_id=package_id,
        cache=1024,
        hyperthreading=True,
        topology={},
        cores=[],
        idle_power_package=10.0,
        max_power_package=100.0,
        parent_id="host-1",
    )


def _detector(dram_devices):
    detector = object.__new__(HardwareDetection)
    detector._devices = {"dram": dram_devices}
    detector._config = _StubConfig()
    return detector


def _socket(package_id, total_memory=32000.0):
    return {
        "package_id": package_id,
        "total_memory": total_memory,
        "power_mode": "auto",
        "idle_power": 3.0,
        "max_power": 12.0,
    }


@pytest.fixture
def host():
    return Host(name="testhost", host_type=HostType.SERVER, os_name="linux")


def test_one_dram_entity_per_socket(host):
    detector = _detector([_socket(0), _socket(1, 16000.0)])

    entities = detector._create_dram_entities(host, [_cpu(0), _cpu(1)])

    assert [e.cpu.package_id for e in entities] == [0, 1]
    assert [e.total_memory for e in entities] == [32000.0, 16000.0]
    assert all(isinstance(e, DRAM) for e in entities)


def test_detected_power_reaches_the_entity(host):
    entities = _detector([_socket(0)])._create_dram_entities(host, [_cpu(0)])

    assert entities[0].idle_power == 3.0
    assert entities[0].max_power == 12.0


def test_socket_without_a_cpu_is_skipped(host):
    """Cannot happen now that both sides read the kernel, but the guard must stay honest."""
    entities = _detector([_socket(2)])._create_dram_entities(host, [_cpu(0)])

    assert entities == []


def test_unavailable_scanner_output_does_not_iterate_a_string(host):
    """The tree builder returns 'Not available' when the module cannot run."""
    entities = _detector("Not available")._create_dram_entities(host, [_cpu(0)])

    assert entities == []


def test_scanner_output_feeds_entity_creation_unchanged(host, monkeypatch):
    """The seam that broke: what get_info emits must be what _create_dram_entities can consume."""
    monkeypatch.setattr(dram_scanner, "get_memory_per_package", lambda: {0: 32000.0, 1: 16000.0})
    monkeypatch.setattr(
        dram_scanner, "_resolve_power_data", lambda power, calibration_file: ("auto", {0: {"idle": 3.0, "max": 12.0}})
    )

    sockets = dram_scanner.get_info({"calibration": None})
    entities = _detector(sockets)._create_dram_entities(host, [_cpu(0), _cpu(1)])

    assert [e.cpu.package_id for e in entities] == [0, 1]
    assert [e.total_memory for e in entities] == [32000.0, 16000.0]
    assert entities[0].idle_power == 3.0
    assert entities[1].idle_power == 0.0, "a socket RAPL reports no power for still gets an entity"
