"""RAPLCPU finds the package zone of a socket by its name, not by its zone number (issue #6)."""

from loguru import logger

from metrion.discovery.system_scanner.bare_metal.linux.max_power_gpu_dram_cpu import rapl_dram_caps
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.metric import CPU_SOCKET_PKG_ENERGY, DRAM_SOCKET_TOTAL_ENERGY
from metrion.monitoring.data_source.cpu.rapl import RAPLCPU

PACKAGE_WRAP_UJ = 262143328850
PSYS_WRAP_UJ = 4294967295000000

# pvs108: the platform zone psys takes intel-rapl:1 and pushes package-1 to intel-rapl:2.
WITH_PSYS = [
    ("intel-rapl:0", "package-0", 1_000_000, PACKAGE_WRAP_UJ),
    ("intel-rapl:0/intel-rapl:0:0", "dram", 200_000, 65712999613),
    ("intel-rapl:1", "psys", 999_000_000, PSYS_WRAP_UJ),
    ("intel-rapl:2", "package-1", 3_000_000, PACKAGE_WRAP_UJ),
    ("intel-rapl:2/intel-rapl:2:0", "dram", 400_000, 65712999613),
]

WITHOUT_PSYS = [
    ("intel-rapl:0", "package-0", 1_000_000, PACKAGE_WRAP_UJ),
    ("intel-rapl:0/intel-rapl:0:0", "dram", 200_000, 65712999613),
    ("intel-rapl:1", "package-1", 3_000_000, PACKAGE_WRAP_UJ),
    ("intel-rapl:1/intel-rapl:1:0", "dram", 400_000, 65712999613),
]


def _powercap(tmp_path, zones):
    """Build a fake intel-rapl powercap tree from (directory, name, energy_uj, max_energy_range_uj)."""
    root = tmp_path / "intel-rapl"
    for directory, name, energy, wrap in zones:
        zone = root / directory
        zone.mkdir(parents=True)
        (zone / "name").write_text(f"{name}\n")
        (zone / "energy_uj").write_text(f"{energy}\n")
        (zone / "max_energy_range_uj").write_text(f"{wrap}\n")
    return root


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


def _rapl(root, package_ids=(0, 1)):
    """One RAPLCPU for the CPUs and one for the DRAMs, as every EntityMonitor holds a single entity type."""
    cpus = [_cpu(package_id) for package_id in package_ids]
    drams = [DRAM(total_memory=16.0, cpu=cpu, idle_power=1.0, max_power=8.0, parent_id=cpu.id) for cpu in cpus]
    return RAPLCPU(cpus, powercap_tree=root), RAPLCPU(drams, powercap_tree=root), cpus, drams


def _zones(rapl):
    """socket -> {metric name: zone directory name}."""
    return {
        package_id: {metric.name: path.name for metric, path in info["metrics"].items()}
        for package_id, info in rapl._socket_info.items()
    }


def test_socket_1_is_package_1_not_the_psys_zone_before_it(tmp_path):
    cpu_rapl, dram_rapl, _, _ = _rapl(_powercap(tmp_path, WITH_PSYS))

    assert (
        _zones(cpu_rapl)
        == _zones(dram_rapl)
        == {
            0: {CPU_SOCKET_PKG_ENERGY.name: "intel-rapl:0", DRAM_SOCKET_TOTAL_ENERGY.name: "intel-rapl:0:0"},
            1: {CPU_SOCKET_PKG_ENERGY.name: "intel-rapl:2", DRAM_SOCKET_TOTAL_ENERGY.name: "intel-rapl:2:0"},
        }
    )


def test_psys_is_never_read_as_a_package(tmp_path):
    cpu_rapl, dram_rapl, cpus, drams = _rapl(_powercap(tmp_path, WITH_PSYS))

    metrics = [CPU_SOCKET_PKG_ENERGY, DRAM_SOCKET_TOTAL_ENERGY]
    measurements = [*cpu_rapl.get_data(metrics).measurements, *dram_rapl.get_data(metrics).measurements]

    by_entity = {(m.metric_id, m.physical_entity_id): m.value for m in measurements}
    assert by_entity == {
        (CPU_SOCKET_PKG_ENERGY.name, cpus[0].id): 1.0,
        (CPU_SOCKET_PKG_ENERGY.name, cpus[1].id): 3.0,
        (DRAM_SOCKET_TOTAL_ENERGY.name, drams[0].id): 0.2,
        (DRAM_SOCKET_TOTAL_ENERGY.name, drams[1].id): 0.4,
    }
    paths = [
        path
        for rapl in (cpu_rapl, dram_rapl)
        for info in rapl._socket_info.values()
        for path in info["metrics"].values()
    ]
    assert all(path.name != "intel-rapl:1" for path in paths)


def test_the_psys_zone_itself_yields_no_metrics(tmp_path):
    """The call that returned psys as CPU_SOCKET_PKG_ENERGY of socket 1 on pvs108."""
    root = _powercap(tmp_path, WITH_PSYS)

    assert RAPLCPU._discover_metrics_per_socket(RAPLCPU.__new__(RAPLCPU), root / "intel-rapl:1") == {}


def test_the_plain_layout_without_psys_still_works(tmp_path):
    cpu_rapl, dram_rapl, _, _ = _rapl(_powercap(tmp_path, WITHOUT_PSYS))

    assert (
        _zones(cpu_rapl)
        == _zones(dram_rapl)
        == {
            0: {CPU_SOCKET_PKG_ENERGY.name: "intel-rapl:0", DRAM_SOCKET_TOTAL_ENERGY.name: "intel-rapl:0:0"},
            1: {CPU_SOCKET_PKG_ENERGY.name: "intel-rapl:1", DRAM_SOCKET_TOTAL_ENERGY.name: "intel-rapl:1:0"},
        }
    )


def test_a_socket_without_a_package_zone_is_logged_and_not_given_another_zone(tmp_path):
    """Only psys sits at intel-rapl:1, so socket 1 must not fall back to it or to package-0."""
    root = _powercap(tmp_path, [zone for zone in WITH_PSYS if not zone[0].startswith("intel-rapl:2")])
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        cpu_rapl, _, _, _ = _rapl(root)
    finally:
        logger.remove(sink_id)

    assert _zones(cpu_rapl) == {
        0: {CPU_SOCKET_PKG_ENERGY.name: "intel-rapl:0", DRAM_SOCKET_TOTAL_ENERGY.name: "intel-rapl:0:0"},
        1: {},
    }
    assert any("package-1" in message for message in messages)


def test_per_die_zones_are_not_mapped_to_a_socket(tmp_path):
    """package-N-die-M zones each wrap on their own; they are reported, never read as the package."""
    root = _powercap(
        tmp_path,
        [
            ("intel-rapl:0", "package-0-die-0", 1_000_000, PACKAGE_WRAP_UJ),
            ("intel-rapl:1", "package-0-die-1", 2_000_000, PACKAGE_WRAP_UJ),
        ],
    )
    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        cpu_rapl, _, _, _ = _rapl(root, package_ids=(0,))
    finally:
        logger.remove(sink_id)

    assert _zones(cpu_rapl) == {0: {}}
    assert any("per die" in message for message in messages)


def test_dram_caps_are_labelled_with_the_package_number_not_the_zone_number(tmp_path):
    root = _powercap(tmp_path, WITH_PSYS)

    assert [cap.package for cap in rapl_dram_caps(root / "intel-rapl:2")] == ["1"]
