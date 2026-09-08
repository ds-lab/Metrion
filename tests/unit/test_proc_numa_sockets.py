import pytest

from metrion.discovery.system_scanner.bare_metal.linux.util import get_nodes_per_package
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM

# ProcNuma pulls in cpu/utils.py, which imports bcc at module level.
pytest.importorskip("bcc", reason="metrion.monitoring.data_source.cpu.utils imports bcc")

from metrion.monitoring.data_source.memory.proc_numa import ProcNuma, parse_numa_maps  # noqa: E402


def _sysfs(tmp_path, nodes, cpu_packages):
    """Build a fake /sys from {node_id: cpulist} and {cpu_id: package_id}."""
    root = tmp_path / "sys"

    for node_id, cpulist in nodes.items():
        node_path = root / "devices/system/node" / f"node{node_id}"
        node_path.mkdir(parents=True)
        (node_path / "cpulist").write_text(f"{cpulist}\n")

    for cpu_id, package_id in cpu_packages.items():
        topology = root / "devices/system/cpu" / f"cpu{cpu_id}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package_id}\n")

    return root


def test_sub_numa_clustering_gives_one_socket_several_nodes(tmp_path):
    root = _sysfs(tmp_path, {0: "0-3", 1: "4-7", 2: "8-11", 3: "12-15"}, dict.fromkeys(range(16), 0))

    assert get_nodes_per_package(root) == {0: [0, 1, 2, 3]}


def test_two_sockets_keep_their_own_nodes(tmp_path):
    root = _sysfs(
        tmp_path,
        {0: "0-3", 1: "4-7", 2: "8-11", 3: "12-15"},
        {**dict.fromkeys(range(8), 0), **dict.fromkeys(range(8, 16), 1)},
    )

    assert get_nodes_per_package(root) == {0: [0, 1], 1: [2, 3]}


def test_node_numbers_are_read_not_counted(tmp_path):
    """Node directories are not necessarily 0..N-1, so the id comes from the directory name."""
    root = _sysfs(tmp_path, {2: "0-3", 7: "4-7"}, dict.fromkeys(range(8), 1))

    assert get_nodes_per_package(root) == {1: [2, 7]}


def test_memory_only_node_belongs_to_no_socket(tmp_path):
    root = _sysfs(tmp_path, {0: "0-3", 1: ""}, dict.fromkeys(range(4), 0))

    assert get_nodes_per_package(root) == {0: [0]}


def test_no_numa_nodes_maps_nothing(tmp_path):
    assert get_nodes_per_package(tmp_path / "sys") == {}


def test_memory_on_every_node_is_counted():
    """Regression: nodes were bucketed as range(len(entities)), so higher ids were dropped."""
    lines = ["7f00 default anon=1 N0=256 N1=256 N2=256 N3=256 kernelpagesize_kB=4"]

    per_node, total = parse_numa_maps(lines, 4, 2048)

    assert sorted(per_node) == [0, 1, 2, 3]
    assert all(per_node[node] == pytest.approx(1.0) for node in per_node)
    assert total == pytest.approx(4.0)


def test_transparent_huge_pages_are_not_dropped():
    """Regression: 'huge' matches inside anon_huge=, which booked ordinary anonymous memory away."""
    line = "7f2c40000000 default anon=131072 dirty=131072 anon_huge=131072 N1=131072 kernelpagesize_kB=4"

    per_node, total = parse_numa_maps([line], 4, 2048)

    assert per_node[1] == pytest.approx(512.0)
    assert total == pytest.approx(512.0)


def test_heap_stack_and_file_maps_all_count():
    """Only 'Private' used to reach the metric, so heap, stack and huge mappings went missing."""
    lines = [
        "00400000 default file=/usr/bin/postgres mapped=256 N0=256 kernelpagesize_kB=4",
        "00e2b000 default heap anon=256 dirty=256 N0=256 kernelpagesize_kB=4",
        "7ffd3c7e1000 default stack anon=256 dirty=256 N0=256 kernelpagesize_kB=4",
        "7f2c80000000 default huge dirty=1 N0=1 kernelpagesize_kB=2048",
    ]

    per_node, total = parse_numa_maps(lines, 4, 2048)

    assert per_node[0] == pytest.approx(5.0)  # 3 x 1 MB of 4 kB pages plus one 2 MB huge page
    assert total == pytest.approx(5.0)


def test_huge_pages_without_an_explicit_page_size_use_the_huge_size():
    per_node, _ = parse_numa_maps(["7f00 default huge N0=1"], 4, 2048)

    assert per_node[0] == pytest.approx(2.0)


def test_an_explicit_page_size_wins_over_the_huge_guess():
    per_node, _ = parse_numa_maps(["7f00 default huge N0=1 kernelpagesize_kB=4"], 4, 2048)

    assert per_node[0] == pytest.approx(4 / 1024)


def test_no_lines_means_no_nodes():
    assert parse_numa_maps([], 4, 2048) == ({}, 0.0)


def _dram(package_id):
    cpu = CPU(
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
    return DRAM(total_memory=32000.0, cpu=cpu, idle_power=3.0, max_power=12.0, parent_id="host-1")


def _source(nodes_per_package):
    source = object.__new__(ProcNuma)
    source._nodes_per_package = nodes_per_package
    return source


def _measured(*node_values):
    return dict(node_values)


def test_socket_memory_sums_all_its_nodes():
    """Regression: only the node whose id matched the socket number was counted."""
    source = _source({0: [0, 1, 2, 3]})

    assert source._memory_on_socket(_measured((0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0)), _dram("0")) == 100.0


def test_sockets_do_not_take_each_others_nodes():
    source = _source({0: [0, 1], 1: [2, 3]})
    measured = _measured((0, 10.0), (1, 20.0), (2, 30.0), (3, 40.0))

    assert source._memory_on_socket(measured, _dram("0")) == 30.0
    assert source._memory_on_socket(measured, _dram("1")) == 70.0


def test_a_node_the_process_never_touched_counts_as_zero():
    source = _source({0: [0, 1]})

    assert source._memory_on_socket(_measured((0, 10.0)), _dram("0")) == 10.0


def test_unreadable_topology_falls_back_to_the_socket_number():
    """Without sysfs the old one-node-per-socket assumption is still better than reporting nothing."""
    source = _source({})

    assert source._memory_on_socket(_measured((0, 10.0), (1, 20.0)), _dram("1")) == 20.0
