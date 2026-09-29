from types import SimpleNamespace

import pytest

ss = pytest.importorskip(
    "metrion.monitoring.data_source.cpu.sched_switch", reason="needs the built metrion_rust extension"
)

from metrion.monitoring.data_source.cpu.collection_plan import SchedSwitchCollectionPlan  # noqa: E402

PERF_ARRAYS = {"cycles_core", "aperf_core", "mperf_core", "load_local_perf", "load_remote_perf", "store_perf"}


def attached_counters(monkeypatch, tmp_path, online, cpu_lists):
    """Run _setup_perf_events on a fake sysfs.

    Returns (map, key, CPU the counter was opened on) for every map entry it set.
    """
    for pmu, cpus in cpu_lists.items():
        path = tmp_path / "sys/devices" / pmu / "cpus"
        path.parent.mkdir(parents=True)
        path.write_text(cpus)
    monkeypatch.setattr(ss, "Path", lambda path: tmp_path / path.lstrip("/"))
    monkeypatch.setattr(ss, "open", lambda path: (tmp_path / path.lstrip("/")).open(), raising=False)
    monkeypatch.setattr(ss, "get_online_cpus", lambda: online)
    monkeypatch.setattr(ss, "read_perf_type", lambda path: 4)
    monkeypatch.setattr(ss, "read_perf_event", lambda path: 0x3C)
    monkeypatch.setattr(ss, "resolve_with_strace", lambda event, cpu: {"type": 4, "config": 0x1D3})

    cpu_of_fd = {}

    def open_counters(attr, pid=-1, cpus=None, group_fds=None):
        fds = []
        for i, cpu in enumerate(cpus):
            if group_fds:
                assert cpu_of_fd[group_fds[i]] == cpu, "group leader opened on another CPU"
            fd = 100 + len(cpu_of_fd)
            cpu_of_fd[fd] = cpu
            fds.append(fd)
        return fds

    monkeypatch.setattr(ss.CustomPerf, "perf_custom_event_open_with_results", staticmethod(open_counters))

    entries = []
    monitor = ss.SchedSwitchBpfMonitor.__new__(ss.SchedSwitchBpfMonitor)
    monitor._collection_plan = SchedSwitchCollectionPlan.all_enabled()
    monitor._bpf = SimpleNamespace(set_perf_fd=lambda name, key, fd: entries.append((name, key, cpu_of_fd[fd])))
    monitor._setup_perf_events()
    return entries


def test_hybrid_counters_are_keyed_by_cpu_id_when_p_and_e_cores_interleave(monkeypatch, tmp_path):
    """The probe reads a counter by the CPU it runs on; listing P-cores before E-cores must not shift that."""
    entries = attached_counters(monkeypatch, tmp_path, list(range(8)), {"cpu_core": "0-1,4-5", "cpu_atom": "2-3,6-7"})

    assert {name for name, _, _ in entries} == {"cycles_core", "aperf_core", "mperf_core"}
    assert all(key == cpu for _, key, cpu in entries)
    assert {key for name, key, _ in entries if name == "cycles_core"} == set(range(8))


def test_counters_are_keyed_by_cpu_id_when_a_cpu_is_offline(monkeypatch, tmp_path):
    entries = attached_counters(monkeypatch, tmp_path, [0, 1, 3], {})

    assert {name for name, _, _ in entries} == PERF_ARRAYS
    assert all(key == cpu for _, key, cpu in entries)
    assert {key for name, key, _ in entries if name == "store_perf"} == {0, 1, 3}
