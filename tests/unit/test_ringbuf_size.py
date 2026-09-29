from unittest.mock import MagicMock

import psutil
import pytest

ss = pytest.importorskip(
    "metrion.monitoring.data_source.cpu.sched_switch", reason="needs the built metrion_rust extension"
)


def test_ringbuf_size_reaches_the_probe_in_bytes(monkeypatch):
    """A libbpf ring buffer's max_entries counts bytes: 64 read as pages or MiB would size it wrong."""
    program = MagicMock()
    monkeypatch.setattr(ss.mar, "BpfProgram", program)
    monitor = ss.SchedSwitchBpfMonitor.__new__(ss.SchedSwitchBpfMonitor)
    monitor._ringbuf_mib = 64

    monitor._create_bpf()

    name, max_entries = program.call_args.args
    assert name == "sched_switch"
    assert max_entries["events"] == 64 * 1024 * 1024
    assert max_entries["cycles_core"] == psutil.cpu_count(logical=True)
