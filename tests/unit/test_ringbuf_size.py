import mmap
from unittest.mock import MagicMock

import pytest

ss = pytest.importorskip(
    "metrion.monitoring.data_source.cpu.sched_switch", reason="needs bcc and the built metrion_rust extension"
)


def test_ringbuf_size_reaches_the_probe_in_pages(monkeypatch):
    """BPF_RINGBUF_OUTPUT counts pages, not bytes: 524288 read as bytes once pinned 2 GiB."""
    bpf = MagicMock()
    monkeypatch.setattr(ss, "BPF", bpf)
    monkeypatch.setattr(ss, "count_cpu_sockets_from_sysfs", lambda: 1)
    monitor = ss.SchedSwitchBpfMonitor.__new__(ss.SchedSwitchBpfMonitor)
    monitor._ringbuf_mib = 64

    monitor._create_bpf()

    cflags = bpf.call_args.kwargs["cflags"]
    pages = int(next(flag for flag in cflags if flag.startswith("-DRINGBUF_PAGES=")).split("=")[1])
    assert pages * mmap.PAGESIZE == 64 * 1024 * 1024
