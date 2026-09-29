import ctypes as ct
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("metrion.metrion_rust", reason="the event modules need the built metrion_rust extension")

from metrion.entity.logical_entity_registry import TaskEvent  # noqa: E402
from metrion.monitoring.data_source.cpu.sched_switch import SchedSwitchBpfMonitor  # noqa: E402
from metrion.monitoring.data_source.network.tcp_monitor import TcpEvent  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

C_PRELUDE = """
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#define TASK_COMM_LEN 16
#define CMDLINE_MAX_LEN 128
"""


def c_layout(source: str, struct: str, tmp_path: Path) -> list[tuple[str, int, int]]:
    """(name, offset, size) of every field of ``struct`` in a BPF source, as clang lays it out."""
    text = (REPO / source).read_text()
    body = re.search(rf"struct {struct}\s*\{{(.*?)\n\}};", text, re.S).group(1)
    fields = re.findall(r"^\s*(?:u8|u16|u32|u64|char)\s+(\w+)", body, re.M)
    prints = "".join(
        f'printf("{f} %zu %zu\\n", offsetof(struct {struct}, {f}), sizeof(((struct {struct} *)0)->{f}));'
        for f in fields
    )
    main = f'int main(void) {{ printf("sizeof %zu 0\\n", sizeof(struct {struct})); {prints} return 0; }}\n'
    program = f"{C_PRELUDE}struct {struct} {{{body}\n}};\n{main}"
    (tmp_path / "layout.c").write_text(program)
    subprocess.run(["clang", "-o", tmp_path / "layout", tmp_path / "layout.c"], check=True)
    out = subprocess.run([tmp_path / "layout"], check=True, capture_output=True, text=True).stdout
    return [(name, int(offset), int(size)) for name, offset, size in (line.split() for line in out.splitlines())]


def ctypes_layout(cls: type[ct.Structure]) -> list[tuple[str, int, int]]:
    fields = [(name, getattr(cls, name).offset, getattr(cls, name).size) for name, *_ in cls._fields_]
    return [("sizeof", ct.sizeof(cls), 0), *fields]


def numpy_layout(dtype) -> list[tuple[str, int, int]]:
    fields = [(name, dtype.fields[name][1], dtype.fields[name][0].itemsize) for name in dtype.names]
    return [("sizeof", dtype.itemsize, 0), *fields]


@pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")
@pytest.mark.parametrize(
    ("source", "struct", "python_layout"),
    [
        ("metrion/entity/logical_entity_registry.bpf.c", "task_event_t", lambda: ctypes_layout(TaskEvent)),
        ("metrion/monitoring/data_source/network/tcp_monitor.bpf.c", "tcp_event_v2", lambda: ctypes_layout(TcpEvent)),
        (
            "metrion/monitoring/data_source/cpu/sched_switch.bpf.c",
            "sched_event_v2",
            lambda: numpy_layout(SchedSwitchBpfMonitor.SCHED_EVENT_DTYPE),
        ),
    ],
)
def test_python_event_matches_the_bpf_struct(source, struct, python_layout, tmp_path):
    """The Python side reads the ring buffer by these offsets; one field out of step garbles every field after it."""
    # Padding fields are named __pad in C and _pad in the numpy dtype.
    normalize = lambda layout: [(name.lstrip("_"), offset, size) for name, offset, size in layout]  # noqa: E731
    assert normalize(python_layout()) == normalize(c_layout(source, struct, tmp_path))
