import os
import sys
import time

import pytest

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="BPF is Linux only"),
    pytest.mark.skipif(sys.platform.startswith("linux") and os.geteuid() != 0, reason="loading BPF needs root"),
]

mmr = pytest.importorskip("metrion.metrion_rust", reason="needs the built metrion_rust extension")

# The objects compiled by build.rs.
OBJECTS = ["logical_entity_registry", "sched_switch", "tcp_monitor"]
MAX_LOAD_SECONDS = 2.0


@pytest.mark.parametrize("name", OBJECTS)
def test_bpf_object_loads(name):
    start = time.monotonic()
    program = mmr.BpfProgram(name)
    elapsed = time.monotonic() - start
    del program

    assert elapsed < MAX_LOAD_SECONDS, f"{name} took {elapsed:.2f} s to load"
