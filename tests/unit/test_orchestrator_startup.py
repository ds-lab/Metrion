from unittest.mock import MagicMock

import pytest
from loguru import logger

from metrion.entity.dram import DRAM

orchestrator = pytest.importorskip("metrion.orchestrator", reason="needs bcc")

REMOTE_READS = "CPU_CORE_PID_MEM_LOAD_L3_MISS_REMOTE_DRAM"


def _startup_warnings(drams, cpu_metrics):
    orch = orchestrator.Orchestrator.__new__(orchestrator.Orchestrator)
    orch._config = MagicMock()
    orch._config.energy_model.get_component_metrics.return_value = cpu_metrics
    orch._entities = {DRAM: [MagicMock() for _ in range(drams)]}

    messages = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        orch._warn_if_remote_reads_cannot_be_placed()
    finally:
        logger.remove(sink_id)
    return messages


@pytest.mark.parametrize(
    ("drams", "cpu_metrics", "warns"),
    [
        (3, [REMOTE_READS], True),
        (3, "*", True),
        (2, [REMOTE_READS], False),
        (3, ["CPU_CORE_PID_WORK"], False),
    ],
)
def test_remote_reads_on_more_than_two_sockets_are_flagged_at_startup(drams, cpu_metrics, warns):
    """The counter does not say which foreign socket a read went to; only with two there is no choice."""
    assert bool(_startup_warnings(drams, cpu_metrics)) is warns
