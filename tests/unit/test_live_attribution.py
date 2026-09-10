from unittest.mock import MagicMock

import polars as pl

from metrion.data_collector import DataSaver
from metrion.energy_attribution.energy_attribution import EnergyAttributionModelBase


def _saver():
    saver = DataSaver.__new__(DataSaver)
    saver.dyn_eam = MagicMock()
    return saver


def _frame(metric):
    return pl.DataFrame(
        {
            "metric_id": [metric],
            "physical_entity_id": ["p"],
            "logical_entity_id": ["le"],
            "value": [1.0],
            "start_ts": [None],
            "end_ts": [None],
        }
    )


def test_the_three_component_results_become_one_flat_list():
    """The earlier version unpacked two triples and died on the third."""
    saver = _saver()
    triples = [
        tuple(_frame(f"{component}_{kind}") for kind in ("TOTAL", "IDLE", "DYNAMIC"))
        for component in ("CPU_PKG_PID", "DRAM_SOCKET_PID", "GPU_DEVICE_PID")
    ]
    saver.dyn_eam.attribute_energy.return_value = tuple(triples)

    result = saver._collect_attribute_energies([], pl.DataFrame())

    assert len({df["metric_id"][0] for df in result}) == 9


def test_empty_results_are_dropped():
    saver = _saver()
    empty = EnergyAttributionModelBase._empty_result()
    saver.dyn_eam.attribute_energy.return_value = (empty, empty, empty)

    assert saver._collect_attribute_energies([], pl.DataFrame()) == []


def test_force_reaches_the_model():
    """The shutdown drain is force=True; without it the last window stays in the buffer."""
    saver = _saver()
    empty = EnergyAttributionModelBase._empty_result()
    saver.dyn_eam.attribute_energy.return_value = (empty, empty, empty)

    saver._collect_attribute_energies([], pl.DataFrame(), force=True)

    assert saver.dyn_eam.attribute_energy.call_args.kwargs["force"] is True
