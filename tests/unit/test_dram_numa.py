import datetime as dt

import polars as pl
import pytest

from metrion.energy_attribution.energy_attribution import DRAMEnergyAttributionModel

# Zwei Sockel, je ein Riegel, drei Threads:
#   p1 auf core-A liest lokal    -> dram-0   (100 Zugriffe)
#   p2 auf core-B liest lokal    -> dram-1   (300 Zugriffe)
#   p3 auf core-A liest entfernt -> dram-1   (200 Zugriffe)
# dram-0 gehoert damit allein p1, dram-1 teilen sich p2 und p3 im Verhaeltnis 60:40.

LOCAL = DRAMEnergyAttributionModel._READ_LOCAL_METRIC
REMOTE = DRAMEnergyAttributionModel._READ_REMOTE_METRIC


def _ts(second):
    return dt.datetime(2026, 1, 1) + dt.timedelta(seconds=second)


def _frames():
    dram_df = pl.DataFrame(
        {
            "id": ["dram-0", "dram-1"],
            "cpu_id": ["cpu-A", "cpu-B"],
            # Beide Riegel haengen am selben Host — genau daran scheiterte die alte Bindung.
            "parent_id": ["host-1", "host-1"],
            "idle_power": [0.0, 0.0],
        }
    )
    core_df = pl.DataFrame({"id": ["core-A", "core-B"], "parent_id": ["cpu-A", "cpu-B"]})
    return dram_df, core_df


def _rows(metric, entity, logical, values, seconds):
    return pl.DataFrame(
        {
            "metric_id": [metric] * len(values),
            "physical_entity_id": [entity] * len(values),
            "logical_entity_id": pl.Series(logical, dtype=pl.String),
            "value": values,
            "start_ts": [_ts(s) for s in seconds],
            "end_ts": [_ts(s) for s in seconds],
        }
    )


def _measurements():
    parts = []
    for module in ("dram-0", "dram-1"):
        # 100 J je Riegel ueber [0, 1); idle_power ist 0, alles ist dynamisch
        parts.append(_rows("DRAM_SOCKET_TOTAL_ENERGY", module, [None] * 2, [0.0, 100.0], [0, 1]))
        for thread in ("p1", "p2", "p3"):
            parts.append(_rows("DRAM_SOCKET_PID_MEMORY", module, [thread], [1000.0], [0]))

    reads = pl.concat(
        [
            _rows(LOCAL, "core-A", ["p1"], [100.0], [0]),
            _rows(LOCAL, "core-B", ["p2"], [300.0], [0]),
            _rows(REMOTE, "core-A", ["p3"], [200.0], [0]),
        ]
    )
    return pl.concat([*parts, reads])


def _dynamic_on(module):
    dram_df, core_df = _frames()
    _, _, dynamic = DRAMEnergyAttributionModel().attribute_energy(
        measurement_df=_measurements(), core_df=core_df, dram_df=dram_df
    )
    return dynamic.filter(pl.col("physical_entity_id") == module)


def test_local_reads_reach_their_own_module():
    """Der Regressionstest: mit der alten Bindung fiel jeder lokale Zugriff heraus."""
    on_dram_0 = _dynamic_on("dram-0")

    p1 = on_dram_0.filter(pl.col("logical_entity_id") == "p1")["value"].sum()
    assert p1 == pytest.approx(100.0), "p1 liest als einziger auf dram-0 und bekommt dessen Energie"


def test_a_remote_read_does_not_count_on_its_own_module():
    """p3 sitzt auf cpu-A; dram-0 ist sein eigener Riegel, dort ist der Zugriff nicht remote."""
    on_dram_0 = _dynamic_on("dram-0")

    p3 = on_dram_0.filter(pl.col("logical_entity_id") == "p3")["value"].sum()
    assert p3 == pytest.approx(0.0), "der entfernte Zugriff darf nicht auf dem eigenen Riegel landen"


def test_shares_differ_between_modules():
    """Mit der alten Bindung waren die Anteile auf beiden Riegeln identisch."""
    on_dram_1 = _dynamic_on("dram-1")

    shares = {row["logical_entity_id"]: row["value"] for row in on_dram_1.iter_rows(named=True)}
    assert shares.get("p2") == pytest.approx(60.0), "300 von 500 Zugriffen"
    assert shares.get("p3") == pytest.approx(40.0), "200 von 500 Zugriffen"
    assert "p1" not in shares or shares["p1"] == pytest.approx(0.0)
