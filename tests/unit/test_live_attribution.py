import datetime as dt
from queue import Empty
from unittest.mock import MagicMock, patch

import polars as pl

from metrion.config_reader import LiveAttributionConfig
from metrion.data_collector import DataSaver, ForgottenEntities
from metrion.energy_attribution.energy_attribution import EnergyAttributionModelBase
from metrion.entity.logical_entity import LogicalEntity

T0 = dt.datetime(2026, 1, 1)
NOTHING = (pl.DataFrame(), pl.DataFrame(), pl.DataFrame())


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

    result = saver._collect_attribute_energies(pl.DataFrame())

    assert len({df["metric_id"][0] for df in result}) == 9


def test_empty_results_are_dropped():
    saver = _saver()
    empty = EnergyAttributionModelBase._empty_result()
    saver.dyn_eam.attribute_energy.return_value = (empty, empty, empty)

    assert saver._collect_attribute_energies(pl.DataFrame()) == []


def test_force_reaches_the_model():
    """The shutdown drain is force=True; without it the last window stays in the buffer."""
    saver = _saver()
    empty = EnergyAttributionModelBase._empty_result()
    saver.dyn_eam.attribute_energy.return_value = (empty, empty, empty)

    saver._collect_attribute_energies(pl.DataFrame(), force=True)

    assert saver.dyn_eam.attribute_energy.call_args.kwargs["force"] is True


def _live_saver():
    """A saver with live attribution on; the model, the writer and the storage are mocks."""
    saver = DataSaver(None, None, None, "duckdb", "unused", LiveAttributionConfig(enabled=True, window=10.0), [])
    saver.dyn_eam = MagicMock()
    saver._energy_writer = MagicMock()
    saver._data_storage = MagicMock()
    return saver


def _at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def _long(metric, value, start_s, end_s, thread="le_1"):
    return pl.DataFrame(
        {
            "metric_id": [metric],
            "physical_entity_id": ["p"],
            "logical_entity_id": [thread],
            "value": [value],
            "start_ts": [_at(start_s)],
            "end_ts": [_at(end_s)],
        }
    )


def _cycle(saver, frames, attributed_until, force=False):
    """One saver round in which the model returns frames and has attributed up to attributed_until."""
    result = pl.concat(frames) if frames else pl.DataFrame()
    saver.dyn_eam.attribute_energy.return_value = ((result, pl.DataFrame(), pl.DataFrame()), NOTHING, NOTHING)
    saver.dyn_eam.attributed_until.return_value = attributed_until
    saver._run_live_attribution([pl.DataFrame()], force=force)


def _written(saver):
    frames = [call.args[0] for call in saver._energy_writer.write.call_args_list]
    return pl.concat(frames) if frames else pl.DataFrame()


def _thread():
    t = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    return LogicalEntity(
        native_thread_id=4711,
        native_pid=4711,
        native_parent_pid=1,
        native_parent_thread_id=1,
        executable_name="python3",
        cmdline="python3 train.py",
        start_time=t,
        pid_start_time=t,
        parent_tid_start_time=t,
        parent_pid_start_time=t,
    )


def test_a_window_is_written_once_every_component_has_passed_it():
    """Written earlier, it would come out twice, each time with one component missing."""
    saver = _live_saver()

    _cycle(saver, [_long("CPU_PKG_PID_TOTAL_ENERGY", 3.0, 1, 2)], attributed_until=_at(5))
    assert _written(saver).is_empty(), "the GPU has not passed the window yet"

    _cycle(saver, [_long("GPU_DEVICE_PID_TOTAL_ENERGY", 4.0, 2, 3)], attributed_until=_at(12))

    written = _written(saver)
    assert written.height == 1
    assert (written["cpu_total_j"][0], written["gpu_total_j"][0]) == (3.0, 4.0)


def test_the_shutdown_writes_the_windows_still_open():
    """Without it the energy of the last seconds before the stop would be lost."""
    saver = _live_saver()
    _cycle(saver, [_long("CPU_PKG_PID_TOTAL_ENERGY", 3.0, 1, 2)], attributed_until=None)

    _cycle(saver, [], attributed_until=None, force=True)

    assert _written(saver)["cpu_total_j"].to_list() == [3.0]


def test_traffic_frames_do_not_stop_the_attribution():
    """They share the measurements queue with the energy inputs, but carry two more columns."""
    saver = _live_saver()
    energy = _long("CPU_PKG_PID_TOTAL_ENERGY", 3.0, 1, 2)
    traffic = energy.with_columns(src_endpoint=pl.lit("a"), dst_endpoint=pl.lit("b"))
    saver.dyn_eam.attribute_energy.return_value = (NOTHING, NOTHING, NOTHING)
    saver.dyn_eam.attributed_until.return_value = None

    saver._run_live_attribution([energy, traffic])

    assert saver.dyn_eam.update_measurements.call_args.args[0].equals(energy)


def test_a_forgotten_thread_keeps_its_identity_until_its_windows_are_written():
    """The registry lets go 60 s after the exit; the result stream may still hold the thread's window open."""
    saver = _live_saver()
    thread = _thread()
    saver._take_entities([[thread], ForgottenEntities([thread.id])])

    _cycle(saver, [_long("CPU_PKG_PID_TOTAL_ENERGY", 3.0, 1, 2, thread.id)], attributed_until=_at(12))
    assert _written(saver)["executable_name"].to_list() == ["python3"]
    assert thread.id in saver._identities, "its windows could still be open"

    _cycle(saver, [], attributed_until=dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(minutes=1))
    assert thread.id not in saver._identities


def _saver_with_one_measurement(store_raw):
    saver = _saver()
    saver._store_raw = store_raw
    saver._data_storage = MagicMock()
    saver._measurements_queue = MagicMock()
    saver._measurements_queue.get_nowait.side_effect = ["/tmp/batch.arrow", Empty]
    return saver


def test_the_raw_stream_is_written_by_default():
    saver = _saver_with_one_measurement(True)

    with patch("metrion.data_collector.read_temp_ipc", return_value=pl.DataFrame({"value": [1.0]})):
        saver._process_measurements()

    assert saver._data_storage.save_measurements.call_count == 1


def test_the_raw_stream_is_dropped_when_switched_off_but_the_attribution_still_sees_it():
    """The point of the switch: the measurements pass through the saver and only the result stream is kept."""
    saver = _saver_with_one_measurement(False)

    with patch("metrion.data_collector.read_temp_ipc", return_value=pl.DataFrame({"value": [1.0]})):
        dfs = saver._process_measurements()

    saver._data_storage.save_measurements.assert_not_called()
    assert len(dfs) == 1
