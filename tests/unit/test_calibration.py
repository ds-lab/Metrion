import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrion import calibration
from metrion.calibration import CalibrationError
from metrion.config_reader.config_schema import AgentConfig
from metrion.discovery.tree_builder import HardwareTreeBuilder
from metrion.orchestrator import Orchestrator

ACTIVE = [
    "-.slice",
    "system.slice",
    "user.slice",
    "user-1003.slice",
    "kubepods.slice",
    "kubepods-besteffort.slice",
    "machine.slice",
    "init.scope",
    "containerd.service",
    "docker.service",
    "docker-cf89.scope",
    "nginx.service",
    "systemd-journald.service",
]


def test_the_default_freezes_top_level_slices_and_container_runtimes():
    units = calibration.select_units(ACTIVE, [], [])

    assert units == ["user.slice", "kubepods.slice", "containerd.service", "docker.service", "docker-cf89.scope"]


def test_freeze_adds_and_skip_removes():
    units = calibration.select_units(ACTIVE, ["machine.slice", "nginx.service"], ["docker*"])

    assert units == ["user.slice", "kubepods.slice", "containerd.service", "machine.slice", "nginx.service"]


def test_the_machine_itself_is_never_frozen():
    assert calibration.select_units(
        ACTIVE, ["system.slice", "systemd-*", "init.scope"], []
    ) == calibration.select_units(ACTIVE, [], [])


def test_a_pattern_without_match_is_a_typo():
    with pytest.raises(CalibrationError, match="kubpods"):
        calibration.select_units(ACTIVE, [], ["kubpods*"])


def test_describe_reads_every_block_of_systemctl_show(monkeypatch):
    output = "TasksCurrent=452\nId=user.slice\nCanFreeze=yes\n\nTasksCurrent=[not set]\nId=a.service\nCanFreeze=no\n"
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output))

    info = calibration.describe(["user.slice", "a.service"])

    assert info["user.slice"]["TasksCurrent"] == "452"
    assert info["a.service"]["CanFreeze"] == "no"


@pytest.fixture
def machine(monkeypatch):
    """A worker that freezes nothing, with two readings 10 s apart."""
    calls = []
    readings = iter(
        [
            {
                ("cpu", "0"): (262_000_000_000, 262_143_328_850, 1e6),  # wraps during the measurement
                ("dram", "0"): (None, None, 1e6),
                ("gpu", "GPU-1"): (1_000_000, None, 1e3),
            },
            {
                ("cpu", "0"): (6_671_150, 262_143_328_850, 1e6),
                ("dram", "0"): (None, None, 1e6),
                ("gpu", "GPU-1"): (1_150_000, None, 1e3),
            },
        ]
    )
    clock = iter([0, 10_000_000_000])

    def run(command, **kwargs):
        calls.append(command[1:])
        return SimpleNamespace(returncode=1 if command[-1] == "stuck.service" else 0, stderr="Unit cannot be frozen")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(calibration.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(calibration.time, "monotonic_ns", lambda: next(clock))
    monkeypatch.setattr(calibration, "read_counters", lambda: next(readings))
    return calls


def test_power_is_the_energy_over_the_measured_time_across_a_wrap(machine):
    result = calibration.measure(["user.slice"], 10)

    assert result["cpu"]["0"] == pytest.approx(15.0)
    assert result["dram"]["0"] is None, "no DRAM counter is not 0 W"
    assert result["gpu"]["GPU-1"] == pytest.approx(15.0)
    assert result["seconds"] == 10.0


def test_a_unit_that_does_not_freeze_is_reported_and_not_thawed(machine):
    result = calibration.measure(["user.slice", "stuck.service"], 10)

    assert result["frozen"] == ["user.slice"]
    assert result["not_frozen"] == {"stuck.service": "Unit cannot be frozen"}
    assert machine[-1] == ["thaw", "user.slice"]


def test_units_are_thawed_even_when_the_measurement_fails(machine, monkeypatch):
    def fail():
        raise OSError("sysfs gone")

    monkeypatch.setattr(calibration, "read_counters", fail)

    with pytest.raises(OSError):
        calibration.measure(["user.slice", "kubepods.slice"], 10)

    assert machine[-1] == ["thaw", "user.slice", "kubepods.slice"]


def test_counters_come_from_the_package_domains(tmp_path, monkeypatch):
    for directory, name, energy in [
        ("intel-rapl:0", "package-0", 100),
        ("intel-rapl:0/intel-rapl:0:0", "core", 50),
        ("intel-rapl:0/intel-rapl:0:1", "dram", 20),
        ("intel-rapl:1", "psys", 999),
    ]:
        domain = tmp_path / directory
        domain.mkdir(parents=True)
        (domain / "name").write_text(name + "\n")
        (domain / "energy_uj").write_text(f"{energy}\n")
        (domain / "max_energy_range_uj").write_text("1000\n")
    monkeypatch.setattr(calibration, "RAPL", tmp_path)
    monkeypatch.setitem(__import__("sys").modules, "pynvml", _no_gpu())

    counters = calibration.read_counters()

    assert counters == {("cpu", "0"): (100, 1000, 1e6), ("dram", "0"): (20, 1000, 1e6)}


def _no_gpu():
    class NVMLError(Exception):
        pass

    def init():
        raise NVMLError("no driver")

    return SimpleNamespace(NVMLError=NVMLError, nvmlInit=init)


def _calibrated(tmp_path, monkeypatch, data) -> Path:
    """The calibration file in tmp_path, holding data measured on this machine."""
    monkeypatch.setattr(calibration, "MACHINE_ID", tmp_path / "machine-id")
    (tmp_path / "machine-id").write_text("abc\n")
    path = tmp_path / "calibration.json"
    if data is not None:
        path.write_text(json.dumps({"machine_id": "abc", **data}))
    return path


def test_idle_power_needs_a_calibration(tmp_path, monkeypatch):
    path = _calibrated(tmp_path, monkeypatch, None)

    with pytest.raises(CalibrationError, match="sudo metrion calibrate"):
        calibration.idle_power("cpu", ["0"], path)


def test_idle_power_refuses_a_device_it_has_not_measured(tmp_path, monkeypatch):
    path = _calibrated(tmp_path, monkeypatch, {"cpu": {"0": 1.3}, "dram": {"0": None}})

    assert calibration.idle_power("dram", ["0"], path) == {"0": None}
    with pytest.raises(CalibrationError, match="cpu 1"):
        calibration.idle_power("cpu", ["0", "1"], path)


def test_a_calibration_from_another_machine_is_refused(tmp_path, monkeypatch):
    path = _calibrated(tmp_path, monkeypatch, {"cpu": {"0": 1.3}})
    (tmp_path / "machine-id").write_text("xyz\n")

    with pytest.raises(CalibrationError, match="another machine"):
        calibration.idle_power("cpu", ["0"], path)


def _orchestrator(components, path):
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator._config = AgentConfig.model_validate(
        {"calibration": str(path), "energy_model": {"components": components}}
    )
    return orchestrator


def test_monitor_stops_before_anything_starts_without_a_calibration(tmp_path, monkeypatch):
    path = _calibrated(tmp_path, monkeypatch, None)

    with pytest.raises(CalibrationError, match=str(path)):
        _orchestrator({}, path).check_calibration()


def test_monitor_needs_no_calibration_when_every_idle_power_is_given(tmp_path, monkeypatch):
    path = _calibrated(tmp_path, monkeypatch, None)
    manual = {"power": [{"id": 0, "idle_power": 5, "max_power": 50}]}

    _orchestrator({"cpu": manual, "dram": manual, "gpu": {"mode": "exclude"}}, path).check_calibration()


def test_the_calibration_path_starts_in_the_working_directory(tmp_path, monkeypatch):
    _calibrated(tmp_path, monkeypatch, {"cpu": {"0": 1.3}, "dram": {"0": None}, "gpu": {}})
    monkeypatch.chdir(tmp_path)

    _orchestrator({}, "./calibration.json").check_calibration()


@pytest.mark.parametrize(
    ("stderr", "shown"),
    [
        ("Traceback (most recent call last):\n  ...\nOSError: sysfs gone\n", "OSError: sysfs gone"),
        ("", "journalctl -u metrion-calibrate"),
    ],
)
def test_a_failed_measurement_says_why(monkeypatch, stderr, shown):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr=stderr))

    with pytest.raises(CalibrationError, match=shown):
        calibration.run(["user.slice"], 10)


def test_a_device_the_calibration_does_not_know_stops_the_discovery(monkeypatch):
    def get_info(params):
        raise CalibrationError("does not know cpu 1")

    monkeypatch.setattr(importlib, "import_module", lambda path: SimpleNamespace(get_info=get_info))

    with pytest.raises(CalibrationError):
        HardwareTreeBuilder().import_hardware_module("bare_metal", "linux", "cpu", {})
