"""
Unit tests for the new configuration schema.
"""

import re
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from metrion.config_reader import (
    AgentConfig,
    ComponentConfig,
    ConfigReader,
    EbpfConfig,
    EnergyModelConfig,
    SourceIntervals,
    StorageConfig,
    SubcomponentPowerConfig,
    TimingConfig,
)


class TestStorageConfig:
    """Tests for StorageConfig, which replaced general.database."""

    def test_default_values(self):
        config = StorageConfig()
        assert config.type == "duckdb"
        assert config.path == "./monitor.db"

    def test_custom_values(self):
        config = StorageConfig(type="parquet", path="/custom/path/run")
        assert config.type == "parquet"
        assert config.path == "/custom/path/run"

    def test_relative_path_is_resolved_against_the_base(self):
        config = StorageConfig(type="duckdb", path="./test.db")
        assert config.get_target(Path("/base")) == "duckdb:////base/test.db"

    def test_absolute_path_is_kept(self):
        config = StorageConfig(type="duckdb", path="/absolute/path/db.duckdb")
        assert config.get_target() == "duckdb:////absolute/path/db.duckdb"

    def test_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            StorageConfig(type="mongodb")

    def test_rejects_the_dropped_backends(self):
        for dropped in ("sqlite", "postgresql"):
            with pytest.raises(ValidationError):
                StorageConfig(type=dropped)


class TestSubcomponentEnergyConfig:
    """Tests for SubcomponentEnergyConfig."""

    def test_numeric_id(self):
        """Test numeric ID validation."""
        config = SubcomponentPowerConfig(id=0, idle_power=10.0, max_power=100.0)
        assert config.id == 0
        assert config.idle_power == 10.0
        assert config.max_power == 100.0

    def test_package_id(self):
        """Test package-N ID format."""
        config = SubcomponentPowerConfig(id="package-0", idle_power="auto", max_power="auto")
        assert config.id == "package-0"
        assert config.idle_power == "auto"
        assert config.max_power == "auto"

    def test_network_interface_ids(self):
        """Test network interface ID formats."""
        for iface_id in ["eth0", "wlan0", "ens33", "enp0s3"]:
            config = SubcomponentPowerConfig(id=iface_id, idle_power="auto", max_power="auto")
            assert config.id == iface_id

    def test_invalid_id_format(self):
        """Test that invalid ID formats raise an error."""
        with pytest.raises(ValueError, match="Invalid id format"):
            SubcomponentPowerConfig(id="invalid-format", idle_power="auto", max_power="auto")

    def test_power_value_normalization(self):
        """Test that power values are normalized correctly."""
        # String "auto" should stay as "auto"
        config = SubcomponentPowerConfig(id=0, idle_power="auto", max_power="AUTO")
        assert config.idle_power == "auto"
        assert config.max_power == "auto"

        # Numeric strings should be converted to float
        config = SubcomponentPowerConfig(id=0, idle_power=10.5, max_power=100)
        assert config.idle_power == 10.5
        assert config.max_power == 100.0


class TestComponentConfig:
    """Tests for ComponentConfig."""

    def test_default_mode(self):
        """Test default component mode is 'include' and power is 'auto'."""
        config = ComponentConfig()
        assert config.mode == "include"
        assert config.power == "auto"

    def test_exclude_mode(self):
        """Test exclude mode."""
        config = ComponentConfig(mode="exclude")
        assert config.mode == "exclude"

    def test_with_power_auto(self):
        """Test component with auto power detection."""
        config = ComponentConfig(mode="include", power="auto")
        assert config.power == "auto"

    def test_with_power_list(self):
        """Test component with explicit power configuration list."""
        config = ComponentConfig(
            mode="include",
            power=[
                SubcomponentPowerConfig(id="package-0", idle_power=1.0, max_power=50.0),
                SubcomponentPowerConfig(id="package-1", idle_power=1.1, max_power=55.0),
            ],
        )
        assert isinstance(config.power, list)
        assert len(config.power) == 2
        assert config.power[0].id == "package-0"


class TestEnergyModelConfig:
    """Tests for EnergyModelConfig."""

    def test_default_values(self):
        """Test default energy model configuration."""
        config = EnergyModelConfig()
        assert config.granularity == "detailed"
        assert config.components == {}

    def test_is_component_included_default(self):
        """Test component inclusion check for undefined components."""
        config = EnergyModelConfig()
        # Components not in config default to included
        assert config.is_component_included("cpu") is True
        assert config.is_component_included("gpu") is True

    def test_is_component_included_explicit(self):
        """Test explicit component inclusion/exclusion."""
        config = EnergyModelConfig(
            components={
                "cpu": ComponentConfig(mode="include"),
                "gpu": ComponentConfig(mode="exclude"),
            }
        )
        assert config.is_component_included("cpu") is True
        assert config.is_component_included("gpu") is False

    def test_get_component_config(self):
        """Test getting component configuration."""
        comp_config = ComponentConfig(mode="include")
        config = EnergyModelConfig(components={"cpu": comp_config})

        assert config.get_component_config("cpu") == comp_config
        assert config.get_component_config("nonexistent") is None


class TestSourceIntervals:
    """Tests for timing.sources."""

    def test_defaults_match_previous_hardcoded_values(self):
        """EntityTypeMonitor._setup used 1.0 s for the energy sources and 5.0 s for the rest."""
        s = SourceIntervals()
        assert (s.rapl, s.nvml, s.proc_numa, s.sched_switch, s.tcp_monitor) == (1.0, 1.0, 5.0, 5.0, 5.0)

    def test_one_field_per_data_source(self):
        """A data source is collected as a whole, so its metrics cannot have separate intervals."""
        assert set(SourceIntervals.model_fields) == {"rapl", "nvml", "proc_numa", "sched_switch", "tcp_monitor"}

    def test_section_is_optional(self):
        """A config without a timing section keeps every default."""
        config = AgentConfig.model_validate({"energy_model": {"components": {}}})
        assert config.timing.sources.rapl == 1.0

    def test_rapl_below_grouping_limit_is_rejected(self):
        """Below 0.25 s the guard band would be wider than the usable window."""
        with pytest.raises(ValidationError):
            SourceIntervals(rapl=0.2)

    def test_only_rapl_has_a_lower_bound(self):
        """The 0.25 s limit comes from the grouping bucket width, which only rapl sets."""
        assert SourceIntervals(sched_switch=0.1).sched_switch == 0.1

    def test_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            SourceIntervals(sched_switch=-1)

    def test_values_are_read_from_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(
            "timing:\n  cycle: 5\n  sources:\n    rapl: 0.5\n    sched_switch: 2.0\n"
            "energy_model:\n  components:\n    cpu:\n      mode: include\n      metrics: '*'\n"
        )

        config = ConfigReader.read_config(str(path))

        assert config.timing.sources.rapl == 0.5
        assert config.timing.sources.sched_switch == 2.0
        assert config.timing.sources.proc_numa == 5.0, "fields left out keep their default"

    def test_misspelled_interval_is_rejected(self):
        """A typo must not silently leave the default in place."""
        with pytest.raises(ValidationError):
            SourceIntervals.model_validate({"tcp_moniter": 5.0})

    def test_shipped_config_yaml_is_valid(self):
        """configs/config.yaml is the file users start from, so it has to validate."""
        shipped = Path(__file__).parents[2] / "configs" / "config.yaml"

        config = AgentConfig.from_yaml_file(shipped)

        assert config.timing.cycle == 10
        assert (config.timing.ebpf_flush, config.timing.storage_flush) == (1.0, 2.0)
        assert config.timing.sources.rapl == 1.0
        assert config.timing.sources.tcp_monitor == 5.0
        assert config.ebpf.ringbuf_mib == 256


class TestEbpfConfig:
    def test_default_is_256_mib(self):
        assert EbpfConfig().ringbuf_mib == 256

    @pytest.mark.parametrize("mib", [0, 48, -64])
    def test_rejects_what_the_kernel_would_reject(self, mib):
        """The kernel only takes a power of two; anything else fails later as an opaque BPF error."""
        with pytest.raises(ValidationError):
            EbpfConfig(ringbuf_mib=mib)

    def test_is_read_from_yaml(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("ebpf:\n  ringbuf_mib: 256\nenergy_model:\n  components: {}\n")

        assert AgentConfig.from_yaml_file(path).ebpf.ringbuf_mib == 256


class TestTimingConfig:
    """Tests for the timing section."""

    def test_defaults(self):
        t = TimingConfig()
        assert (t.cycle, t.ebpf_flush, t.storage_flush) == (5.0, 1.0, 2.0)
        assert t.sources.rapl == 1.0

    def test_defaults_match_the_previously_hardcoded_values(self):
        """1.0 s stood in sched_switch.py and tcp_monitor.py, 2.0 s was the IpcAsyncWriter default."""
        t = TimingConfig()
        assert (t.ebpf_flush, t.storage_flush) == (1.0, 2.0)

    def test_cycle_must_be_positive(self):
        with pytest.raises(ValidationError):
            TimingConfig(cycle=0)

    def test_cycle_is_read_from_yaml(self, tmp_path):
        """collection_interval moved out of general into timing.cycle."""
        path = tmp_path / "c.yaml"
        path.write_text("timing:\n  cycle: 12\nenergy_model:\n  components: {}\n")

        assert AgentConfig.from_yaml_file(path).timing.cycle == 12


class TestStrictSections:
    """Unknown keys have to fail loudly - a silently ignored option is the bug we keep hitting."""

    def test_every_model_forbids_extra_keys(self):
        """Catches a section added later without extra="forbid"."""
        import metrion.config_reader.config_schema as schema

        models = [
            obj
            for obj in vars(schema).values()
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
        ]

        assert models, "no models found - did the module move?"
        assert [m.__name__ for m in models if m.model_config.get("extra") != "forbid"] == []

    def test_a_config_from_before_the_rename_is_rejected(self):
        """general.database and sampling are gone; an unmigrated config must not start silently."""
        old_style = {
            "general": {"database": {"type": "duckdb", "path": "./monitor.db"}, "collection_interval": 10},
            "sampling": {"rapl": 1.0},
            "energy_model": {"components": {}},
        }

        with pytest.raises(ValidationError) as exc:
            AgentConfig.model_validate(old_style)

        assert {e["loc"][0] for e in exc.value.errors()} == {"general", "sampling"}

    def test_misspelled_component_is_rejected(self):
        """A typo here used to be silent: the entry was ignored and the real component stayed on."""
        with pytest.raises(ValidationError):
            EnergyModelConfig.model_validate({"components": {"dramm": {"mode": "exclude"}}})

    def test_known_components_are_accepted(self):
        config = EnergyModelConfig.model_validate(
            {"components": {name: {"mode": "include"} for name in ("cpu", "gpu", "dram", "network", "host")}}
        )
        assert len(config.components) == 5


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default agent configuration."""
        config = AgentConfig()
        assert config.storage.type == "duckdb"
        assert config.debug is False
        assert config.timing.cycle == 5.0
        assert config.energy_model.granularity == "detailed"

    def test_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
debug: true

storage:
  type: duckdb
  path: ./test.db

timing:
  cycle: 10

energy_model:
  granularity: compressed
  components:
    cpu:
      mode: include
      power:
        - id: package-0
          idle_power: 1.5
          max_power: 50.0
    gpu:
      mode: exclude
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AgentConfig.from_yaml_file(f.name)

            assert config.storage.type == "duckdb"
            assert config.debug is True
            assert config.timing.cycle == 10.0
            assert config.energy_model.granularity == "compressed"
            assert config.energy_model.is_component_included("cpu") is True
            assert config.energy_model.is_component_included("gpu") is False

            # Check energy config
            cpu_config = config.energy_model.get_component_config("cpu")
            assert cpu_config is not None
            assert isinstance(cpu_config.power, list)
            assert len(cpu_config.power) == 1
            assert cpu_config.power[0].idle_power == 1.5

    def test_from_yaml_file_with_auto_power(self):
        """Test loading configuration with auto power detection."""
        yaml_content = """
debug: false

storage:
  type: duckdb
  path: ./test.db

timing:
  cycle: 5

energy_model:
  granularity: detailed
  components:
    cpu:
      mode: include
      power: auto
    gpu:
      mode: exclude
      power: auto
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = AgentConfig.from_yaml_file(f.name)

            cpu_config = config.energy_model.get_component_config("cpu")
            assert cpu_config.power == "auto"

            gpu_config = config.energy_model.get_component_config("gpu")
            assert gpu_config.power == "auto"

    def test_get_subcomponent_power_config(self):
        """Test getting power configuration for subcomponents."""
        config = AgentConfig(
            energy_model=EnergyModelConfig(
                components={
                    "cpu": ComponentConfig(
                        mode="include",
                        power=[
                            SubcomponentPowerConfig(id="package-0", idle_power=1.0, max_power=50.0),
                        ],
                    ),
                }
            )
        )

        power_config = config.get_subcomponent_power_config("cpu", "package-0")
        assert power_config.idle_power == 1.0
        assert power_config.max_power == 50.0

        # Non-existent subcomponent returns auto
        power_config = config.get_subcomponent_power_config("cpu", "package-99")
        assert power_config.idle_power == "auto"
        assert power_config.max_power == "auto"

        # Non-existent component returns auto
        power_config = config.get_subcomponent_power_config("nonexistent", "0")
        assert power_config.idle_power == "auto"
        assert power_config.max_power == "auto"

    def test_get_subcomponent_power_config_with_auto(self):
        """Test getting power config when power is set to auto."""
        config = AgentConfig(
            energy_model=EnergyModelConfig(
                components={
                    "cpu": ComponentConfig(mode="include", power="auto"),
                }
            )
        )

        # When power is "auto", all subcomponents should return auto
        power_config = config.get_subcomponent_power_config("cpu", "package-0")
        assert power_config.idle_power == "auto"
        assert power_config.max_power == "auto"


class TestConfigReader:
    """Tests for ConfigReader."""

    def test_read_config(self):
        """Test reading configuration through ConfigReader."""
        yaml_content = """
debug: false

storage:
  type: duckdb
  path: ./test.db

timing:
  cycle: 5

energy_model:
  granularity: detailed
  components:
    cpu:
      mode: include
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)

            assert isinstance(config, AgentConfig)
            assert config.storage.type == "duckdb"
            assert config.debug is False

    def test_the_storage_path_gets_a_timestamp(self):
        yaml_content = """
debug: false

storage:
  type: duckdb
  path: ./test.db

timing:
  cycle: 5

energy_model:
  granularity: detailed
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)

            assert config.storage.path != "test.db"
            assert re.fullmatch(r"test_\d{4}(_\d{2}){5}\.db", config.storage.path)

    def test_a_parquet_directory_keeps_its_suffixless_name(self):
        yaml_content = """
storage:
  type: parquet
  path: ./runs/prod

timing:
  cycle: 5

energy_model:
  granularity: detailed
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)

            assert re.fullmatch(r"runs/prod_\d{4}(_\d{2}){5}", config.storage.path)


class TestMetricsConfig:
    """Tests for metrics configuration in ComponentConfig."""

    def test_default_metrics_is_all(self):
        """Test default metrics configuration is '*' (all metrics)."""
        config = ComponentConfig()
        assert config.metrics == "*"

    def test_metrics_all_wildcard(self):
        """Test setting metrics to '*' for all metrics."""
        config = ComponentConfig(metrics="*")
        assert config.metrics == "*"

    def test_metrics_specific_list(self):
        """Test setting specific metrics list."""
        metrics_list = ["CPU_SOCKET_PKG_ENERGY", "CPU_CORE_PID_TIME"]
        config = ComponentConfig(metrics=metrics_list)
        assert config.metrics == metrics_list

    def test_metrics_none_defaults_to_all(self):
        """Test that None metrics value defaults to '*'."""
        config = ComponentConfig(metrics=None)
        assert config.metrics == "*"

    def test_invalid_metrics_string(self):
        """Test that invalid string metrics value raises error."""
        with pytest.raises(ValueError, match="Invalid metrics value"):
            ComponentConfig(metrics="invalid")

    def test_energy_model_get_component_metrics_default(self):
        """Test getting default metrics for component not in config."""
        energy_config = EnergyModelConfig(components={})
        assert energy_config.get_component_metrics("cpu") == "*"

    def test_energy_model_get_component_metrics_wildcard(self):
        """Test getting wildcard metrics config."""
        energy_config = EnergyModelConfig(components={"cpu": ComponentConfig(metrics="*")})
        assert energy_config.get_component_metrics("cpu") == "*"

    def test_energy_model_get_component_metrics_list(self):
        """Test getting specific metrics list."""
        metrics_list = ["CPU_SOCKET_PKG_ENERGY", "CPU_CORE_PID_TIME"]
        energy_config = EnergyModelConfig(components={"cpu": ComponentConfig(metrics=metrics_list)})
        assert energy_config.get_component_metrics("cpu") == metrics_list

    def test_yaml_config_with_metrics_wildcard(self):
        """Test loading YAML config with metrics wildcard."""
        yaml_content = """
debug: false

storage:
  type: duckdb
  path: ./test.db

energy_model:
  components:
    cpu:
      mode: include
      metrics: "*"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)
            assert config.energy_model.get_component_metrics("cpu") == "*"

    def test_yaml_config_with_metrics_list(self):
        """Test loading YAML config with specific metrics list."""
        yaml_content = """
debug: false

storage:
  type: duckdb
  path: ./test.db

energy_model:
  components:
    cpu:
      mode: include
      metrics:
        - CPU_SOCKET_PKG_ENERGY
        - CPU_CORE_PID_TIME
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)
            metrics = config.energy_model.get_component_metrics("cpu")
            assert isinstance(metrics, list)
            assert "CPU_SOCKET_PKG_ENERGY" in metrics
            assert "CPU_CORE_PID_TIME" in metrics


class TestLiveAttributionConfig:
    def test_a_window_shorter_than_the_slowest_energy_source_is_rejected(self):
        """Such windows can hold no energy sample and would report 0 J although the device ran."""
        with pytest.raises(ValidationError, match="window"):
            AgentConfig(live_attribution={"enabled": True, "window": 2.0}, timing={"sources": {"nvml": 5.0}})

    def test_a_cutoff_shorter_than_the_waiting_steps_is_rejected(self):
        """Measurements would arrive after their interval was attributed and be dropped as late."""
        # Default timing: ebpf_flush 1 + 2 x storage_flush 2 + slowest source 5 + cycle 5 = 15 s
        with pytest.raises(ValidationError, match="cutoff"):
            AgentConfig(live_attribution={"enabled": True, "cutoff": 10.0})

        assert AgentConfig(live_attribution={"cutoff": 10.0}).live_attribution.cutoff == 10.0, "only checked when on"

    def test_dropping_the_raw_stream_without_live_attribution_is_rejected(self):
        """Neither stored nor attributed, so the run would produce nothing and only an empty file shows it."""
        with pytest.raises(ValidationError, match="discard every measurement"):
            AgentConfig(storage={"store_raw_measurements": False})

    def test_dropping_the_raw_stream_with_live_attribution_is_allowed(self):
        config = AgentConfig(storage={"store_raw_measurements": False}, live_attribution={"enabled": True})
        assert config.storage.store_raw_measurements is False
