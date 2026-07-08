"""
Unit tests for the new configuration schema.
"""

import tempfile
from pathlib import Path

import pytest

from metrion.config_reader import (
    AgentConfig,
    ComponentConfig,
    ConfigReader,
    DatabaseConfig,
    EnergyModelConfig,
    SubcomponentPowerConfig,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default database configuration values."""
        config = DatabaseConfig()
        assert config.type == "duckdb"
        assert config.path == "./monitor.db"

    def test_custom_values(self):
        """Test custom database configuration values."""
        config = DatabaseConfig(type="sqlite", path="/custom/path/db.sqlite")
        assert config.type == "sqlite"
        assert config.path == "/custom/path/db.sqlite"

    def test_connection_string(self):
        """Test connection string generation."""
        config = DatabaseConfig(type="duckdb", path="./test.db")
        conn_str = config.get_connection_string(Path("/base"))
        assert conn_str == "duckdb:////base/test.db"

    def test_connection_string_absolute_path(self):
        """Test connection string with absolute path."""
        config = DatabaseConfig(type="sqlite", path="/absolute/path/db.sqlite")
        conn_str = config.get_connection_string()
        assert conn_str == "sqlite:////absolute/path/db.sqlite"


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


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default agent configuration."""
        config = AgentConfig()
        assert config.general.database.type == "duckdb"
        assert config.general.debug is False
        assert config.general.collection_interval == 5.0
        assert config.energy_model.granularity == "detailed"

    def test_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
general:
  database:
    type: sqlite
    path: ./test.db
  debug: true
  collection_interval: 10

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

            assert config.general.database.type == "sqlite"
            assert config.general.debug is True
            assert config.general.collection_interval == 10.0
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
general:
  database:
    type: duckdb
    path: ./test.db
  debug: false
  collection_interval: 5

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
general:
  database:
    type: duckdb
    path: ./test.db
  debug: false
  collection_interval: 5

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
            assert config.general.database.type == "duckdb"
            assert config.general.debug is False

    def test_read_config_with_debug_mode(self):
        """Test that debug mode adds timestamp to database path."""
        yaml_content = """
general:
  database:
    type: duckdb
    path: ./test.db
  debug: true
  collection_interval: 5

energy_model:
  granularity: detailed
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = ConfigReader.read_config(f.name)

            # In debug mode, the path should have a timestamp added
            assert "test_" in config.general.database.path
            assert ".db" in config.general.database.path


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
general:
  database:
    type: duckdb
    path: ./test.db
  debug: false

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
general:
  database:
    type: duckdb
    path: ./test.db
  debug: false

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
