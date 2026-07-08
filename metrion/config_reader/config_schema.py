"""
Pydantic configuration schema for the Metrion agent.

This module defines the configuration structure for:
- Database settings (type, path)
- General settings (debug mode)
- Energy model configuration (granularity, components)
- Component configuration (CPU, GPU, DRAM, Network) with include/exclude modes
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────
# Power configuration
# ─────────────────────────────
PowerValue = float | Literal["auto"]


@dataclass
class PowerConfig:
    idle_power: PowerValue = "auto"
    max_power: PowerValue = "auto"


# ─────────────────────────────
# Database section
# ─────────────────────────────
class DatabaseConfig(BaseModel):
    """Database configuration for storing collected metrics."""

    type: Literal["sqlite", "duckdb"] = "duckdb"
    path: str = "./monitor.db"

    def get_connection_string(self, base_path: Path | None = None) -> str:
        """
        Generate the database connection string.

        Parameters
        ----------
        base_path : Path, optional
            Base path for resolving relative database paths.

        Returns
        -------
        str
            Database connection string in the format 'type:///path'.
        """
        if base_path is None:
            base_path = Path(__file__).parents[2]

        db_path = Path(self.path)
        if not db_path.is_absolute():
            db_path = base_path / db_path

        return f"{self.type}:///{db_path}"


# ─────────────────────────────
# General section
# ─────────────────────────────
class GeneralConfig(BaseModel):
    """General configuration settings."""

    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    debug: bool = False
    collection_interval: float = Field(default=5.0, gt=0, description="Data collection interval in seconds")


# ─────────────────────────────
# Subcomponent power configuration
# ─────────────────────────────
class SubcomponentPowerConfig(BaseModel):
    """
    Power configuration for a hardware subcomponent.

    This is used ONLY for defining power characteristics (idle/max power)
    of specific subcomponents like CPU packages or GPUs.
    """

    id: str | int
    idle_power: PowerValue = "auto"
    max_power: PowerValue = "auto"

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str | int) -> str | int:
        """
        Validate subcomponent ID format.

        Accepts:
        - Numeric IDs (for GPUs, etc.)
        - Package IDs in format 'package-<n>' (for RAPL)
        - Network interface names 'eth<n>', 'wlan<n>', 'ens<n>', 'enp<s>', etc.

        Parameters
        ----------
        v : str | int
            The ID value to validate.

        Returns
        -------
        str | int
            The validated ID.

        Raises
        ------
        ValueError
            If the ID format is invalid.
        """
        if isinstance(v, int):
            return v

        # Pattern for valid string IDs
        valid_patterns = [
            r"^package-\d+$",  # RAPL package IDs
            r"^eth\d+$",  # Traditional ethernet
            r"^wlan\d+$",  # Wireless
            r"^ens\d+$",  # Systemd predictable naming
            r"^enp\d+s\d+.*$",  # PCI-based naming
            r"^lo$",  # Loopback
            r"^\d+$",  # Pure numeric string
        ]

        if any(re.match(pattern, str(v)) for pattern in valid_patterns):
            return v

        raise ValueError(
            f"Invalid id format: '{v}'. Expected numeric ID, 'package-<n>', "
            "or network interface name (eth<n>, wlan<n>, ens<n>, enp<s>, etc.)"
        )

    @field_validator("idle_power", "max_power", mode="before")
    @classmethod
    def validate_power(cls, v: str | float | int) -> PowerValue:
        """
        Validate and normalize power values.

        Parameters
        ----------
        v : str | float | int
            The power value to validate.

        Returns
        -------
        PowerValue
            Normalized power value (float or 'auto').
        """
        if isinstance(v, str):
            v = v.strip().lower()
            if v == "auto":
                return "auto"
            try:
                return float(v)
            except ValueError as exc:
                raise ValueError(f"Invalid power value: '{v}'. Expected 'auto' or a numeric value.") from exc
        return float(v)


# Type for subcomponents: either "auto" for all, or a list of specific configs
SubcomponentsConfig = Literal["auto"] | list[SubcomponentPowerConfig]

# Type for metrics: either "*" for all metrics, or a list of specific metric names
MetricsConfig = Literal["*"] | list[str]


# ─────────────────────────────
# Component configuration
# ─────────────────────────────
class ComponentConfig(BaseModel):
    """
    Configuration for a hardware component type (CPU, GPU, DRAM, Network).

    The `mode` field controls whether this component is monitored.
    The `power` field defines power characteristics and can be:
    - "auto": Auto-detect power values for all subcomponents
    - A list of SubcomponentPowerConfig for specific power settings
    The `metrics` field defines which metrics to collect:
    - "*": Collect all applicable metrics for this component
    - A list of metric names: Collect only the specified metrics
    """

    mode: Literal["include", "exclude"] = "include"
    power: SubcomponentsConfig = "auto"
    metrics: MetricsConfig = "*"

    @field_validator("power", mode="before")
    @classmethod
    def validate_power(cls, v):
        """Validate and normalize the power configuration."""
        if v is None:
            return "auto"
        if isinstance(v, str):
            if v.lower() == "auto":
                return "auto"
            raise ValueError(f"Invalid energy value: '{v}'. Expected 'auto' or a list of subcomponent configs.")
        return v

    @field_validator("metrics", mode="before")
    @classmethod
    def validate_metrics(cls, v):
        """Validate and normalize the metrics configuration."""
        if v is None:
            return "*"
        if isinstance(v, str):
            if v == "*":
                return "*"
            raise ValueError(f"Invalid metrics value: '{v}'. Expected '*' or a list of metric names.")
        if isinstance(v, list):
            return [str(m) for m in v]
        return v


# ─────────────────────────────
# Energy model section
# ─────────────────────────────
class EnergyModelConfig(BaseModel):
    """Configuration for the energy model and component settings."""

    granularity: Literal["compressed", "detailed"] = "detailed"
    components: dict[str, ComponentConfig] = Field(default_factory=dict)

    def is_component_included(self, component_name: str) -> bool:
        """
        Check if a component should be included in monitoring.

        Parameters
        ----------
        component_name : str
            Name of the component (cpu, gpu, dram, network).

        Returns
        -------
        bool
            True if the component should be included.
        """
        component_name = component_name.lower()
        if component_name not in self.components:
            return True  # Default to include if not specified

        return self.components[component_name].mode == "include"

    def get_component_config(self, component_name: str) -> ComponentConfig | None:
        """
        Get configuration for a specific component.

        Parameters
        ----------
        component_name : str
            Name of the component.

        Returns
        -------
        ComponentConfig | None
            Component configuration or None if not found.
        """
        return self.components.get(component_name.lower())

    def get_component_metrics(self, component_name: str) -> MetricsConfig:
        """
        Get the metrics configuration for a specific component.

        Parameters
        ----------
        component_name : str
            Name of the component (cpu, gpu, dram, network).

        Returns
        -------
        MetricsConfig
            Either "*" for all metrics or a list of specific metric names.
        """
        component_name = component_name.lower()
        if component_name not in self.components:
            return "*"  # Default to all metrics if not specified

        return self.components[component_name].metrics


# ─────────────────────────────
# Root configuration
# ─────────────────────────────
class AgentConfig(BaseModel):
    """Root configuration for the Metrion agent."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    energy_model: EnergyModelConfig = Field(default_factory=EnergyModelConfig)

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "AgentConfig":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        AgentConfig
            Parsed configuration object.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        ValidationError
            If the configuration is invalid.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls.model_validate(config_dict)

    def get_database_connection_string(self, base_path: Path | None = None) -> str:
        """
        Get the database connection string.

        Parameters
        ----------
        base_path : Path, optional
            Base path for resolving relative paths.

        Returns
        -------
        str
            Database connection string.
        """
        return self.general.database.get_connection_string(base_path)

    def get_subcomponent_power_config(self, component_type: str, subcomponent_id: str | int) -> PowerConfig:
        """
        Get power configuration for a specific subcomponent.

        Parameters
        ----------
        component_type : str
            Type of component (cpu, gpu, dram, network).
        subcomponent_id : str | int
            ID of the subcomponent.

        Returns
        -------
        tuple[PowerValue, PowerValue]
            Tuple of (idle_power, max_power) values.
        """
        component = self.energy_model.get_component_config(component_type)
        if component is None:
            return PowerConfig("auto", "auto")

        # If power is "auto", return auto for all subcomponents
        if component.power == "auto":
            return PowerConfig("auto", "auto")
        # Search for specific subcomponent config
        for subcomp in component.power:
            if str(subcomp.id) == str(subcomponent_id):
                return PowerConfig(subcomp.idle_power, subcomp.max_power)

        return PowerConfig("auto", "auto")

    def check_if_component_included(self, component_type: str) -> bool:
        """
        Check if a component type is included in monitoring.

        Parameters
        ----------
        component_type : str
            Type of component (cpu, gpu, dram, network).

        Returns
        -------
        bool
            True if the component is included, False otherwise.
        """
        return self.energy_model.is_component_included(component_type)
