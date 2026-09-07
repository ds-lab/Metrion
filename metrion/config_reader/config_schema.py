"""
Pydantic configuration schema for the Metrion agent.

This module defines the configuration structure for:
- Storage settings (type, path)
- Timing: the collection cycle and the interval of every data source
- Energy model configuration (granularity, components)
- Component configuration (CPU, GPU, DRAM, Network) with include/exclude modes

Every section rejects unknown keys, so a misspelled option fails at startup instead of
silently leaving the default in place.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ─────────────────────────────
# Power configuration
# ─────────────────────────────
PowerValue = float | Literal["auto"]


@dataclass
class PowerConfig:
    idle_power: PowerValue = "auto"
    max_power: PowerValue = "auto"


# ─────────────────────────────
# Storage section
# ─────────────────────────────
class StorageConfig(BaseModel):
    """Where the collected data is written."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sqlite", "duckdb"] = "duckdb"
    path: str = "./monitor.db"

    def get_target(self, base_path: Path | None = None) -> str:
        """
        Build the target that the storage backend is opened with.

        Parameters
        ----------
        base_path : Path, optional
            Root for a relative path. Defaults to the repository root.

        Returns
        -------
        str
            Connection string in the format 'type:///path'.
        """
        if base_path is None:
            base_path = Path(__file__).parents[2]

        target = Path(self.path)
        if not target.is_absolute():
            target = base_path / target

        return f"{self.type}:///{target}"


# ─────────────────────────────
# Timing section
# ─────────────────────────────
class SourceIntervals(BaseModel):
    """
    Collection interval of every data source, in seconds.

    One field per data source, not per metric: a data source is collected as a whole, so all its
    metrics share one interval.

    This section is keyed by data source, energy_model.components is keyed by component. The two
    do not line up: RAPLCPU serves both the cpu and the dram component, while cpu is served by
    RAPLCPU and SchedSwitchMonitor together. energy_model decides which sources run at all, this
    section decides how often each of them runs.

    The meaning differs between the two kinds of source. For the polled sources (rapl, nvml,
    proc_numa) the interval is the sample rate — every read produces one measurement. For the
    eBPF sources (sched_switch, tcp_monitor) the kernel probe records every event continuously
    and a producer process buffers them; the interval only says how often the agent picks up
    what has been buffered. Raising it costs freshness, never events.
    """

    model_config = ConfigDict(extra="forbid")

    # Polled sources: one read, one sample.
    rapl: float = Field(default=1.0, ge=0.25, description="Read the RAPL energy counters (CPU and DRAM package)")
    nvml: float = Field(default=1.0, gt=0, description="Query NVML (device energy and per-process GPU utilization)")
    proc_numa: float = Field(default=5.0, gt=0, description="Parse /proc/<pid>/numa_maps for the per-NUMA footprint")

    # eBPF sources: the kernel records continuously, this is the pickup interval.
    sched_switch: float = Field(default=5.0, gt=0, description="Pick up buffered scheduler events (CPU time per PID)")
    tcp_monitor: float = Field(default=5.0, gt=0, description="Pick up buffered socket events (traffic per PID)")


class TimingConfig(BaseModel):
    """
    Everything that decides how often something happens, in seconds.

    The fields add up to the delay between an event in the kernel and the moment its energy is
    attributed, so this is the section to change when latency matters.
    """

    model_config = ConfigDict(extra="forbid")

    cycle: float = Field(default=5.0, gt=0, description="Drain the data source queues and hand a batch to storage")
    ebpf_flush: float = Field(default=1.0, gt=0, description="eBPF producer: buffered events -> IPC writer")
    storage_flush: float = Field(default=2.0, gt=0, description="IPC writer: buffered rows -> file on disk")
    sources: SourceIntervals = Field(default_factory=SourceIntervals)


# ─────────────────────────────
# Subcomponent power configuration
# ─────────────────────────────
class SubcomponentPowerConfig(BaseModel):
    """
    Power configuration for a hardware subcomponent.

    This is used ONLY for defining power characteristics (idle/max power)
    of specific subcomponents like CPU packages or GPUs.
    """

    model_config = ConfigDict(extra="forbid")

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

# The components METRION knows about. Anything else in energy_model.components is a typo, and a
# typo there is silent: the misspelled entry is ignored and the real component keeps its default.
ComponentName = Literal["cpu", "gpu", "dram", "network", "host"]


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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    granularity: Literal["compressed", "detailed"] = "detailed"
    components: dict[ComponentName, ComponentConfig] = Field(default_factory=dict)

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

    model_config = ConfigDict(extra="forbid")

    debug: bool = False
    storage: StorageConfig = Field(default_factory=StorageConfig)
    timing: TimingConfig = Field(default_factory=TimingConfig)
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
