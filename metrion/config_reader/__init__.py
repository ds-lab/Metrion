"""
Configuration reader package for Metrion agent.

This package provides configuration loading and validation using Pydantic.
"""

from metrion.config_reader.config_reader import ConfigReader, read_config
from metrion.config_reader.config_schema import (
    AgentConfig,
    ComponentConfig,
    EnergyModelConfig,
    LiveAttributionConfig,
    SourceIntervals,
    StorageConfig,
    SubcomponentPowerConfig,
    TimingConfig,
)

__all__ = [
    "ConfigReader",
    "read_config",
    "AgentConfig",
    "StorageConfig",
    "TimingConfig",
    "SourceIntervals",
    "EnergyModelConfig",
    "ComponentConfig",
    "SubcomponentPowerConfig",
    "LiveAttributionConfig",
]
