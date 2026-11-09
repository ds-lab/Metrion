"""
Configuration reader package for Metrion agent.

This package provides configuration loading and validation using Pydantic.
"""

from metrion.config_reader.config_reader import ConfigReader, read_config
from metrion.config_reader.config_schema import (
    AgentConfig,
    ComponentConfig,
    EbpfConfig,
    EnergyModelConfig,
    LiveAttributionConfig,
    MemoryEventsConfig,
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
    "EbpfConfig",
    "MemoryEventsConfig",
    "SourceIntervals",
    "EnergyModelConfig",
    "ComponentConfig",
    "SubcomponentPowerConfig",
    "LiveAttributionConfig",
]
