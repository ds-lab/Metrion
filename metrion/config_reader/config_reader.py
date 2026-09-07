"""
Configuration reader module for loading and parsing Metrion agent configuration.

This module provides functionality to load YAML configuration files and
convert them into validated AgentConfig objects using Pydantic.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from metrion.config_reader.config_schema import AgentConfig


class ConfigReader:
    """
    Reader class for loading and parsing configuration files.

    This class provides methods to read YAML configuration files and
    return validated AgentConfig objects.
    """

    @classmethod
    def read_config(cls, config_path: str | Path) -> AgentConfig:
        """
        Read and validate configuration from a YAML file.

        Parameters
        ----------
        config_path : str | Path
            Path to the YAML configuration file.

        Returns
        -------
        AgentConfig
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        pydantic.ValidationError
            If the configuration is invalid.
        """
        config = AgentConfig.from_yaml_file(config_path)

        # Configure logging level based on debug setting
        cls._configure_logging(config.debug)

        # Apply debug mode transformations if needed
        if config.debug:
            config = cls._apply_debug_transformations(config)

        logger.info(f"Configuration loaded from {config_path}")
        logger.debug(f"Storage: {config.storage.type} at {config.storage.path}")
        logger.debug(f"Collection cycle: {config.timing.cycle}s")
        logger.debug(f"Energy model granularity: {config.energy_model.granularity}")

        return config

    @classmethod
    def _configure_logging(cls, debug: bool) -> None:
        """
        Configure loguru logging level based on debug setting.

        Parameters
        ----------
        debug : bool
            If True, set logging level to DEBUG. Otherwise, set to INFO.
        """
        # Remove default handler and add new one with appropriate level
        logger.remove()
        log_level = "DEBUG" if debug else "INFO"
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",  # noqa: E501
        )

    @classmethod
    def _apply_debug_transformations(cls, config: AgentConfig) -> AgentConfig:
        """
        Append a timestamp to the storage path so debug runs do not overwrite each other.

        Parameters
        ----------
        config : AgentConfig
            The configuration to modify.

        Returns
        -------
        AgentConfig
            The same object, with storage.path rewritten.
        """
        original_path = Path(config.storage.path)
        timestamp = datetime.now(UTC).strftime("_%Y_%m_%d_%H_%M_%S")
        config.storage.path = str(original_path.with_stem(original_path.stem + timestamp))

        return config


# Backwards compatibility - expose read_config as module-level function
def read_config(config_path: str | Path) -> AgentConfig:
    """
    Read and validate configuration from a YAML file.

    This is a convenience function that wraps ConfigReader.read_config.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    AgentConfig
        Validated configuration object.
    """
    return ConfigReader.read_config(config_path)
