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
        cls._configure_logging(config.general.debug)

        # Apply debug mode transformations if needed
        if config.general.debug:
            config = cls._apply_debug_transformations(config)

        logger.info(f"Configuration loaded from {config_path}")
        logger.debug(f"Database type: {config.general.database.type}")
        logger.debug(f"Collection interval: {config.general.collection_interval}s")
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
        Apply debug-mode specific transformations to the configuration.

        In debug mode, the database path is modified to include a timestamp
        to avoid overwriting previous data.

        Parameters
        ----------
        config : AgentConfig
            The original configuration.

        Returns
        -------
        AgentConfig
            Modified configuration with debug transformations applied.
        """
        # Add timestamp to database path in debug mode
        original_path = Path(config.general.database.path)
        timestamp = datetime.now(UTC).strftime("_%Y_%m_%d_%H_%M_%S")
        new_path = original_path.with_stem(original_path.stem + timestamp)

        # Create a new config with the modified path
        config_dict = config.model_dump()
        config_dict["general"]["database"]["path"] = str(new_path)

        return AgentConfig.model_validate(config_dict)


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
