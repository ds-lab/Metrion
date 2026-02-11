import os
import signal
import sys
import time
from pathlib import Path
from time import sleep

import click
from loguru import logger

from metrion.config_reader.config_reader import ConfigReader
from metrion.config_reader.config_schema import AgentConfig
from metrion.data_collector import DataCollector
from metrion.data_record import DataRecord
from metrion.discovery.hardware_detection import HardwareDetection
from metrion.entity.host import Host
from metrion.entity.logical_entity_registry import LogicalEntityRegistry
from metrion.entity.physical_entity import PhysicalEntity
from metrion.metric import Metric
from metrion.monitoring.entity_monitor import EntityTypeMonitor


class Orchestrator:
    """
    Orchestrates the configuration, initialization, and monitoring of hardware entities.

    This class manages the lifecycle of data collection, storage, and monitoring for various physical entities
    (CPU, GPU, RAM, Host) as defined in the configuration.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the Orchestrator.

        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file.
        """
        self._config_path = config_path
        self._config: AgentConfig | None = None
        self._data_collector: DataCollector | None = None
        self._entities: dict[type[PhysicalEntity], list[PhysicalEntity]] = {}
        self._entity_monitors: list[EntityTypeMonitor] = []
        self._running = False
        self._main_pid = os.getpid()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGQUIT, self._signal_handler)

    def start(self, timeout_minutes: float | None = None) -> None:
        """
        Start the orchestrator: load config, initialize components, and begin monitoring loop.

        Parameters
        ----------
        timeout_minutes : float | None
            Optional timeout in minutes. If None, monitoring runs indefinitely until interrupted.
        """
        self.read_config(self._config_path)
        self.start_logical_entity_registry()
        self.create_data_collector()
        self.create_physical_entities()
        self.save_physical_entities()
        self.create_entity_monitors()
        self.start_monitoring(timeout_minutes=timeout_minutes)

    def stop(self) -> None:
        """
        Stop monitoring and cleanup resources gracefully.

        This method stops all entity monitors, the data collector, and closes the data storage connection.
        """
        self._running = False
        # Stop all entity monitors
        for entity_monitor in self._entity_monitors:
            try:
                logger.debug(f"Stopping entity monitor for {entity_monitor._entities[0].__class__.__name__}")
                final_records = entity_monitor.stop()
                logger.debug(
                    f"Entity monitor for {entity_monitor._entities[0].__class__.__name__} stopped successfully!"
                )
                if final_records:
                    self.save_data_record(final_records)
                logger.debug(
                    "Final data records for entity monitor for"
                    + f"{entity_monitor._entities[0].__class__.__name__} saved successfully!"
                )
            except Exception as e:
                logger.error(f"Error stopping entity monitor for {entity_monitor._entities[0].__class__.__name__}: {e}")

        # final save of all logical entites
        new_entites = self._logical_entity_registry.stop()
        self.save_data_record(DataRecord(new_entites, []))

        # Stop data collector
        if self._data_collector:
            try:
                self._data_collector.stop()
            except Exception as e:
                logger.error(f"Error stopping data collector: {e}")

        logger.info("Orchestrator stopped successfully")

    def _signal_handler(self, signum, frame):
        """
        Handle SIGINT and SIGTERM signals for graceful shutdown.

        Parameters
        ----------
        signum : int
            Signal number received.
        frame : FrameType
            Current stack frame (unused).
        """
        if os.getpid() != self._main_pid:
            return
        # Prevent any more of our handler running
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGQUIT, signal.SIG_IGN)

        logger.info("Attempting graceful shutdown…")
        self.stop()

        # exit the process cleanly now that we’ve cleaned up
        sys.exit(0)

    def read_config(self, path: str) -> None:
        """
        Read and parse the configuration file.

        Parameters
        ----------
        path : str
            Path to the configuration YAML file.
        """
        self._config = ConfigReader.read_config(path)

    def start_logical_entity_registry(
        self,
    ) -> None:
        # logical entity registry
        self._logical_entity_registry = LogicalEntityRegistry.get_instance()

    def create_data_collector(self) -> None:
        """
        Create the data collector using the configured storage strategy and data storage.
        """
        assert self._config is not None, "Config must be loaded before creating data collector."
        storage_type = self._config.general.database.type
        storage_path = self._config.get_database_connection_string(Path(__file__).parent.parent)

        self._data_collector = DataCollector(storage_type=storage_type, storage_path=storage_path)

    def create_physical_entities(self) -> None:
        """
        Detect and create physical entities (CPU, GPU, RAM, Host, NIC) using hardware detection.
        """
        assert self._config is not None, "Config must be loaded before creating entities."
        hardware_detection = HardwareDetection(self._config)
        self._entities = hardware_detection.get_physical_entities()

    def save_physical_entities(self) -> None:
        """
        Save the detected physical entities to the data storage.
        """
        for entities in self._entities.values():
            self.save_data_record(DataRecord(entities=entities, measurements=[]))

    def create_entity_monitors(self) -> None:
        """
        Create entity monitors for each physical entity type based on the config.

        Only creates monitors for components that are set to 'include' mode in the
        energy_model configuration.
        """
        assert self._config is not None, "Config must be loaded before creating entity monitors."

        for entity_type, entities in self._entities.items():
            entity_type_name = entity_type.__name__.lower()
            # Host is always included
            if entity_type is Host:
                desired_metrics = self._get_metrics_for_component(entity_type)
            elif self._config.energy_model.is_component_included(entity_type_name):
                desired_metrics = self._get_metrics_for_component(entity_type)
            else:
                logger.debug(f"Skipping {entity_type_name} monitoring (excluded in config)")
                continue

            if len(desired_metrics) > 0:
                self._entity_monitors.append(EntityTypeMonitor(entities, desired_metrics))
                logger.debug(
                    f"Created monitor for {entity_type_name} with metrics: {[m.name for m in desired_metrics]}"
                )

    def _get_metrics_for_component(self, entity_type: type) -> list[Metric]:
        """
        Get metrics for a component type based on configuration.

        Parameters
        ----------
        entity_type : type
            The entity type (Host, CPU, GPU, DRAM, Network).

        Returns
        -------
        list[Metric]
            List of metrics to collect for this component.
        """
        assert self._config is not None, "Config must be loaded before getting metrics."

        # Get all metrics that match this component prefix
        entity_name = entity_type.__name__.lower()
        all_metrics = Metric.get_all_metrics()

        # Get the metrics configuration for this component
        metrics_config = self._config.energy_model.get_component_metrics(entity_name)
        applicable_metrics = []

        if metrics_config == "*":
            # Include all metrics for this component
            for metric in all_metrics.values():
                if metric.is_applicable(entity_type):
                    applicable_metrics.append(metric)
        else:
            # Include only the specified metrics
            for metric_name in metrics_config:
                metric = all_metrics.get(metric_name)
                if metric is not None:
                    # Verify the metric is applicable to this component
                    if metric.is_applicable(entity_type):
                        applicable_metrics.append(metric)
                    else:
                        logger.warning(
                            f"Metric '{metric_name}' is not applicable to component '{entity_name}', skipping"
                        )
                else:
                    logger.warning(f"Unknown metric '{metric_name}' specified for component '{entity_name}'")

        return applicable_metrics

    def start_monitoring(self, timeout_minutes: float | None = None) -> None:
        """
        Start the main monitoring loop.

        This method iterates over all entity monitors, collects data, and saves it at intervals specified in the config.
        The loop continues until a shutdown signal is received or the timeout is reached.

        Parameters
        ----------
        timeout_minutes : float | None
            Optional timeout in minutes. If None, monitoring runs indefinitely until interrupted.
        """
        assert self._config is not None, "Config must be loaded before starting monitoring."
        self._running = True

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60 if timeout_minutes is not None else None

        if timeout_minutes is not None:
            logger.info(
                f"Starting monitoring loop with {timeout_minutes} minute(s) timeout. Press Ctrl+C to stop gracefully."
            )
        else:
            logger.info("Starting monitoring loop. Press Ctrl+C to stop gracefully.")

        while self._running:
            # Check timeout
            if timeout_seconds is not None:
                logger.warning("Timeout is set. Monitoring will stop after the specified duration.")
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning("Timeout reached. Stopping monitoring.")
                    logger.info(f"Timeout of {timeout_minutes} minute(s) reached. Stopping monitoring.")
                    self.stop()
                    break
            time_start = time.time()
            self._logical_entity_registry.update()
            for entity_monitor in self._entity_monitors:
                if not self._running:  # Check for shutdown during iteration
                    break
                try:
                    m_data_record = entity_monitor.get_data()
                    self.save_data_record(m_data_record)
                except Exception as e:
                    logger.error(f"Error measuring {entity_monitor._entities[0].__class__}: {e}")
                    if not self._running:
                        break

            self._logical_entity_registry.update()
            new_entites = self._logical_entity_registry.get_new_entities()
            self.save_data_record(DataRecord(new_entites, []))

            time_end = time.time()
            logger.debug(f"Total Iteration took {time_end - time_start} seconds")

            # Sleep in small increments to check for shutdown signal more frequently
            if self._config.general.collection_interval and self._running:
                sleep_time = self._config.general.collection_interval
                while sleep_time > 0 and self._running:
                    sleep_increment = min(0.05, sleep_time)  # Sleep in 50ms increments for faster response
                    sleep(sleep_increment)
                    sleep_time -= sleep_increment

    def save_data_record(self, data_record: DataRecord) -> None:
        """
        Save a data record using the data collector.

        Parameters
        ----------
        data_record : DataRecord
            The data record to be saved.
        """
        assert self._config is not None, "Config must be loaded before starting monitoring."
        assert self._data_collector is not None, "Data collector must be initialized before saving."

        self._data_collector.add(data_record)


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    default=str(Path(__file__).parent.parent / "configs" / "config.yaml"),
    help="Path to the configuration file.",
)
@click.option(
    "--timeout",
    "-t",
    "timeout_minutes",
    type=float,
    default=None,
    help="Timeout in minutes. If not specified, monitoring runs indefinitely until interrupted with Ctrl+C.",
)
def cli(config_path: str, timeout_minutes: float | None):
    """Metrion - A Linux agent for monitoring process-level resource usage metrics.

    Start the monitoring agent to collect energy and resource usage data.
    The agent will run until interrupted with Ctrl+C or until the optional timeout is reached.

    Examples:

        metrion                           # Run indefinitely with default config

        metrion --timeout 5               # Run for 5 minutes

        metrion -c /path/to/config.yaml   # Use custom config file

        metrion -t 10 -c custom.yaml      # Run for 10 minutes with custom config
    """
    orchestrator = Orchestrator(config_path)
    orchestrator.start(timeout_minutes=timeout_minutes)


def main():
    """Entry point for the metrion CLI."""
    cli()


if __name__ == "__main__":
    main()
