from abc import ABC, abstractmethod
from collections.abc import Sequence

from metrion.config_reader.config_schema import EbpfConfig, TimingConfig
from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.metric import Metric
from metrion.util import check_entity_type


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    This class defines the interface for data sources that monitor or collect data from entities.
    """

    _entity_metric_mapping: dict[type[Entity], set[Metric]] = {}

    @abstractmethod
    def __init__(
        self,
        entities: Sequence[Entity],
        interval: float | None = None,
        enabled_metrics: Sequence[Metric] | None = None,
        timing: TimingConfig | None = None,
        ebpf: EbpfConfig | None = None,
    ):
        """
        Initialize the data source with a list of entities.

        Parameters
        ----------
        entities : Sequence of Entity
            The entities that this data source will monitor or collect data from.
        interval : float or None, optional
            The collection interval in seconds. If None, the interval is unknown or not applicable.
        enabled_metrics : Sequence of Metric, optional
            Metrics enabled in config for this data source's component.
        timing : TimingConfig, optional
            The timing section of the config. Only the sources that run an eBPF producer read it.
        ebpf : EbpfConfig, optional
            The ebpf section of the config. Only SchedSwitchMonitor reads it.

        Notes
        -----
        Subclasses may override this method to set up resources or threading flags.
        """
        self.entities: Sequence[Entity] = entities
        self._interval = interval
        self._enabled_metrics: Sequence[Metric] | None = enabled_metrics
        self._timing = timing or TimingConfig()
        self._ebpf = ebpf or EbpfConfig()
        check_entity_type(entities)

    @abstractmethod
    def get_data(self, metrics: list[Metric]) -> DataRecord:
        """
        Collect data for the specified metrics from the managed entities.

        Parameters
        ----------
        metrics : list of Metric
            The metrics to collect data for.

        Returns
        -------
        DataRecord
            The collected data for the specified metrics and entities.

        Notes
        -----
        Each subclass must implement its own logic for data collection.
        """
        pass

    @classmethod
    def get_metrics_by_entity_type(cls) -> dict[type[Entity], set[Metric]]:
        """
        Get a mapping from entity type to the set of applicable metrics.

        Returns
        -------
        dict of type[Entity] to set[Metric]
            Mapping from entity type to applicable metrics.
        """
        return cls._entity_metric_mapping

    @classmethod
    @abstractmethod
    def is_applicable(cls, entity: Entity) -> bool:
        """
        Determine whether this data source can handle the given entity.

        Parameters
        ----------
        entity : Entity
            The entity to check for compatibility.

        Returns
        -------
        bool
            True if this data source can handle the entity, False otherwise.
        """
        pass

    @abstractmethod
    def clean_up(self) -> None:
        """
        Release resources or perform any necessary cleanup.

        Notes
        -----
        Subclasses should override if cleanup is required.
        """
        pass

    def stop(self, metrics: list[Metric]) -> DataRecord | None:
        """
        Stop or pause data collection for the specified metrics.

        Parameters
        ----------
        metrics : list of Metric
            The metrics for which collection should be stopped.

        Returns
        -------
        DataRecord or None
            Optionally return a final DataRecord containing any remaining data,
            or None if there is no final data to return.

        Notes
        -----
        Subclasses may override to implement shutdown or pausing logic (for example,
        stopping background threads or flushing buffers). The default implementation
        does nothing and returns None.
        """
        return None
