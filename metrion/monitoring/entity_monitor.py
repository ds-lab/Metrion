import queue as q
from collections.abc import Sequence

from loguru import logger

from metrion.config_reader.config_schema import EbpfConfig, TimingConfig
from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.metric import Metric
from metrion.monitoring.data_source.data_source_factory import DataSourceFactory
from metrion.monitoring.data_source.data_source_runner import DataSourceRunner
from metrion.queue_util import drain_until_sentinel
from metrion.util import check_entity_type


class EntityTypeMonitor:
    """
    Monitors a set of entities and collects metric data from registered data sources.

    Parameters
    ----------
    entities : list of Entity
        The entities to monitor.
    metrics : list of Metric
        The metrics to collect for the entities.
    timing : TimingConfig, optional
        The timing section of the config. Defaults to TimingConfig().
    ebpf : EbpfConfig, optional
        The ebpf section of the config. Defaults to EbpfConfig().
    """

    # One config field per data source class. The names are deliberately spelled out because the
    # classes follow no shared naming pattern.
    _INTERVAL_FIELD_BY_SOURCE = {
        "RAPLCPU": "rapl",
        "NVMLDataSource": "nvml",
        "ProcNuma": "proc_numa",
        "SchedSwitchMonitor": "sched_switch",
        "TCPDataSource": "tcp_monitor",
    }

    def __init__(
        self,
        entities: Sequence[Entity],
        metrics: list[Metric],
        timing: TimingConfig | None = None,
        ebpf: EbpfConfig | None = None,
    ):
        """
        Initialize the EntityTypeMonitor.

        Parameters
        ----------
        entities : Sequence[Entity]
            The entities to monitor.
        metrics : list of Metric
            The metrics to collect for the entities.
        timing : TimingConfig, optional
            The timing section of the config. Defaults to TimingConfig().
        ebpf : EbpfConfig, optional
            The ebpf section of the config. Defaults to EbpfConfig().
        """
        self._entities = entities
        self._metrics = metrics
        self._timing = timing or TimingConfig()
        self._ebpf = ebpf or EbpfConfig()

        check_entity_type(entities)
        DataSourceFactory.auto_register()
        self._data_source_metrics = DataSourceFactory.get_sources_by_entity_type_and_metric_filter(
            type(self._entities[0]), self._metrics
        )

        self._data_source_runners: dict[DataSourceRunner, q.Queue] = {}

        self._setup()
        self._start()

    def _interval_for(self, data_source: type) -> float:
        """Collection interval for a data source, taken from timing.sources."""
        name = data_source.__name__
        field = self._INTERVAL_FIELD_BY_SOURCE.get(name)
        if field is None:
            logger.warning(f"No collection interval configured for {name}, using 5.0 s")
            return 5.0
        return getattr(self._timing.sources, field)

    def _setup(self):
        """
        Set up data source runners and their associated queues.
        """
        for data_source, metrics in self._data_source_metrics.items():
            queue = q.Queue()
            interval = self._interval_for(data_source)
            runner = DataSourceRunner(data_source, self._entities, metrics, interval, queue, self._timing, self._ebpf)
            self._data_source_runners[runner] = queue
            logger.info(f"{data_source.__name__}: collection interval {interval} s")

    def _start(self):
        """
        Start all data source runners.
        """
        for runner in self._data_source_runners:
            runner.start()

    def get_data(self) -> DataRecord:
        """
        Retrieve and merge data from all data source runners.

        Returns
        -------
        DataRecord
            A DataRecord containing merged entities and measurements from all sources.
        """
        merged_measurements = []
        merged_entities = []
        for queue in self._data_source_runners.values():
            while not queue.empty():
                data_record = queue.get_nowait()
                if data_record == DataSourceRunner.SENT_END:
                    # A runner that already finished left its marker here. Acknowledge it,
                    # otherwise queue.join() would never return.
                    queue.task_done()
                    continue
                merged_measurements.extend(data_record.measurements)
                merged_entities.extend(data_record.entities)

                queue.task_done()

        return DataRecord(entities=merged_entities, measurements=merged_measurements)

    def _drain_queue_for_runner(self, runner: DataSourceRunner, deadline_s: float = 15.0) -> DataRecord:
        """
        Drain the queue of one runner up to its SENT_END marker or the deadline.

        Parameters
        ----------
        runner : DataSourceRunner
            The runner to drain the queue for.
        deadline_s : float
            Maximum time to wait for SENT_END, in seconds.

        Returns
        -------
        DataRecord
            A DataRecord containing merged entities and measurements from the runner's queue.
        """
        queue = self._data_source_runners[runner]

        items, seen = drain_until_sentinel(
            queue, DataSourceRunner.SENT_END, deadline_s=deadline_s, name=f"runner-{runner._id}"
        )
        if not seen:
            logger.warning(f"DataSourceRunner {runner._id}: SENT_END not received within {deadline_s}s")

        entities = []
        measurements = []
        for item in items:
            entities.extend(item.entities)
            measurements.extend(item.measurements)
            queue.task_done()
        if seen:
            queue.task_done()

        return DataRecord(entities=entities, measurements=measurements)

    def stop(self) -> DataRecord | None:
        """
        Stop all data source runners and collect what they produced.
        """
        # Ask every runner to stop first, so they wind down in parallel.
        for runner in self._data_source_runners:
            logger.debug(f"Stopping DataSourceRunner from entity monitor {runner._id}")
            runner.request_stop()

        entities = []
        measurements = []
        for runner in self._data_source_runners:
            # Drain dead runners too: their SENT_END and data sit in the queue either way.
            drain_data = self._drain_queue_for_runner(runner)
            entities.extend(drain_data.entities)
            measurements.extend(drain_data.measurements)

            final_data_record = runner.get_final_data_records()
            if final_data_record:
                entities.extend(final_data_record.entities)
                measurements.extend(final_data_record.measurements)
            logger.debug(f"DataSourceRunner {runner._id} finished")

        for runner in self._data_source_runners:
            runner.join(timeout=20.0)
            if runner.is_alive():
                logger.warning(f"DataSourceRunner {runner._id} did not terminate, giving up on it")

        return DataRecord(entities=entities, measurements=measurements)
