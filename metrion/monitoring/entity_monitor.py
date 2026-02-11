import queue as q
from collections.abc import Sequence

from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.metric import CPU_SOCKET_PKG_ENERGY, DRAM_SOCKET_TOTAL_ENERGY, GPU_DEVICE_TOTAL_ENERGY, Metric
from metrion.monitoring.data_source.data_source_factory import DataSourceFactory
from metrion.monitoring.data_source.data_source_runner import DataSourceRunner
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
    """

    def __init__(self, entities: Sequence[Entity], metrics: list[Metric]):
        """
        Initialize the EntityTypeMonitor.

        Parameters
        ----------
        entities : Sequence[Entity]
            The entities to monitor.
        metrics : list of Metric
            The metrics to collect for the entities.
        """
        self._entities = entities
        self._metrics = metrics

        check_entity_type(entities)
        DataSourceFactory.auto_register()
        self._data_source_metrics = DataSourceFactory.get_sources_by_entity_type_and_metric_filter(
            type(self._entities[0]), self._metrics
        )

        self._data_source_runners: dict[DataSourceRunner, q.Queue] = {}

        self._setup()
        self._start()

    def _setup(self):
        """
        Set up data source runners and their associated queues.
        """
        for data_source, metrics in self._data_source_metrics.items():
            queue = q.Queue()
            if (
                CPU_SOCKET_PKG_ENERGY in metrics
                or GPU_DEVICE_TOTAL_ENERGY in metrics
                or DRAM_SOCKET_TOTAL_ENERGY in metrics
            ):
                runner = DataSourceRunner(data_source, self._entities, metrics, 1.0, queue)
            else:
                runner = DataSourceRunner(data_source, self._entities, metrics, 5.0, queue)
            self._data_source_runners[runner] = queue

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
                merged_measurements.extend(data_record.measurements)
                merged_entities.extend(data_record.entities)

                queue.task_done()

        return DataRecord(entities=merged_entities, measurements=merged_measurements)

    def _drain_queue_for_runner(self, runner: DataSourceRunner) -> DataRecord:
        """
        Drain the queue for a specific runner.

        Parameters
        ----------
        runner : DataSourceRunner
            The runner to drain the queue for.

        Returns
        -------
        DataRecord
            A DataRecord containing merged entities and measurements from the runner's queue.
        """
        data_record = DataRecord(entities=[], measurements=[])
        queue = self._data_source_runners[runner]
        while True:
            try:
                data_record_tmp = queue.get_nowait()
                if data_record_tmp == DataSourceRunner.SENT_END:
                    queue.task_done()
                    break
                data_record.entities.extend(data_record_tmp.entities)
                data_record.measurements.extend(data_record_tmp.measurements)
                queue.task_done()
            except q.Empty:
                continue

        return data_record

    def stop(self) -> DataRecord | None:
        """
        Stop all data source runners and wait for their queues to be processed.
        """
        from loguru import logger

        final_data_records = DataRecord(entities=[], measurements=[])
        for runner in self._data_source_runners.keys():
            if runner.is_alive():
                logger.debug(f"Stopping DataSourceRunner from entity monitor {runner._id}")
                runner.request_stop()
                logger.debug(f"Waiting for DataSourceRunner {runner._id} to finish")
                drain_data = self._drain_queue_for_runner(runner)
                final_data_record = runner.get_final_data_records()
                logger.debug(f"DataSourceRunner {runner._id} finished")

                if final_data_record:
                    final_data_records.entities.extend(final_data_record.entities)
                    final_data_records.measurements.extend(final_data_record.measurements)

                if drain_data:
                    final_data_records.entities.extend(drain_data.entities)
                    final_data_records.measurements.extend(drain_data.measurements)

        for runner, queue in self._data_source_runners.items():
            logger.debug(f"Waiting for DataSourceRunner {runner._id} to join")
            runner.join(timeout=60.0)
            logger.debug(f"Waiting for DataSourceRunner {runner._id} queue to join")
            queue.join()
            logger.debug(f"DataSourceRunner {runner._id} joined")

        return final_data_records
