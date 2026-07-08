import queue
import threading
import uuid
from collections.abc import Sequence

from loguru import logger

from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.metric import Metric
from metrion.monitoring.data_source.data_source_base import DataSource


class DataSourceRunner(threading.Thread):
    SENT_END = "__SENT_END__"

    def __init__(
        self,
        datasource: type[DataSource],
        entities: Sequence[Entity],
        metrics: list[Metric],
        interval: float,
        queue: queue.Queue,
    ):
        super().__init__(daemon=True)
        self._datasource = datasource(entities, interval=interval, enabled_metrics=metrics)
        self._metrics = metrics
        self._interval = interval
        self._stop_event = threading.Event()
        self._queue = queue
        self._final_data_records = DataRecord(entities=[], measurements=[])
        self._done_event = threading.Event()
        self._id = uuid.uuid4()

    def run(self):
        try:
            while not self._stop_event.is_set():
                data = self._datasource.get_data(self._metrics)
                self._datasource.clean_up()
                self._queue.put_nowait(data)

                if self._stop_event.wait(self._interval):
                    logger.debug("DataSourceRunner stopped initiated")
                    break

            self._finalize()

        finally:
            self._queue.put(self.SENT_END)
            self._done_event.set()
            logger.debug("DataSourceRunner stopped")

    def _finalize(self):
        logger.debug("Finalizing DataSourceRunner")
        final_data_records = self._datasource.stop(self._metrics)
        if final_data_records:
            self._final_data_records.entities.extend(final_data_records.entities)
            self._final_data_records.measurements.extend(final_data_records.measurements)
        self._datasource.clean_up()
        logger.debug("DataSourceRunner finalized")

    def request_stop(self):
        self._stop_event.set()

    def get_final_data_records(self) -> DataRecord:
        # ensures caller won’t read partial results
        self._done_event.wait()
        return self._final_data_records
