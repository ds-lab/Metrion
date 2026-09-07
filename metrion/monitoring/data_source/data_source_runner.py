import queue
import threading
import time
import uuid
from collections.abc import Sequence

from loguru import logger

from metrion.config_reader.config_schema import TimingConfig
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
        timing: TimingConfig | None = None,
    ):
        super().__init__(daemon=True)
        self._datasource = datasource(entities, interval=interval, enabled_metrics=metrics, timing=timing)
        self._metrics = metrics
        self._interval = interval
        self._stop_event = threading.Event()
        self._queue = queue
        self._final_data_records = DataRecord(entities=[], measurements=[])
        self._done_event = threading.Event()
        self._id = uuid.uuid4()
        self._late_ticks = 0

    def run(self):
        try:
            start = time.monotonic()
            tick = 0

            while not self._stop_event.is_set():
                data = self._datasource.get_data(self._metrics)
                self._datasource.clean_up()
                self._queue.put_nowait(data)

                # Wake up on the fixed grid start + tick * interval, not interval seconds after
                # the work happens to finish. Otherwise the period is interval + work time.
                tick += 1
                sleep_for = start + tick * self._interval - time.monotonic()

                if sleep_for < 0:
                    # The work took longer than the interval. Skip forward to the next grid
                    # point that is still ahead of us instead of falling further behind.
                    behind = int(-sleep_for // self._interval) + 1
                    tick += behind
                    self._late_ticks += behind
                    sleep_for = max(0.0, start + tick * self._interval - time.monotonic())

                if self._stop_event.wait(sleep_for):
                    logger.debug("DataSourceRunner stopped initiated")
                    break

            self._finalize()

        finally:
            if self._late_ticks:
                logger.warning(
                    f"DataSourceRunner {self._id}: work took longer than the "
                    f"{self._interval}s interval, the schedule was realigned {self._late_ticks} times"
                )
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

    def get_final_data_records(self, timeout: float = 20.0) -> DataRecord:
        # ensures caller won’t read partial results
        if not self._done_event.wait(timeout):
            logger.warning(f"DataSourceRunner {self._id}: did not finalize within {timeout}s")
        return self._final_data_records
