import hashlib
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any, override

import numpy as np
import polars as pl

# Snowflake layout: 41 bits of Unix milliseconds, 10 bits of instance, 12 bits of sequence.
_SEQ_BITS = 12
_SEQ_MASK = (1 << _SEQ_BITS) - 1
_INSTANCE_BITS = 10
_INSTANCE_MASK = (1 << _INSTANCE_BITS) - 1


class Measurement:
    _instance_id: int | None = None
    # Millisecond and sequence of the last id handed out, as one number: ms << _SEQ_BITS | seq.
    _last_slot: int = -1
    _ids_lock = threading.Lock()

    """
    Represents a measurement that may be either a single point in time or an interval.

    Attributes
    ----------
    id : int
        Unique identifier for the measurement.
    metric : Metric
        The metric associated with this measurement.
    entity_id: int
        The entity ID associated with this measurement.
    value : any
        The measured value.
    start_ts : Optional[datetime]
        The start time of the measurement (None if it's a point measurement).
    end_ts : datetime
        The end time of the measurement (defaults to now).
    """

    def __init__(
        self,
        value: Any,
        metric_id: str,
        physical_entity_id: str,
        logical_entity_id: str | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ):
        now = datetime.now(UTC)

        self._value = value
        self._metric_id = metric_id
        self._physical_entity_id = physical_entity_id
        self._logical_entity_id = logical_entity_id
        self._start_ts = start_ts if start_ts else now
        self._end_ts = end_ts if end_ts else now

        self._id: int = int(self.next_ids(1)[0])

    @property
    def value(self) -> Any:
        """Returns the measured value."""
        return self._value

    @property
    def metric_id(self) -> str:
        return self._metric_id

    @property
    def physical_entity_id(self) -> str:
        """Returns the physical entity ID associated with this measurement."""
        return self._physical_entity_id

    @property
    def logical_entity_id(self) -> str | None:
        """Returns the logical entity ID associated with this measurement."""
        return self._logical_entity_id

    @property
    def start_ts(self) -> datetime:
        """Returns the start time of the measurement."""
        return self._start_ts

    @property
    def end_ts(self) -> datetime:
        """Returns the end time of the measurement."""
        return self._end_ts

    @property
    def is_interval(self) -> bool:
        """Returns True if this is an interval measurement."""
        return not self._start_ts == self._end_ts

    @property
    def duration(self) -> float | None:
        """Returns the duration in seconds if this is an interval measurement."""
        return (self._end_ts - self._start_ts).total_seconds() if self.is_interval else None

    @property
    def id(self) -> int:
        """Returns the unique identifier of the measurement."""
        return self._id

    def to_df(self) -> pl.DataFrame:
        """Converts the measurement to a Polars DataFrame with naive UTC timestamps."""
        schema = {
            "metric_id": pl.String,
            "physical_entity_id": pl.String,
            "logical_entity_id": pl.String,
            "start_ts": pl.Datetime("us"),
            "end_ts": pl.Datetime("us"),
            "value": pl.Float64,
            "id": pl.Int64,
        }

        data = {
            "metric_id": [self._metric_id],
            "physical_entity_id": [self._physical_entity_id],
            "logical_entity_id": [self._logical_entity_id],
            "start_ts": [self._start_ts],
            "end_ts": [self._end_ts],
            "value": [self._value],
            "id": [self._id],
        }
        return pl.DataFrame(data, schema=schema)

    def __str__(self):
        return (
            f"Measurement(metric={self._metric_id}, value={self._value},"
            + f"physical_entity_id={self._physical_entity_id}, logical_entity_id={self._logical_entity_id},"
            + f"start_ts={self._start_ts}, end_ts={self._end_ts})"
        )

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get_host_uuid(cls) -> str:
        host_uuid = os.environ.get("METRION_HOST_ID", None)
        if host_uuid is None:
            host_uuid = str(uuid.uuid4())
            os.environ["METRION_HOST_ID"] = host_uuid
        return host_uuid

    @classmethod
    def get_instance_id(cls) -> int:
        if cls._instance_id is None:
            host_uuid = cls.get_host_uuid()

            u = uuid.UUID(host_uuid).bytes if "-" in host_uuid else host_uuid.encode()
            digest = hashlib.sha256(u).digest()

            cls._instance_id = int.from_bytes(digest[:2], "big") & _INSTANCE_MASK

        return cls._instance_id

    @classmethod
    def next_ids(cls, n: int) -> np.ndarray:
        """
        Draw n snowflake ids as an int64 array, unique and increasing within this process.

        A millisecond holds 4096 ids. A batch larger than what is left of the current millisecond
        continues into the following ones instead of waiting for them.
        """
        instance = cls.get_instance_id()
        with cls._ids_lock:
            now_slot = (time.time_ns() // 1_000_000) << _SEQ_BITS
            first = max(now_slot, cls._last_slot + 1)
            cls._last_slot = first + n - 1

        slots = np.arange(first, first + n, dtype=np.int64)
        millis = slots >> _SEQ_BITS
        return (millis << (_INSTANCE_BITS + _SEQ_BITS)) | (instance << _SEQ_BITS) | (slots & _SEQ_MASK)


class CommunicationMeasurement(Measurement):
    """
    Represents a communication measurement between two entities.

    Attributes
    ----------
    src_endpoint : str
        The source endpoint ID.
    dst_endpoint : str
        The destination endpoint ID.
    """

    def __init__(
        self,
        value: Any,
        metric_id: str,
        physical_entity_id: str,
        logical_entity_id: str,
        src_endpoint: str,
        dst_endpoint: str,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
    ):
        super().__init__(
            value=value,
            metric_id=metric_id,
            physical_entity_id=physical_entity_id,
            logical_entity_id=logical_entity_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        self._src_endpoint = src_endpoint
        self._dst_endpoint = dst_endpoint

    @override
    def to_df(self) -> pl.DataFrame:
        """Converts the communication measurement to a Polars DataFrame."""
        base_df = super().to_df()
        comm_schema = {
            "src_endpoint": pl.String,
            "dst_endpoint": pl.String,
        }
        comm_data = {
            "src_endpoint": [self._src_endpoint],
            "dst_endpoint": [self._dst_endpoint],
        }
        comm_df = pl.DataFrame(comm_data, schema=comm_schema)
        return base_df.hstack(comm_df)

    @property
    def src_endpoint(self) -> str:
        """Returns the source endpoint ID."""
        return self._src_endpoint

    @property
    def dst_endpoint(self) -> str:
        """Returns the destination endpoint ID."""
        return self._dst_endpoint

    def __str__(self):
        return (
            f"CommunicationMeasurement(metric={self.metric_id}, value={self.value},"
            + f"src_endpoint={self._src_endpoint}, dst_endpoint={self._dst_endpoint},"
            + f"start_ts={self.start_ts}, end_ts={self.end_ts})"
        )
