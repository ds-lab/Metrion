import hashlib
import os
import uuid
from datetime import UTC, datetime
from typing import Any, override

import polars as pl
from snowflake import SnowflakeGenerator


class Measurement:
    _GEN: None | SnowflakeGenerator = None

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
        now = datetime.now()

        self._value = value
        self._metric_id = metric_id
        self._physical_entity_id = physical_entity_id
        self._logical_entity_id = logical_entity_id
        self._start_ts = start_ts if start_ts else now
        self._end_ts = end_ts if end_ts else now

        # Create the snowflake ID
        gen = self.get_snowflake_generator()
        id = next(gen)
        assert id is not None
        self._id: int = id

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
        """Converts the measurement to a Polars DataFrame."""
        schema = {
            "metric_id": pl.String,
            "physical_entity_id": pl.String,
            "logical_entity_id": pl.String,
            "start_ts": pl.Datetime,
            "end_ts": pl.Datetime,
            "value": pl.Float64,
            "id": pl.Int64,
        }

        data = {
            "metric_id": [self._metric_id],
            "physical_entity_id": [self._physical_entity_id],
            "logical_entity_id": [self._logical_entity_id],
            "start_ts": [self._start_ts.replace(tzinfo=UTC)],
            "end_ts": [self._end_ts.replace(tzinfo=UTC)],
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
    def get_snowflake_generator(cls) -> SnowflakeGenerator:
        if cls._GEN is None:
            host_uuid = cls.get_host_uuid()

            u = uuid.UUID(host_uuid).bytes if "-" in host_uuid else host_uuid.encode()
            digest = hashlib.sha256(u).digest()

            instance_id = int.from_bytes(digest[:2], "big") & 1023
            cls._GEN = SnowflakeGenerator(instance=instance_id)

        return cls._GEN


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
