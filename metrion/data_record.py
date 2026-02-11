from collections.abc import Sequence

import polars as pl

from metrion.entity.entity import Entity
from metrion.entity.metadata import Metadata
from metrion.measurement import Measurement


class DataRecord:
    def __init__(
        self,
        entities: Sequence[Entity],
        measurements: Sequence[Measurement | pl.DataFrame],
        metadata: Sequence[Metadata] | None = None,
    ) -> None:
        # assume the dataframe has the following columns:
        # id : int Unique identifier for the measurement.
        # metric : Metric The metric associated with this measurement -> the name as unique identifier.
        # physical_entity_id: int The entity ID associated with this measurement.
        # logical_entity_id: str | None The logical entity ID associated with this measurement.
        # value : any The measured value.
        # start_ts : Optional[datetime] The start time of the measurement (None if it's a point measurement).
        # end_ts : datetime The end time of the measurement (defaults to now).

        self._entities = entities
        self._measurements = measurements
        self._metadata = metadata or []

    @property
    def entities(self) -> Sequence[Entity]:
        return self._entities

    @property
    def measurements(self) -> Sequence[Measurement | pl.DataFrame]:
        return self._measurements

    @property
    def metadata(self) -> Sequence[Metadata]:
        return self._metadata

    def __str__(self) -> str:
        """Detailed representation of the DataRecord."""
        entities_str = "\n    ".join(repr(entity) for entity in self._entities[:5])
        if len(self._entities) > 5:
            entities_str += "\n    ..."

        measurements_str = "\n    ".join(repr(measurement) for measurement in self._measurements[:5])
        if len(self._measurements) > 5:
            measurements_str += "\n    ..."

        return f"DataRecord(\n  entities=[\n    {entities_str}\n  ],\n  measurements=[\n    {measurements_str}\n  ]\n)"
