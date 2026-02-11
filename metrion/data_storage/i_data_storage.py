from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.measurement import Measurement


class DataStorageI(ABC):
    """
    Interface for data storage classes.
    """

    @abstractmethod
    def save_data_record(self, data_record: DataRecord, progressbar: bool = False):
        """
        Save a DataRecord to the storage.

        Parameters
        ----------
        data_record : DataRecord
            The DataRecord to save.
        progressbar : bool, optional
            Whether to display a progressbar, by default False

        Raises
        ------
        NotImplementedError
            If the method is not overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save_entities(self, entities: Sequence[Entity], progressbar: bool = False):
        """
        Save entities to the storage.

        Parameters
        ----------
        entities : Sequence[Entity]
            The entities to save.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def save_measurements(self, measurements: Sequence[Measurement | pl.DataFrame], progressbar: bool = False):
        """
        Save measurements to the storage.

        Parameters
        ----------
        measurements : Sequence[Measurement]
            The measurements to save.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def close(self):
        """
        Close the data storage connection.

        Raises
        ------
        NotImplementedError
            If the method is not overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
