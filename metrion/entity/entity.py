import hashlib
import uuid
from abc import ABC
from copy import deepcopy

import polars as pl


class Entity(ABC):  # noqa: B024
    """
    Represents a generic entity that can generate measurements.

    Attributes
    ----------
    id : str
        Unique identifier for the entity (derived from unique attributes).
    name : str
        Human-readable name of the entity.
    type : str
        Type of the entity (e.g., "Host", "Process").
    parent_id: str | None
        Identifier of the parent entity (e.g., a process is a child of a host).
    """

    def __init__(
        self,
        id: str | None = None,
        parent_id: str | None = None,
    ):
        """
        Initializes an Entity instance.

        Parameters
        ----------
        parent_id : str | None
            Identifier of the parent entity (e.g., a process is a child of a host).
        """
        self._parent_id = parent_id
        t = id if id is not None else self._generate_id()
        self._id = deepcopy(t)

    def _generate_id(self) -> str:
        """
        Generates a deterministic ID based on entity attributes.

        Returns
        -------
        str
            A unique identifier for the entity.
        """
        return str(uuid.uuid4().int)

    def _to_df(self) -> pl.DataFrame:
        """Converts the entity to a Polars DataFrame."""
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    @property
    def parent_id(self) -> str | None:
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: str | None):
        self._parent_id = value

    def __eq__(self, value):
        return self.id == value.id

    def __hash__(self):
        hashed_id = int(hashlib.md5(self.id.encode("utf-8")).hexdigest(), 16)
        return hashed_id

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id})"

    def __repr__(self):
        return self.__str__()
