from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class Metadata:
    """
    A flexible metadata entry that can be attached to any object.

    Each metadata record stores its own unique identifier, a reference to the
    object it belongs to, a type string describing the metadata, and the value
    itself (stored as a string for persistence).

    Parameters
    ----------
    reference_id : str
        Identifier of the object this metadata belongs to.
    metadata_type : str
        A semantic label for the metadata (for example: "net.dns", "team").
    metadata_value : Any
        The metadata value; it is cast to ``str`` for storage.
    id : str, optional
        Unique identifier for this metadata entry. Defaults to a random UUID.
    """

    reference_id: str
    metadata_type: str
    metadata_value: Any
    id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        # Always store a string representation of the value to simplify DB storage.
        self.metadata_value = str(self.metadata_value)
