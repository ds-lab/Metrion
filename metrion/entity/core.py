import hashlib
from typing import override

import polars as pl

from metrion.entity.physical_entity import PhysicalEntity


class Core(PhysicalEntity):
    def __init__(
        self,
        core_id: int,
        package_id: int,
        thread_siblings: list[int],
        mhz: int,
        parent_id: str | None = None,
    ):
        self._core_id = core_id
        self._package_id = package_id
        self._thread_siblings = thread_siblings
        self._mhz = mhz
        super().__init__(id=None, parent_id=parent_id)

    @property
    def core_id(self) -> int:
        return self._core_id

    @core_id.setter
    def core_id(self, value: int):
        self._core_id = value

    @property
    def package_id(self) -> int:
        return self._package_id

    @package_id.setter
    def package_id(self, value: int):
        self._package_id = value

    @property
    def thread_siblings(self) -> list[int]:
        return self._thread_siblings

    @thread_siblings.setter
    def thread_siblings(self, value: list[int]):
        self._thread_siblings = value

    @property
    def mhz(self) -> int:
        return self._mhz

    @mhz.setter
    def mhz(self, value: int):
        self._mhz = value

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    def _to_df(self) -> pl.DataFrame:
        # Represent thread_siblings as a comma-separated string
        thread_siblings_str = str(self.thread_siblings).strip("[]").replace(" ", "")
        schema = {
            "id": pl.String,
            "core_id": pl.Int64,
            "package_id": pl.Int64,
            "thread_siblings": pl.String,
            "mhz": pl.Float64,
            "parent_id": pl.String,
        }
        data = {
            "id": [self.id],
            "core_id": [self.core_id],
            "package_id": [self.package_id],
            "thread_siblings": [thread_siblings_str],
            "mhz": [self.mhz],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data, schema=schema)

    @override
    def _generate_id(self) -> str:
        parent_id = self.parent_id if self.parent_id is not None else "no_parent"
        unique_string = f"{self.core_id}{self.package_id}{self.thread_siblings}{parent_id}"
        hashed_id = hashlib.md5(unique_string.encode()).hexdigest()
        return hashed_id

    @override
    def __str__(self):
        return (
            self.__class__.__name__
            + f"(id={self.id}, core_id={self.core_id}, package_id={self.package_id},"
            + f"thread_siblings={self.thread_siblings},"
            + f"mhz={self.mhz})"
        )
