import hashlib
from typing import override

import polars as pl

from metrion.entity.core import Core
from metrion.entity.physical_entity import PhysicalEntity


class CPU(PhysicalEntity):
    def __init__(
        self,
        manufacturer: str,
        model: str,
        physical_cores: int,
        logical_cores: int,
        package_id: int,
        cache: int,
        hyperthreading: bool,
        topology: dict[int, int],
        cores: list[Core],
        idle_power_package: float,
        max_power_package: float | None = None,
        max_energy_range_j: float | None = None,
        parent_id: str | None = None,
    ):
        self._manufacturer = manufacturer
        self._model = model
        self._physical_cores = physical_cores
        self._logical_cores = logical_cores
        self._socket_id = package_id
        self._cache = cache
        self._hyperthreading = hyperthreading
        self._topology = topology
        self._cores = cores
        self._idle_power_package = idle_power_package
        self._max_power_package = max_power_package
        self._max_energy_range_j = max_energy_range_j
        super().__init__(id=None, parent_id=parent_id)

    def _to_df(self) -> pl.DataFrame:
        core_numbers_string = ",".join(map(str, self.get_core_numbers()))
        schema = {
            "id": pl.String,
            "manufacturer": pl.String,
            "model": pl.String,
            "physical_cores": pl.Int64,
            "logical_cores": pl.Int64,
            "package_id": pl.String,
            "cache": pl.Int64,
            "hyperthreading": pl.Boolean,
            "core_ids": pl.String,
            "idle_power_package": pl.Float64,
            "max_power_package": pl.Float64,
            "max_energy_range_j": pl.Float64,
            "parent_id": pl.String,
        }
        data = {
            "id": [self.id],
            "manufacturer": [self.manufacturer],
            "model": [self.model],
            "physical_cores": [self.physical_cores],
            "logical_cores": [self.logical_cores],
            "package_id": [self.package_id],
            "cache": [self.cache],
            "hyperthreading": [self.hyperthreading],
            "core_ids": [core_numbers_string],
            "idle_power_package": [self.idle_power_package],
            "max_power_package": [self._max_power_package],
            "max_energy_range_j": [self._max_energy_range_j],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data, schema=schema)

    @property
    def manufacturer(self) -> str:
        return self._manufacturer

    @manufacturer.setter
    def manufacturer(self, value: str):
        self._manufacturer = value

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str):
        self._model = value

    @property
    def physical_cores(self) -> int:
        return self._physical_cores

    @physical_cores.setter
    def physical_cores(self, value: int):
        self._physical_cores = value

    @property
    def logical_cores(self) -> int:
        return self._logical_cores

    @logical_cores.setter
    def logical_cores(self, value: int):
        self._logical_cores = value

    @property
    def package_id(self) -> int:
        return self._socket_id

    @package_id.setter
    def package_id(self, value: int):
        self._socket_id = value

    @property
    def topology(self) -> dict[int, int]:
        return self._topology

    @topology.setter
    def topology(self, value: dict[int, int]):
        self._topology = value

    @property
    def cache(self) -> int:
        return self._cache

    @cache.setter
    def cache(self, value: int):
        self._cache = value

    @property
    def hyperthreading(self) -> bool:
        return self._hyperthreading

    @hyperthreading.setter
    def hyperthreading(self, value: bool):
        self._hyperthreading = value

    @property
    def cores(self) -> list[Core]:
        return self._cores

    @cores.setter
    def cores(self, value: list[Core]):
        self._cores = value

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    @property
    def idle_power_package(self) -> float:
        """
        Returns the idle power consumption of the package in watts.
        """
        return self._idle_power_package if self._idle_power_package is not None else 0.0

    @idle_power_package.setter
    def idle_power_package(self, value: float):
        """
        Sets the idle power consumption of the package in watts.
        """
        self._idle_power_package = value

    @property
    def max_power_package(self) -> float | None:
        """Returns the maximum power consumption of the package in watts."""
        return self._max_power_package

    @max_power_package.setter
    def max_power_package(self, value: float | None):
        """Sets the maximum power consumption of the package in watts."""
        self._max_power_package = value

    @property
    def max_energy_range_j(self) -> float | None:
        """Returns the value the RAPL package energy counter wraps at, in joules."""
        return self._max_energy_range_j

    def get_core_by_core_number(self, core_id: int) -> Core | None:
        """Returns the Core object with the specified core_id."""
        for core in self.cores:
            if core.core_id == core_id:
                return core
        return None

    def get_core_numbers(self) -> list[int]:
        """Returns a list of core IDs for this CPU."""
        return [core.core_id for core in self.cores]

    @override
    def _generate_id(self) -> str:
        assert self.parent_id is not None, "Parent ID must be set before generating ID"
        unique_string = (
            f"{self.manufacturer}{self.model}{self.physical_cores}{self.logical_cores}{self.package_id}{self.cache}"
            f"{self.hyperthreading}{self.parent_id}"
        )
        hashed_id = hashlib.md5(unique_string.encode()).hexdigest()
        return hashed_id

    @override
    def __str__(self):
        return (
            self.__class__.__name__
            + f"(id={self.id}, manufacturer={self.manufacturer}, model={self.model},"
            + f"physical_cores={self.physical_cores}, logical_cores={self.logical_cores},package_id={self.package_id}"
            + f",cache={self.cache},hyperthreading={self.hyperthreading}"
            + f"cores_count={len(self.cores)})"
        )
