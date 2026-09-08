from typing import override

import polars as pl

from metrion.entity.cpu import CPU
from metrion.entity.entity import Entity


class DRAM(Entity):
    def __init__(self, total_memory: float, cpu: CPU, idle_power: float, max_power: float, parent_id: str | None):
        super().__init__(parent_id=parent_id)
        self._cpu = cpu
        self._total_memory = total_memory
        self._idle_power = idle_power
        self._max_power = max_power

    def _to_df(self):
        schema = {
            "id": pl.String,
            "package_id": pl.String,
            "cpu_id": pl.String,
            "idle_power": pl.Float64,
            "max_power": pl.Float64,
            "total_memory": pl.Float64,
            "parent_id": pl.String,
        }
        data = {
            "id": [self.id],
            "package_id": [self.cpu.package_id],
            "cpu_id": [self.cpu.id],
            "idle_power": [self.idle_power],
            "max_power": [self._max_power],
            "total_memory": [self.total_memory],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data, schema=schema)

    @property
    def cpu(self) -> CPU:
        return self._cpu

    @cpu.setter
    def cpu(self, value: CPU):
        self._cpu = value

    @property
    def total_memory(self) -> float:
        return self._total_memory

    @total_memory.setter
    def total_memory(self, value: float):
        self._total_memory = value

    @property
    def idle_power(self) -> float:
        return self._idle_power

    @idle_power.setter
    def idle_power(self, value: float):
        self._idle_power = value

    @property
    def max_power(self) -> float:
        return self._max_power

    @max_power.setter
    def max_power(self, value: float):
        self._max_power = value

    @override
    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, cpu={self.cpu}, total_memory={self.total_memory})"
