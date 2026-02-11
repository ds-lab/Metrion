import os
from enum import Enum
from typing import override

import polars as pl

from metrion.entity.physical_entity import PhysicalEntity

ENV_VAR_HOST_ID = "METRION_HOST_ID"


class HostType(Enum):
    SERVER = "server"
    VM = "vm"
    CONTAINER = "container"


class Host(PhysicalEntity):
    def __init__(self, name: str, host_type: HostType, os_name: str, parent_id: str | None = None):
        super().__init__(id=None, parent_id=parent_id)
        self._name = name
        self._host_type = host_type
        self._os: str = os_name
        # Set the environment variable when the host is created
        host_uuid = os.environ.get(ENV_VAR_HOST_ID, None)
        if host_uuid is None:
            host_uuid = str(self._id)
            os.environ[ENV_VAR_HOST_ID] = self._id
        self._id = host_uuid

    def _to_df(self):
        schema = {
            "id": pl.String,
            "name": pl.String,
            "host_type": pl.String,
            "os": pl.String,
            "parent_id": pl.String,
        }
        data = {
            "id": [self.id],
            "name": [self.name],
            "host_type": [self.host_type.value],
            "os": [self.os],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data, schema=schema)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def host_type(self) -> HostType:
        return self._host_type

    @host_type.setter
    def host_type(self, value: HostType):
        self._host_type = value

    @property
    def os(self) -> str:
        return self._os

    @os.setter
    def os(self, value: str):
        self._os = value

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value
        # Update the environment variable when the ID changes
        os.environ[ENV_VAR_HOST_ID] = value

    @staticmethod
    def get_host_id_from_env() -> str | None:
        """
        Retrieve the host ID from the environment variable.

        Returns
        -------
        str | None
            The host ID if set, otherwise None.
        """
        return os.environ.get(ENV_VAR_HOST_ID)

    @override
    def __str__(self):
        return f"{self.__class__.__name__}(id={self.id}, name={self.name}, type={self.host_type}, os={self.os})"
