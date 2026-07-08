from typing import override

import polars as pl

from metrion.entity.physical_entity import PhysicalEntity


class GPU(PhysicalEntity):
    """Represents a GPU physical entity with power and compute capability info.

    Parameters
    ----------
    manufacturer : str
        GPU manufacturer (e.g., 'NVIDIA').
    model : str
        GPU model name (e.g., 'NVIDIA A100-SXM4-40GB').
    serial_number : str
        GPU serial number.
    uuid : str
        Unique GPU identifier from NVML.
    pci_id : str
        PCI bus ID.
    idle_power : float
        Measured idle power consumption in watts.
    max_power : float
        Maximum power cap in watts.
    compute_capability_major : int
        CUDA compute capability major version (e.g., 8 for Ampere).
    compute_capability_minor : int
        CUDA compute capability minor version (e.g., 0 for GA100).
    parent_id : str | None
        Parent entity ID (typically the host).
    """

    def __init__(
        self,
        manufacturer: str,
        model: str,
        serial_number: str,
        uuid: str,
        pci_id: str,
        idle_power: float,
        max_power: float,
        compute_capability_major: int = 0,
        compute_capability_minor: int = 0,
        parent_id: str | None = None,
    ):
        super().__init__(id=uuid, parent_id=parent_id)
        self._manufacturer = manufacturer
        self._model = model
        self._serial_number = serial_number
        self._uuid = uuid
        self._pci_id = pci_id
        self._idle_power = idle_power
        self._max_power = max_power
        self._compute_capability_major = compute_capability_major
        self._compute_capability_minor = compute_capability_minor

    def _to_df(self):
        schema = {
            "id": pl.String,
            "manufacturer": pl.String,
            "model": pl.String,
            "serial_number": pl.String,
            "uuid": pl.String,
            "pci_id": pl.String,
            "idle_power": pl.Float64,
            "max_power": pl.Float64,
            "compute_capability_major": pl.Int32,
            "compute_capability_minor": pl.Int32,
            "parent_id": pl.String,
        }
        data = {
            "id": [self.id],
            "manufacturer": [self.manufacturer],
            "model": [self.model],
            "serial_number": [self.serial_number],
            "uuid": [self.uuid],
            "pci_id": [self.pci_id],
            "idle_power": [self.idle_power],
            "max_power": [self._max_power],
            "compute_capability_major": [self._compute_capability_major],
            "compute_capability_minor": [self._compute_capability_minor],
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
    def serial_number(self) -> str:
        return self._serial_number

    @serial_number.setter
    def serial_number(self, value: str):
        self._serial_number = value

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, value: str):
        self._uuid = value

    @property
    def pci_id(self) -> str:
        return self._pci_id

    @pci_id.setter
    def pci_id(self, value: str):
        self._pci_id = value

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    @property
    def idle_power(self) -> float:
        """
        Returns the idle power consumption of the physical entity in watts.
        If the idle power is not set, it returns 0.0.
        """
        return self._idle_power if self._idle_power is not None else 0.0

    @idle_power.setter
    def idle_power(self, value: float):
        """
        Sets the idle power consumption of the physical entity in watts.
        """
        if value < 0:
            raise ValueError("Idle power consumption cannot be negative.")
        self._idle_power = value

    @property
    def max_power(self) -> float:
        """Returns the maximum power consumption of the physical entity in watts."""
        return self._max_power

    @max_power.setter
    def max_power(self, value: float):
        """Sets the maximum power consumption of the physical entity in watts."""
        self._max_power = value

    @property
    def compute_capability_major(self) -> int:
        """CUDA compute capability major version."""
        return self._compute_capability_major

    @property
    def compute_capability_minor(self) -> int:
        """CUDA compute capability minor version."""
        return self._compute_capability_minor

    @property
    def is_ampere_or_newer(self) -> bool:
        """True if GPU is Ampere (CC 8.x) or newer."""
        return self._compute_capability_major >= 8

    @property
    def has_averaged_power_api(self) -> bool:
        """True if nvmlDeviceGetPowerUsage returns 1s averaged power.

        Ampere+ GPUs (except GA100 at CC 8.0) return averaged power.
        GA100 and older GPUs return instantaneous power readings.
        """
        if self._compute_capability_major > 8:
            return True
        if self._compute_capability_major == 8 and self._compute_capability_minor > 0:
            return True
        return False

    @override
    def _generate_id(self) -> str:
        return self._id

    @override
    def __str__(self):
        return (
            f"{self.__class__.__name__}(id={self.id}, manufacturer={self.manufacturer}, model={self.model},"
            + f"serial_number={self.serial_number}, uuid={self.uuid}, pci_id={self.pci_id})"
        )
