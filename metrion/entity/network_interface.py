import socket
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import override

import polars as pl
from loguru import logger

from metrion.entity.host import Host
from metrion.entity.physical_entity import PhysicalEntity


class NetworkInterfaceKind(Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"


class AddressFamily(Enum):
    IPV4 = "ipv4"
    IPV6 = "ipv6"


@dataclass
class NetworkAddress:
    family: AddressFamily
    address: str
    prefix_length: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()), init=False)


class NetworkInterface(PhysicalEntity):
    """
    Represents a Network Interface Card (NIC) on a host.
    """

    def __init__(
        self,
        name: str,
        kind: NetworkInterfaceKind,
        mac_address: str | None,
        addresses: list[NetworkAddress],
        parent_id: str,
    ):
        self._name = name
        self._kind = kind
        self._mac_address = mac_address
        self._addresses = addresses
        self._parent_id = parent_id
        super().__init__(id=self._generate_stable_id(), parent_id=parent_id)

    def _generate_stable_id(self) -> str:
        import hashlib

        if self._mac_address:
            return hashlib.md5(self._mac_address.encode()).hexdigest()
        else:
            # Fallback to name and parent_id if MAC is not available (e.g. virtual interfaces)
            return hashlib.md5(f"{self._name}+{self._parent_id}".encode()).hexdigest()

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> NetworkInterfaceKind:
        return self._kind

    @property
    def mac_address(self) -> str | None:
        return self._mac_address

    @property
    def addresses(self) -> list[NetworkAddress]:
        return self._addresses

    @override
    def _to_df(self) -> pl.DataFrame:
        # Convert addresses to string representation
        addr_str = ",".join([f"{a.address}/{a.prefix_length}" for a in self.addresses])

        data = {
            "id": [self.id],
            "name": [self.name],
            "kind": [self.kind.value],
            "mac_address": [self.mac_address],
            "addresses": [addr_str],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data)

    @override
    def __str__(self) -> str:
        return (
            f"NetworkInterface(id={self.id[:12]}..., name={self.name}, kind={self.kind.value}, mac={self.mac_address})"
        )


class NetworkInterfaceRegistry:
    """
    Registry to manage NetworkInterface entities and provide lookups.
    This is a Singleton, so you can instantiate it anywhere and get the same instance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._interfaces = {}
            cls._address_index = {}
            cls._name_index = {}
            cls._ifindex_index = {}
            cls._unknown_interface_id = None
            cls._instance._register_unknown_interface()
        return cls._instance

    def _register_unknown_interface(self):
        host_id = Host.get_host_id_from_env() or "unknown_host"
        unknown_interface = NetworkInterface(
            name="unknown_nic",
            kind=NetworkInterfaceKind.VIRTUAL,
            mac_address=None,
            addresses=[],
            parent_id=host_id,
        )
        self.add(unknown_interface)
        self._ifindex_index[0] = unknown_interface.id
        self._unknown_interface_id = unknown_interface.id

    def add(self, interface: NetworkInterface):
        self._interfaces[interface.id] = interface
        self._name_index[interface.name] = interface.id

        try:
            ifindex = socket.if_nametoindex(interface.name)
            self._ifindex_index[ifindex] = interface.id
        except OSError:
            pass

        for addr in interface.addresses:
            self._address_index[addr.address] = interface.id

    def get_by_address(self, address: str) -> NetworkInterface | None:
        nic_id = self._address_index.get(address)
        if nic_id:
            return self._interfaces.get(nic_id)
        return None

    def get_unknown_interface(self) -> NetworkInterface:
        return self._interfaces[self._unknown_interface_id]

    def get_id_by_address(self, address: str) -> str | None:
        id = self._address_index.get(address, None)
        if id is None:
            logger.debug(f"Network interface not found for address {address}")
        return id

    def get_id_by_name(self, name: str) -> str | None:
        id = self._name_index.get(name, None)
        if id is None:
            logger.debug(f"Network interface not found for name {name}")
        return id

    def get_id_by_ifindex(self, ifindex: int) -> str | None:
        id = self._ifindex_index.get(ifindex, None)
        if id is None:
            logger.trace(f"Network interface not found for ifindex {ifindex} using unknown interface ID")
            id = self._unknown_interface_id
        return id
