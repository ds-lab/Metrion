import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import override

import polars as pl

from metrion.entity.entity import Entity


class NetworkTransport(StrEnum):
    """
    Network transport protocol (OSI Layer 4).

    Based on OpenTelemetry semantic conventions for network.transport.
    """

    TCP = "tcp"
    UDP = "udp"


@dataclass(frozen=True)
class NetworkAddress:
    """
    A network-level address (IP, Port, Protocol).

    This class represents a network endpoint, which can be either:
    - A **Remote Endpoint**: The destination of an outgoing connection.
    - A **Local Endpoint**: The listening address of a local server.

    The combination of (ip_address, port, transport) uniquely identifies
    the network socket/endpoint.

    Parameters
    ----------
    ip_address : str
        The IP address (IPv4 or IPv6).
    port : int
        The port number.
    transport : NetworkTransport | str
        The transport protocol (tcp, udp, etc.).

    Examples
    --------
    >>> # TCP connection to a database (Remote Endpoint)
    >>> addr = NetworkAddress("10.0.0.5", 5432, NetworkTransport.TCP)
    >>> addr.to_key()
    'tcp://10.0.0.5:5432'

    >>> # Local Web Server (Local Endpoint)
    >>> addr = NetworkAddress("0.0.0.0", 80, NetworkTransport.TCP)
    """

    ip_address: str
    port: int
    transport: NetworkTransport | str = NetworkTransport.TCP

    def to_key(self) -> str:
        """
        Generate a stable, canonical key for this network address.

        Returns
        -------
        str
            A canonical key in format: "{transport}://{ip_address}:{port}"
        """
        transport_str = self.transport.value if isinstance(self.transport, NetworkTransport) else self.transport
        return f"{transport_str}://{self.ip_address}:{self.port}"

    def to_hash(self) -> str:
        """
        Generate a stable hash identifier for this network address.

        Returns
        -------
        str
            SHA-256 hash of the canonical key.
        """
        return hashlib.sha256(self.to_key().encode("utf-8")).hexdigest()


class CommunicationEndpoint(Entity):
    """
    Represents a Network Endpoint (Socket/Address) involved in communication.

    This entity represents a specific network address (IP:Port) and its context.
    It acts as a node in the communication graph.

    It can represent:
    1. **Local Endpoint**: A socket on the monitored host.
       - Has a `parent_id` (the process owning the socket).
       - Has a `nic_id` (the physical interface it is bound to).
       - Example: `192.168.1.5:54321` (Client Socket) or `0.0.0.0:80` (Server Socket).
    2. **Remote Endpoint**: An address on a remote host.
       - Typically has no `parent_id` (unless we monitor that host too).
       - Example: `8.8.8.8:443` (Google DNS).

    Attributes
    ----------
    network_address : NetworkAddress
        The address (IP, Port, Protocol) of **THIS** endpoint.
    """

    def __init__(
        self,
        network_address: NetworkAddress,
    ):
        self._network_address = network_address
        endpoint_id = self._generate_id()
        super().__init__(id=endpoint_id, parent_id=None)

    @property
    def network_address(self) -> NetworkAddress:
        """The address (IP:Port) of this endpoint."""
        return self._network_address

    def _to_df(self) -> pl.DataFrame:
        """
        Convert the communication endpoint to a Polars DataFrame.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - id
            - ip_address
            - port
            - transport
            - endpoint_type
            - parent_id
            - nic_id
        """
        transport = self._network_address.transport
        transport_str = transport.value if isinstance(transport, NetworkTransport) else transport
        data = {
            "id": [self.id],
            "ip_address": [self._network_address.ip_address],
            "port": [self._network_address.port],
            "transport": [transport_str],
            "parent_id": [self.parent_id],
        }
        return pl.DataFrame(data)

    @override
    def _generate_id(self) -> str:
        """
        Generate a unique ID for this endpoint.
        """
        unique_str = self._network_address.to_key()

        return hashlib.sha256(unique_str.encode("utf-8")).hexdigest()

    @override
    def __str__(self) -> str:
        owner_str = f"owner={self.parent_id[:8]}..., " if self.parent_id else "owner=None, "
        return f"CommunicationEndpoint(id={self.id[:8]}..., {owner_str}remote={self._network_address.to_key()}"
