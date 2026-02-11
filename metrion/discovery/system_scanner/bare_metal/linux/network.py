# %%
import os
import re
import socket

import psutil


def calculate_prefix_length(netmask, family):
    """
    Convert a netmask to prefix length for IPv4/IPv6.

    Returns:
        int | None
    """
    if not netmask:
        return None

    # Normalize in case something has leading/trailing spaces
    netmask = netmask.strip()

    if family == socket.AF_INET:
        try:
            # Example: "255.255.255.0"
            octets = netmask.split(".")
            if len(octets) != 4:
                return None

            # Count 1-bits over all octets
            return sum(bin(int(o)).count("1") for o in octets)
        except (ValueError, TypeError):
            return None

    elif family == socket.AF_INET6:
        try:
            # psutil often already returns prefix length as a string for IPv6 (e.g. "64")
            if netmask.isdigit():
                return int(netmask)

            # Otherwise assume it's an IPv6 mask, e.g. "ffff:ffff:ffff:ffff::"
            netmask_bytes = socket.inet_pton(socket.AF_INET6, netmask)
            return bin(int.from_bytes(netmask_bytes, "big")).count("1")
        except (OSError, ValueError, TypeError):
            return None

    return None


def get_info(config: dict) -> list[dict]:
    """
    Retrieves network interface information, including MAC addresses and network statistics.

    Returns
    -------
    list[dict]
        A list of dictionaries where each dictionary represents a network interface and contains:
        - name (str): The name of the interface.
        - mac_address (str): The MAC address of the interface.
        - addresses (list of dict): A list of assigned IP addresses with details.
        - is_up (bool): Whether the interface is active (if available in stats).
        - speed (int): The speed of the interface in Mbps (if available in stats).
        - type (str): The type of interface ('physical' or 'virtual').
        - pci_address (str, optional): The PCI address of the device (physical only).
        - vendor_id (str, optional): The vendor ID of the device (physical only).
        - device_id (str, optional): The device ID of the device (physical only).
    """
    net_info = []

    for interface, addrs in psutil.net_if_addrs().items():
        mac_address = "unknown"
        mac_path = f"/sys/class/net/{interface}/address"

        if os.path.exists(mac_path):
            with open(mac_path) as f:
                mac_address = f.read().strip()

        device_info = {"type": "virtual"}
        device_symlink = f"/sys/class/net/{interface}/device"
        if os.path.exists(device_symlink):
            try:
                device_path = os.path.realpath(device_symlink)
                # Check if it looks like a PCI device path
                if "/pci" in device_path or re.search(r"/[0-9a-f]{4}:[0-9a-f]{2}:", device_path):
                    device_info["type"] = "physical"
                    device_info["pci_address"] = os.path.basename(device_path)

                    vendor_path = os.path.join(device_symlink, "vendor")
                    if os.path.exists(vendor_path):
                        with open(vendor_path) as f:
                            device_info["vendor_id"] = f.read().strip()

                    device_id_path = os.path.join(device_symlink, "device")
                    if os.path.exists(device_id_path):
                        with open(device_id_path) as f:
                            device_info["device_id"] = f.read().strip()
            except OSError:
                pass

        addresses = []
        for addr in addrs:
            prefix_len = calculate_prefix_length(addr.netmask, addr.family)
            addresses.append(
                {
                    "address": addr.address,
                    "family": "ipv4" if addr.family == socket.AF_INET else "ipv6",
                    "prefix_length": prefix_len,
                    "netmask": addr.netmask,
                }
            )

        net_info.append(
            {
                "name": interface,
                "mac_address": mac_address,
                "addresses": addresses,
                **device_info,
            }
        )

    net_stats = psutil.net_if_stats()
    for nic in net_info:
        interface = nic["name"]
        if interface in net_stats:
            stats = net_stats[interface]
            nic.update(
                {
                    "is_up": stats.isup,
                    "speed": stats.speed,
                }
            )

    return net_info


if __name__ == "__main__":
    print(get_info({}))
