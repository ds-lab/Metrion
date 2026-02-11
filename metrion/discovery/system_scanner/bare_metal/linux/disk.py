import os

import psutil


def get_info():
    """
    Retrieves disk information, including partitions, usage statistics, and serial numbers.

    Returns
    -------
    dict
        A dictionary where keys are disk device names, and values contain disk details such as:
        - mountpoint (str): The mount location of the disk.
        - filesystem (str): The type of filesystem used.
        - total_space (int): Total disk space in bytes.
        - used_space (int): Used disk space in bytes.
        - free_space (int): Available disk space in bytes.
        - percent_used (float): Percentage of disk space used.
        - serial (str): Disk serial number (or 'Unknown' if not accessible).
        - error (str, optional): Error message if access to the disk is restricted.
    """
    disk_info = {}

    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            serial_path = f"/sys/class/block/{partition.device.split('/')[-1]}/serial"

            serial = "Unknown"
            if os.path.exists(serial_path):
                with open(serial_path) as f:
                    serial = f.read().strip()

            disk_info[partition.device] = {
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "total_space": usage.total,
                "used_space": usage.used,
                "free_space": usage.free,
                "percent_used": usage.percent,
                "serial": serial,
            }
        except PermissionError:
            disk_info[partition.device] = {"error": "Permission denied"}
        except Exception as e:
            disk_info[partition.device] = {"error": str(e)}

    return disk_info
