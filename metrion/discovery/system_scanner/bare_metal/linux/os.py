import os

socket = __import__("socket")


def get_info(config: dict) -> dict:
    """
    Retrieves system information, including OS details and hostname.

    Returns
    -------
    dict
        A dictionary containing:
        - uname (dict): System information with the following keys:
            - sysname (str): Name of the operating system.
            - hostname (str): Network hostname of the system.
            - release (str): OS release version.
            - version (str): Detailed OS version information.
            - machine (str): Hardware architecture.
        - os_release (dict or None): OS release details from "/etc/os-release", if available.
    """
    info = {
        "general": {
            "sysname": os.uname().sysname,
            "hostname": os.uname().nodename,
            "release": os.uname().release,
            "version": os.uname().version,
            "machine": os.uname().machine,
        },
        "detailed": {},
    }

    try:
        with open("/etc/os-release") as f:
            for line in f:
                key, value = line.strip().split("=", 1)
                info["detailed"][key] = value.strip('"')
    except FileNotFoundError:
        info["detailed"] = {}

    return info
