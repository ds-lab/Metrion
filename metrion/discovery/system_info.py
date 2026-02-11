import os
import platform


def detect_environment():
    """
    Detects the system environment (Bare Metal, Virtual Machine, or Container).

    Returns
    -------
    str
        One of the following values:
        - "container": If the system is running inside a Docker or containerized environment.
        - "vm": If the system is running inside a virtual machine (VMware, VirtualBox, QEMU, KVM, Hyper-V).
        - "bare_metal": If the system is running directly on hardware.
    """
    # Detect container environment via /.dockerenv or /proc/self/cgroup
    if os.path.exists("/.dockerenv"):
        return "container"

    if os.path.exists("/proc/self/cgroup"):
        try:
            with open("/proc/self/cgroup", errors="ignore") as f:
                if any("docker" in line or "containerd" in line for line in f):
                    return "container"
        except Exception as e:
            print(f"Error reading /proc/self/cgroup: {e}")

    # Detect virtual machine via DMI/SMBIOS (Linux only)
    if os.path.exists("/sys/class/dmi/id/product_name"):
        try:
            with open("/sys/class/dmi/id/product_name") as f:
                product_name = f.read().strip().lower()
                if any(keyword in product_name for keyword in ["vmware", "virtualbox", "qemu", "kvm", "hyper-v"]):
                    return "vm"
        except Exception as e:
            print(f"⚠ Error reading /sys/class/dmi/id/product_name: {e}")

    return "bare_metal"


def detect_os():
    """
    Determines the operating system type.

    Returns
    -------
    str
        One of the following values:
        - "windows": If the OS is Windows.
        - "linux": If the OS is Linux.
        - "darwin": If the OS is macOS.
        - "unknown": If the OS could not be determined.
    """
    os_map = {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}
    return os_map.get(platform.system(), "unknown")
