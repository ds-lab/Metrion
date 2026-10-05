from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from metrion.discovery.tree_builder import HardwareTreeBuilder
from metrion.entity.core import Core
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.gpu import GPU
from metrion.entity.host import Host, HostType
from metrion.entity.network_interface import (
    AddressFamily,
    NetworkAddress,
    NetworkInterface,
    NetworkInterfaceKind,
    NetworkInterfaceRegistry,
)
from metrion.entity.physical_entity import PhysicalEntity

if TYPE_CHECKING:
    from metrion.config_reader.config_schema import AgentConfig


class HardwareDetection:
    """
    Hardware detection class responsible for discovering physical entities on the host system.

    This class provides functionality to detect and create entity objects for hardware components
    such as CPUs, GPUs, RAM, and the host system itself.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the HardwareDetection with configuration.

        Parameters
        ----------
        config : AgentConfig
            Agent configuration object containing energy model settings.
        """
        self._config = config
        self._devices: dict[str, Any] = {}
        self._entities: list[PhysicalEntity] = []

    def _group_entities_by_type(self) -> dict[type[PhysicalEntity], list[PhysicalEntity]]:
        """
        Group detected entities by their type.

        Returns
        -------
        dict[type[PhysicalEntity], list[PhysicalEntity]]
            Dictionary where keys are entity types and values are lists of entity instances.
        """
        grouped_entities: dict[type[PhysicalEntity], list[PhysicalEntity]] = {}
        for entity in self._entities:
            entity_type = type(entity)
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append(entity)

        return grouped_entities

    def get_physical_entities(self) -> dict[type[PhysicalEntity], list[PhysicalEntity]]:
        """
        Get all physical entities detected on the host system.

        Returns
        -------
        dict[type[PhysicalEntity], list[PhysicalEntity]]
            Dictionary where keys are entity types and values are lists of entity instances.
        """
        self._discover_hardware_devices()
        self._create_entities_from_devices()
        return self._group_entities_by_type()

    def _discover_hardware_devices(self) -> None:
        """
        Discover hardware devices per host system.

        This method builds a hardware tree dynamically using the HardwareTreeBuilder.
        Components are filtered based on the energy_model configuration.
        """
        logger.debug("Building hardware tree dynamically")
        builder = HardwareTreeBuilder()

        # Build list of components to detect based on config
        components_to_detect = self._get_components_to_detect()
        builder.build_tree(components_to_detect)

        tree = builder.get_tree()
        self._devices = self._flatten_dict_layer(tree)

    def _get_components_to_detect(self) -> list[dict[str, Any]]:
        """
        Get the list of components to detect based on energy model configuration.

        Returns
        -------
        list[dict[str, Any]]
            List of component configurations for hardware detection.
        """
        components = []

        # Always include OS for host information
        components.append({"NAME": "OS"})

        # Map component names to their tree builder format
        component_mapping = {
            "cpu": "CPU",
            "gpu": "GPU",
            "dram": "DRAM",
            "network": "NETWORK",
        }

        for component_name, tree_name in component_mapping.items():
            if self._config.energy_model.is_component_included(component_name):
                component_config: dict[str, Any] = {
                    "NAME": tree_name,
                    "calibration": Path(self._config.calibration).absolute(),
                }

                # Add idle/max power settings from energy config if specified
                comp_config = self._config.energy_model.get_component_config(component_name)
                if comp_config and comp_config.power != "auto":
                    idle_powers = []
                    for subcomp in comp_config.power:
                        idle_powers.append(
                            {
                                "ID": subcomp.id,
                                "IDLE_POWER": subcomp.idle_power,
                                "MAX_POWER": subcomp.max_power,
                            }
                        )
                    if idle_powers:
                        component_config["power"] = idle_powers

                components.append(component_config)
                logger.debug(f"Including component: {tree_name}")
            else:
                logger.debug(f"Excluding component: {tree_name}")

        return components

    def _flatten_dict_layer(self, nested_dict: dict) -> dict[str, Any]:
        """
        Flatten a nested dictionary by one layer.

        Parameters
        ----------
        nested_dict : dict
            Nested dictionary to flatten.

        Returns
        -------
        dict of str to Any
            Flattened dictionary.
        """
        result = {}
        for key, subdict in nested_dict.items():
            if isinstance(subdict, dict):
                # Add the key to the subdict and merge
                result.update({**subdict, "hostname": key})
            else:
                result[key] = subdict
        return result

    def _parse_device_tree(self, node: Any, path: str = "") -> dict[str, Any]:
        """
        Parse device tree recursively to extract relevant hardware information.

        Parameters
        ----------
        node : Any
            Current node in the device tree.
        path : str, optional
            Current path in the tree structure.

        Returns
        -------
        dict of str to Any
            Parsed device information dictionary.
        """
        result = {}

        if isinstance(node, dict):
            for key, value in node.items():
                new_path = f"{path}.{key}"
                result.update(self._parse_device_tree(value, new_path))

        elif isinstance(node, list):
            for index, item in enumerate(node):
                category = path.split(".")[-1]
                if self._is_device_category_relevant(category):
                    result.update({category: node})

                new_path = f"{path}[{index}]"
                result.update(self._parse_device_tree(item, new_path))
        else:
            # Handle leaf nodes
            if "os.general.hostname" in path:
                result.update({"hostname": node})
            elif "os.detailed.PRETTY_NAME" in path:
                result.update({"os": node})
            elif "env_type" in path:
                if node == "bare_metal":
                    result.update({"type": "server"})

        return result

    def _is_device_category_relevant(self, category: str) -> bool:
        """
        Check if a device category is relevant for detection.

        Parameters
        ----------
        category : str
            Device category to check.

        Returns
        -------
        bool
            True if category should be included, False otherwise.
        """
        # Map category names to config component names
        category_to_component = {
            "cpu": "cpu",
            "gpu": "gpu",
            "dram": "dram",
            "network": "network",
            "host": "host",
            "os": "host",
        }

        component_name = category_to_component.get(category.lower(), category.lower())
        return self._config.energy_model.is_component_included(component_name)

    def _create_entities_from_devices(self) -> None:
        """
        Create entity objects from discovered hardware devices.

        This method transforms the raw device information into proper entity objects
        that can be used throughout the system.
        """
        entities = []

        # Check if we have basic host information
        if not self._has_valid_host_info():
            logger.warning("Insufficient host information to create entities")
            return

        # Create host entity
        host_entity = self._create_host_entity()
        entities.append(host_entity)

        # Create CPU entities
        if self._config.check_if_component_included("cpu"):
            cpu_entities = self._create_cpu_entities(host_entity)
            entities.extend(cpu_entities)

        # Create GPU entities
        if self._config.check_if_component_included("gpu"):
            entities.extend(self._create_gpu_entities(host_entity))

        # Create DRAM entities
        if self._config.check_if_component_included("dram") and self._config.check_if_component_included("cpu"):
            entities.extend(self._create_dram_entities(host_entity, cpu_entities))  # type: ignore

        # Create Network Interface entities
        if self._config.check_if_component_included("network"):
            entities.extend(self._create_nic_entities(host_entity))

        self._entities.extend(entities)

    def _has_valid_host_info(self) -> bool:
        """
        Check if we have sufficient information to create a host entity.

        Returns
        -------
        bool
            True if host information is valid, False otherwise.
        """
        try:
            return "hostname" in self._devices["os"]["general"] and "os" in self._devices
        except (KeyError, TypeError):
            return False

    def _create_host_entity(self) -> Host:
        """
        Create a Host entity from device information.

        Returns
        -------
        Host
            Host entity object.
        """
        name = self._devices["os"]["general"]["hostname"]
        os_name = self._devices["os"]["detailed"]["PRETTY_NAME"]
        host_type = "server" if self._devices.get("env_type") == "bare_metal" else "unknown"

        return Host(name=name, os_name=os_name, host_type=HostType(host_type))

    def _create_nic_entities(self, host_entity: Host) -> list[PhysicalEntity]:
        """
        Create Network Interface entities from device information.

        Parameters
        ----------
        host_entity : Host
            Parent host entity.

        Returns
        -------
        list of PhysicalEntity
            List of Network Interface entity objects.
        """
        nic_entities = []
        nic_registry = NetworkInterfaceRegistry()
        nic_entities.append(nic_registry.get_unknown_interface())

        if "network" not in self._devices:
            return nic_entities

        for nic_data in self._devices["network"]:
            try:
                kind = (
                    NetworkInterfaceKind.PHYSICAL
                    if nic_data.get("type") == "physical"
                    else NetworkInterfaceKind.VIRTUAL
                )

                addresses = []
                for addr_info in nic_data.get("addresses", []):
                    if isinstance(addr_info, str):
                        addr_str = addr_info
                        prefix_len = None
                    else:
                        addr_str = addr_info.get("address")
                        prefix_len = addr_info.get("prefix_length")

                    if ":" in addr_str:
                        family = AddressFamily.IPV6
                        if prefix_len is None:
                            prefix_len = 128  # Default prefix length for IPv6
                    else:
                        family = AddressFamily.IPV4
                        if prefix_len is None:
                            prefix_len = 32  # Default prefix length for IPv4

                    network_address = NetworkAddress(
                        family=family,
                        address=addr_str,
                        prefix_length=prefix_len,
                    )
                    addresses.append(network_address)

                nic_entity = NetworkInterface(
                    name=nic_data.get("name", "unknown"),
                    kind=kind,
                    mac_address=nic_data.get("mac_address"),
                    addresses=addresses,
                    parent_id=host_entity.id,
                )
                nic_registry.add(nic_entity)
                nic_entities.append(nic_entity)

            except KeyError as e:
                logger.error(f"Missing NIC data field: {e}")
            except Exception as e:
                logger.error(f"Error creating Network Interface entity: {e}")

        return nic_entities

    def _create_cpu_entities(self, host_entity: Host) -> list[CPU]:
        """
        Create CPU entities from device information.

        Parameters
        ----------
        host_entity : Host
            Parent host entity.

        Returns
        -------
        list of CPU
            List of CPU entity objects.
        """
        cpu_entities = []

        if "cpu" not in self._devices:
            return cpu_entities

        for cpu_data in self._devices["cpu"]:
            try:
                package_id = cpu_data["package_id"]

                # Get power config from agent config
                power_config = self._config.get_subcomponent_power_config("cpu", package_id)

                # Use config values if not "auto", otherwise use detected values
                idle_power_package = (
                    cpu_data.get("idle_power_package", 0.0)
                    if power_config.idle_power == "auto"
                    else power_config.idle_power
                )

                max_power_package = (
                    cpu_data.get("max_power_package") if power_config.max_power == "auto" else power_config.max_power
                )

                cpu_entity = CPU(
                    parent_id=host_entity.id,
                    manufacturer=cpu_data["manufacturer"],
                    model=cpu_data["model"],
                    physical_cores=cpu_data["physical_cores"],
                    logical_cores=cpu_data["logical_cores"],
                    package_id=package_id,
                    cache=cpu_data["cache"],
                    hyperthreading=cpu_data["hyperthreading"],
                    topology=cpu_data["topology"],
                    cores=[],
                    idle_power_package=idle_power_package,
                    max_power_package=max_power_package,
                    max_energy_range_j=cpu_data.get("max_energy_range_j"),
                )

                # Create core entities for this CPU
                cores = self._create_core_entities(cpu_entity, cpu_data)
                cpu_entity.cores = cores

                cpu_entities.append(cpu_entity)

            except KeyError as e:
                logger.error(f"Missing CPU data field: {e}")
            except Exception as e:
                logger.error(f"Error creating CPU entity: {e}")

        return cpu_entities

    def _create_core_entities(self, cpu_entity: CPU, cpu_data: dict) -> list[Core]:
        """
        Create Core entities for a CPU.

        Parameters
        ----------
        cpu_entity : CPU
            Parent CPU entity.
        cpu_data : dict
            CPU data dictionary.

        Returns
        -------
        list of Core
            List of Core entity objects.
        """
        cores = []

        logical_cpus = cpu_data.get("logical_cpus", [])
        topology = cpu_data.get("topology", {})

        for logical_cpu in logical_cpus:
            try:
                core = Core(
                    core_id=logical_cpu["id"],
                    package_id=cpu_data["package_id"],
                    thread_siblings=topology.get(logical_cpu["id"], [logical_cpu["id"]]),
                    mhz=logical_cpu["frequency_mhz"],
                    parent_id=cpu_entity.id,
                )
                cores.append(core)
            except KeyError as e:
                logger.error(f"Missing core data field: {e}")
            except Exception as e:
                logger.error(f"Error creating Core entity: {e}")

        return cores

    def _create_gpu_entities(self, host_entity: Host) -> list[GPU]:
        """
        Create GPU entities from device information.

        Parameters
        ----------
        host_entity : Host
            Parent host entity.

        Returns
        -------
        list of GPU
            List of GPU entity objects.
        """
        gpu_entities = []

        if "gpu" not in self._devices:
            return gpu_entities

        for idx, gpu_data in enumerate(self._devices["gpu"]):
            try:
                # Get power config from agent config
                power_config = self._config.get_subcomponent_power_config("gpu", idx)

                # Use config value if not "auto", otherwise use detected value
                gpu_idle_power = (
                    gpu_data.get("idle_power", 0.0) if power_config.idle_power == "auto" else power_config.idle_power
                )
                gpu_max_power = (
                    gpu_data.get("max_power", 0.0) if power_config.max_power == "auto" else power_config.max_power
                )

                gpu_entity = GPU(
                    parent_id=host_entity.id,
                    manufacturer=gpu_data["manufacturer"],
                    model=gpu_data["model"],
                    serial_number=gpu_data["serial_number"],
                    uuid=gpu_data["uuid"],
                    pci_id=gpu_data["pci_id"],
                    idle_power=gpu_idle_power,
                    max_power=gpu_max_power,
                    compute_capability_major=gpu_data.get("compute_capability_major", 0),
                    compute_capability_minor=gpu_data.get("compute_capability_minor", 0),
                )
                gpu_entities.append(gpu_entity)

            except KeyError as e:
                logger.error(f"Missing GPU data field: {e}")
            except Exception as e:
                logger.error(f"Error creating GPU entity: {e}")

        return gpu_entities

    def _create_dram_entities(self, host_entity: Host, cpu_entities: list[CPU]) -> list[DRAM]:
        """
        Create RAM entities from device information.

        The scanner reports one entry per CPU socket, numbered the way the kernel numbers sockets,
        so every entry matches one of the CPU entities.

        Parameters
        ----------
        host_entity : Host
            Parent host entity.
        cpu_entities : list of CPU
            List of CPU entities already created.

        Returns
        -------
        list of DRAM
            List of DRAM entity objects, one per socket.
        """
        dram_entities = []

        if "dram" not in self._devices:
            return dram_entities

        sockets = self._devices["dram"]
        if not isinstance(sockets, list):
            logger.warning(f"DRAM detection returned no usable data ({sockets}), skipping DRAM entity creation.")
            return dram_entities

        cpu_by_package = {cpu.package_id: cpu for cpu in cpu_entities}

        for socket_data in sockets:
            package_id = socket_data.get("package_id")
            try:
                cpu_entity = cpu_by_package.get(package_id)
                if not cpu_entity:
                    logger.warning(f"No CPU found for package_id {package_id}, skipping DRAM entity creation.")
                    continue

                total_memory = socket_data["total_memory"]

                # Get power config from agent config
                power_config = self._config.get_subcomponent_power_config("dram", package_id)

                # Use config value if not "auto", otherwise use detected value
                dram_idle_power = (
                    socket_data.get("idle_power", 0.0) if power_config.idle_power == "auto" else power_config.idle_power
                )

                dram_max_power = (
                    socket_data.get("max_power", 0.0) if power_config.max_power == "auto" else power_config.max_power
                )

                if total_memory > 0:
                    dram_entity = DRAM(
                        total_memory=total_memory,
                        cpu=cpu_entity,
                        parent_id=host_entity.id,
                        idle_power=dram_idle_power,
                        max_power=dram_max_power,
                        max_energy_range_j=socket_data.get("max_energy_range_j"),
                    )
                    dram_entities.append(dram_entity)

            except KeyError as e:
                logger.error(f"Missing DRAM data field for socket {package_id}: {e}")
            except Exception as e:
                logger.error(f"Error creating DRAM entity for socket {package_id}: {e}")

        return dram_entities
