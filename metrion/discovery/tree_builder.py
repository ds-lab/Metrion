import importlib
import json
import os
import sys

from loguru import logger

from metrion.discovery.system_info import detect_environment, detect_os


class HardwareTreeBuilder:
    """
    A class to build and manage a hardware tree structure based on system environment and operating system.
    """

    def __init__(self, update_interval=60):
        """
        Initializes the hardware scanner and builds the tree.

        Parameters
        ----------
        update_interval : int, optional
            The interval in seconds for updating the tree (default is 60 seconds).
        """
        self.tree = {}
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure module directory is loaded
        # self.build_tree()

    def import_hardware_module(self, env_type, os_type, component, params=None):
        """
        Attempts to load a specific hardware module dynamically.

        Parameters
        ----------
        env_type : str
            The environment type (e.g., 'bare_metal', 'vm', 'container').
        os_type : str
            The operating system type (e.g., 'linux', 'windows', 'darwin').
        component : str
            The hardware component to be loaded (e.g., 'cpu', 'gpu').
        params : dict, optional
            Additional parameters to pass to the module's get_info function.

        Returns
        -------
        dict or str
            Returns the hardware information as a dictionary if successful, otherwise a string indicating failure.
        """
        # module_path = f"metrion.hardware_detection.hardware_modules.{env_type}.{os_type}.{component}"
        module_path = f"metrion.discovery.system_scanner.{env_type}.{os_type}.{component}"
        logger.debug(f"Loading module: {module_path}")
        try:
            module = importlib.import_module(module_path)
            if params:
                return module.get_info(params)
            else:
                return module.get_info({})
        except ModuleNotFoundError as e:
            logger.debug(f"Module not found: {module_path}: {e}")
            return "Not available"
        except Exception as e:
            logger.opt(exception=True).warning(f"Error loading {module_path}: {e}")
            return "Error"

    def build_tree(self, components=None):
        """
        Constructs the hardware tree based on the detected environment and operating system.
        Components = [
            {'NAME': 'OS'},
            {'NAME': 'CPU', 'IDLE_POWER': 'auto'},
            {'NAME': 'GPU', 'IDLE_POWER': [{'ID': 0, 'IDLE_POWER': 20.0}]}
        ]
        """
        if components is None:
            components = [
                {"NAME": "OS"},
                {"NAME": "CPU"},
                {"NAME": "GPU"},
                {"NAME": "DRAM"},
                {"NAME": "STORAGE"},
                {"NAME": "NETWORK"},
            ]

        logger.debug("Starting to construct hardware tree for: " + ", ".join(c["NAME"] for c in components))

        env_type = detect_environment()
        os_type = detect_os()
        hostname = os.uname().nodename
        self.tree = {hostname: {"env_type": env_type, "os_type": os_type}}

        # self.tree[hostname].update(self.import_hardware_module(env_type, os_type, "host"))
        # self.tree[hostname].update(self.import_hardware_module(env_type, os_type, "os"))

        for component in components:
            component_name = component["NAME"].lower()
            params = {k: v for k, v in component.items() if k != "NAME"}
            self.tree[hostname][component_name] = self.import_hardware_module(env_type, os_type, component_name, params)
        logger.debug("Finished to construct hardware tree!")

    def get_tree(self):
        """
        Returns the current hardware tree.

        Returns
        -------
        dict
            The hardware tree structure.
        """
        return self.tree

    def get_subtree(self, path):
        """
        Retrieves a specific subtree from the hardware tree.

        Parameters
        ----------
        path : list
            A list representing the path to the desired subtree (e.g., ['bare_metal', 'linux', 'cpu']).

        Returns
        -------
        dict or str
            The requested subtree or an error message if the path is invalid.
        """
        subtree = self.tree
        try:
            for key in path:
                subtree = subtree[key]
            return subtree
        except KeyError:
            return "Invalid path"

    def export_tree(self, filename="hardware_tree.json"):
        """
        Saves the hardware tree as a JSON file in a dedicated export directory.

        Parameters
        ----------
        filename : str, optional
            The name of the JSON file to save (default is 'hardware_tree.json').
        """
        export_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "export")
        os.makedirs(export_dir, exist_ok=True)
        file_path = os.path.join(export_dir, filename)

        with open(file_path, "w") as f:
            json.dump(self.tree, f, indent=4)
        print(f"Hardware tree exported to {file_path}")
