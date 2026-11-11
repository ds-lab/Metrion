import re
from enum import Enum
from pathlib import Path
from typing import override

from loguru import logger

from metrion.config_reader.config_schema import EbpfConfig, TimingConfig
from metrion.data_record import DataRecord
from metrion.entity.cpu import CPU
from metrion.entity.dram import DRAM
from metrion.entity.entity import Entity
from metrion.measurement import Measurement
from metrion.metric import CPU_SOCKET_PKG_ENERGY, CPU_SOCKET_PP0_ENERGY, DRAM_SOCKET_TOTAL_ENERGY, Metric
from metrion.monitoring.data_source.data_source_base import DataSource
from metrion.monitoring.timing_context import TimingContext, TimingContextRegistry


class PowerDomains(Enum):
    """Represents available power domains in Intel RAPL."""

    PKG = "package"
    DRAM = "dram"  # PP1
    CORE = "core"  # PP0


# Name of a top-level zone that meters a CPU package, package-<physical id>. On CPUs with several
# dies per package the kernel names one zone per die instead, package-<physical id>-die-<die id>.
PACKAGE_ZONE_NAME = re.compile(r"package-(\d+)(?:-die-(\d+))?")


class RAPLCPU(DataSource):
    """Represents an Intel RAPL power monitoring data source."""

    _entity_metric_mapping = {
        CPU: {CPU_SOCKET_PKG_ENERGY, CPU_SOCKET_PP0_ENERGY},
        DRAM: {DRAM_SOCKET_TOTAL_ENERGY},
    }

    _domain_to_metric = {
        PowerDomains.PKG: CPU_SOCKET_PKG_ENERGY,
        PowerDomains.DRAM: DRAM_SOCKET_TOTAL_ENERGY,
        PowerDomains.CORE: CPU_SOCKET_PP0_ENERGY,
    }

    def __init__(
        self,
        entities: list[Entity],
        powercap_tree: Path | None = None,
        interval: float | None = None,
        enabled_metrics: list[Metric] | None = None,
        timing: TimingConfig | None = None,
        ebpf: EbpfConfig | None = None,
    ):
        """Initializes a RAPL data source for a list of entities.

        Parameters
        ----------
        entities : list[Entity]
            The entities to monitor (must be of type CPU or RAM).
        powercap_tree : Path, optional
            The base path for powercap data.
            Defaults to /sys/devices/virtual/powercap/intel-rapl.
        interval : float, optional
             The collection interval in seconds.
        """
        super().__init__(entities, interval=interval, enabled_metrics=enabled_metrics, timing=timing, ebpf=ebpf)
        self.powercap_tree = powercap_tree or Path("/sys/devices/virtual/powercap/intel-rapl")
        self._socket_info: dict[int, dict] = {}
        self._timing_registered_pkg = False
        self._timing_registered_dram = False

        for entity in self.entities:
            if not self.is_applicable(entity):
                continue

            package_id = -1
            if isinstance(entity, CPU):
                package_id = entity.package_id
            elif isinstance(entity, DRAM):
                package_id = entity.cpu.package_id

            if package_id not in self._socket_info:
                socket_path = self._package_zone(package_id)
                self._socket_info[package_id] = {
                    "path": socket_path,
                    "metrics": self._discover_metrics_per_socket(socket_path) if socket_path else {},
                    "entities": [],
                }
            self._socket_info[package_id]["entities"].append(entity)

    def _metric_to_entity_type(self, metric: Metric) -> type[Entity]:
        """Map a metric to its corresponding entity type.

        Parameters
        ----------
        metric : Metric
            The metric to map.

        Returns
        -------
        type[Entity]
            The entity type corresponding to the metric.
        """
        for entity_type, metrics in self._entity_metric_mapping.items():
            if metric in metrics:
                return entity_type
        raise ValueError(f"Metric '{metric.name}' is not mapped to any entity type.")

    def _package_zone(self, package_id: int) -> Path | None:
        """Find the top-level RAPL zone of a CPU package by its name, package-<package_id>.

        The zone number is not the package id: zones are numbered in the order the kernel
        registers them, so a platform zone (psys) can take intel-rapl:1 and push package-1
        to intel-rapl:2.

        Parameters
        ----------
        package_id : int
            The physical package id of the socket.

        Returns
        -------
        Path or None
            The zone directory, None (logged as an error) if no zone is named package-<package_id>.
        """
        dies = []
        for zone in sorted(self.powercap_tree.glob("intel-rapl:[0-9]*")):
            name_file = zone / "name"
            if zone.name.count(":") != 1 or not name_file.is_file():
                continue
            match = PACKAGE_ZONE_NAME.fullmatch(name_file.read_text().strip())
            if match is None or int(match[1]) != package_id:
                continue
            if match[2] is None:
                return zone
            dies.append(zone.name)

        if dies:
            # Every die has its own counter and wraps on its own, while the entities and the wrap
            # correction downstream expect one counter per package. Summing them is not done here.
            logger.error(
                f"RAPL: package {package_id} is metered per die ({', '.join(dies)}), which is not supported; "
                f"socket {package_id} gets no package or DRAM energy"
            )
        else:
            logger.error(
                f"RAPL: no zone named package-{package_id} under {self.powercap_tree}; "
                f"socket {package_id} gets no package or DRAM energy"
            )
        return None

    def _discover_metrics_per_socket(self, socket_path: Path) -> dict[Metric, Path]:
        """Discover the power domains of one package zone.

        Parameters
        ----------
        socket_path : Path
            The package zone, one named package-<n> (see _package_zone).

        Returns
        -------
        dict[Metric, Path]
            The metrics of the package and its sub-domains, mapped to their directory paths.
        """
        metrics = {}
        socket_name_file = socket_path / "name"
        if not socket_name_file.is_file() or not PACKAGE_ZONE_NAME.fullmatch(socket_name_file.read_text().strip()):
            # psys and other platform zones are no package, nor are their sub-domains.
            return metrics
        metrics[CPU_SOCKET_PKG_ENERGY] = socket_path

        for domain_dir in socket_path.glob("intel-rapl:*:*"):
            name_file = domain_dir / "name"
            if name_file.exists():
                try:
                    domain_name = name_file.read_text().strip().lower()
                    power_domain = PowerDomains(domain_name)
                    metrics[self._domain_to_metric[power_domain]] = domain_dir
                except ValueError:
                    continue
                except Exception as e:
                    raise RuntimeError(f"Failed reading {name_file}: {e}") from e

        return metrics

    def get_available_metrics(self) -> set[Metric]:
        """Get available metrics based on discovered power domains.

        Returns
        -------
        set[Metric]
            A set of available metrics.
        """
        metrics = set()
        for info in self._socket_info.values():
            for metric in info["metrics"].keys():
                if metric not in metrics:
                    metrics.add(metric)
        return metrics

    @override
    @classmethod
    def is_applicable(cls, entity: Entity) -> bool:
        """Check if the data source is applicable for the given entity.

        Parameters
        ----------
        entity : Entity
            The entity to check.

        Returns
        -------
        bool
            True if the entity is a CPU or RAM, False otherwise.
        """
        return isinstance(entity, CPU | DRAM)

    @override
    def get_data(self, metrics: list[Metric]) -> DataRecord:
        """Retrieve energy consumption data for the specified metrics.

        Parameters
        ----------
        metrics : list[Metric]
            The metrics to retrieve data for.

        Returns
        -------
        DataRecord
            A data record containing the collected measurements.
        """
        measurements = []
        all_entities = []
        available_metrics = self.get_available_metrics()

        for metric in metrics:
            if metric not in available_metrics:
                continue

            entity_type_for_metric = self._metric_to_entity_type(metric)

            for info in self._socket_info.values():
                if metric in info["metrics"].keys():
                    for entity in info["entities"]:
                        if isinstance(entity, entity_type_for_metric):
                            domain_path = info["metrics"][metric]
                            energy = self._read_power_domain_energy(domain_path)

                            measurement = Measurement(
                                value=energy,
                                metric_id=metric.name,
                                physical_entity_id=entity.id,
                                logical_entity_id=None,
                            )
                            measurements.append(measurement)
                            if entity not in all_entities:
                                all_entities.append(entity)

        if not self._timing_registered_pkg:
            for m in measurements:
                if m.metric_id == CPU_SOCKET_PKG_ENERGY.name:
                    start_time = m.start_ts
                    ctx = TimingContext(
                        source_name="intel_rapl_pkg",
                        start_time_ms=int(start_time.timestamp() * 1000),  # epoch ms
                        interval_seconds=self._interval if self._interval else 1.0,
                        start_datetime=start_time,
                    )
                    TimingContextRegistry.instance().register(ctx)
                    self._timing_registered_pkg = True
                    break

        if not self._timing_registered_dram:
            for m in measurements:
                if m.metric_id == DRAM_SOCKET_TOTAL_ENERGY.name:
                    start_time = m.start_ts
                    ctx = TimingContext(
                        source_name="intel_rapl_dram",
                        start_time_ms=int(start_time.timestamp() * 1000),  # epoch ms
                        interval_seconds=self._interval if self._interval else 1.0,
                        start_datetime=start_time,
                    )
                    TimingContextRegistry.instance().register(ctx)
                    self._timing_registered_dram = True
                    break

        return DataRecord(entities=all_entities, measurements=measurements)

    def _read_power_domain_energy(self, domain_path: Path) -> float:
        """Read energy consumption for a given power domain in joules.

        Parameters
        ----------
        domain_path : Path
            The path to the power domain directory.

        Returns
        -------
        float
            Energy consumption in joules.
        """
        energy_file = domain_path / "energy_uj"
        if not energy_file.exists():
            raise FileNotFoundError(f"Energy file not found at {energy_file}")

        try:
            energy_microjouls = int(energy_file.read_text().strip())
            return energy_microjouls / 1e6
        except Exception as e:
            raise RuntimeError(f"Error reading energy sample from {energy_file}: {e}") from e

    @override
    def clean_up(self) -> None:
        """Perform any necessary cleanup."""
        pass

    @override
    def stop(self, metrics: list[Metric]) -> DataRecord | None:
        return self.get_data(metrics)
