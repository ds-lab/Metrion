import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import polars as pl
import psutil
import pynvml
from loguru import logger

from metrion.data_record import DataRecord
from metrion.entity.entity import Entity
from metrion.entity.gpu import GPU
from metrion.entity.logical_entity_registry import LogicalEntityRegistry
from metrion.measurement import Measurement
from metrion.metric import (
    GPU_DEVICE_PID_COMPUTE_UTILIZATION,
    GPU_DEVICE_PID_DECODER_UTILIZATION,
    GPU_DEVICE_PID_ENCODER_UTILIZATION,
    GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION,
    GPU_DEVICE_PID_MEMORY,
    GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION,
    GPU_DEVICE_TOTAL_ENERGY,
    GPU_DEVICE_TOTAL_MEMORY,
    GPU_DEVICE_TOTAL_MEMORY_FREE,
    GPU_DEVICE_TOTAL_MEMORY_USED,
    GPU_DEVICE_TOTAL_MEMORY_UTILIZATION,
    GPU_DEVICE_TOTAL_POWER_AVERAGE,
    GPU_DEVICE_TOTAL_POWER_INSTANT,
    GPU_DEVICE_TOTAL_TEMPERATURE,
    Metric,
)
from metrion.monitoring.data_source.cpu.utils import get_pid_starttime_ns
from metrion.monitoring.data_source.data_source_base import DataSource
from metrion.monitoring.timing_context import TimingContext, TimingContextRegistry


class NVMLDataSource(DataSource):
    """
    A data source for NVIDIA GPUs using the NVML library.
    """

    _entity_metric_mapping: dict[type[Entity], set[Metric]] = {
        GPU: {
            GPU_DEVICE_TOTAL_MEMORY_USED,
            GPU_DEVICE_TOTAL_MEMORY_FREE,
            GPU_DEVICE_TOTAL_MEMORY,
            GPU_DEVICE_TOTAL_ENERGY,
            GPU_DEVICE_TOTAL_TEMPERATURE,
            GPU_DEVICE_TOTAL_MEMORY_UTILIZATION,
            GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION,
            GPU_DEVICE_TOTAL_POWER_INSTANT,
            GPU_DEVICE_TOTAL_POWER_AVERAGE,
            GPU_DEVICE_PID_MEMORY,
            GPU_DEVICE_PID_COMPUTE_UTILIZATION,
            GPU_DEVICE_PID_ENCODER_UTILIZATION,
            GPU_DEVICE_PID_DECODER_UTILIZATION,
            GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION,
        }
    }

    def __init__(self, entities: list[Entity], interval: float | None = None):
        """
        Initializes the NVML data source.

        Parameters
        ----------
        entities : list of Entity
            A list of entities to monitor. This data source will filter for GPU entities.
        interval : float, optional
             The collection interval in seconds.
        """
        super().__init__(entities, interval=interval)
        self.gpu_entities: list[GPU] = [e for e in self.entities if isinstance(e, GPU)]
        self._handles: dict[str, Any] = {}
        self._last_seen: dict[str, int] = {}
        self._timing_registered = False
        self._logical_entity_registry = LogicalEntityRegistry.get_instance()

        try:
            pynvml.nvmlInit()
            logger.debug("NVML initialized successfully")
            for gpu in self.gpu_entities:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(gpu.uuid)
                    self._handles[gpu.uuid] = handle
                    self._last_seen[gpu.uuid] = 0
                except pynvml.NVMLError as e:
                    logger.warning(f"Could not get handle for GPU with UUID {gpu.uuid}: {e}")
        except pynvml.NVMLError as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.gpu_entities = []

        self._metric_lookup: dict[Metric, Callable[[], DataRecord]] = {
            GPU_DEVICE_TOTAL_MEMORY_USED: self._read_device_memory_consumption,
            GPU_DEVICE_TOTAL_MEMORY_FREE: self._read_device_memory_consumption,
            GPU_DEVICE_TOTAL_MEMORY: self._read_device_memory_consumption,
            GPU_DEVICE_TOTAL_ENERGY: self._read_device_energy_consumption,
            GPU_DEVICE_TOTAL_TEMPERATURE: self._read_device_temperature,
            GPU_DEVICE_TOTAL_MEMORY_UTILIZATION: self._read_device_utilization,
            GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION: self._read_device_utilization,
            GPU_DEVICE_TOTAL_POWER_INSTANT: self._read_device_power,
            GPU_DEVICE_TOTAL_POWER_AVERAGE: self._read_device_power,
            GPU_DEVICE_PID_MEMORY: self._read_pids_memory_consumption,
            GPU_DEVICE_PID_COMPUTE_UTILIZATION: self._read_pids_compute_utilization,
            GPU_DEVICE_PID_ENCODER_UTILIZATION: self._read_pids_compute_utilization,
            GPU_DEVICE_PID_DECODER_UTILIZATION: self._read_pids_compute_utilization,
            GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION: self._read_pids_compute_utilization,
        }

    @classmethod
    def is_applicable(cls, entity: Entity) -> bool:
        """
        Checks if this data source is applicable for the given entity.

        Parameters
        ----------
        entity : Entity
            The entity to check.

        Returns
        -------
        bool
            True if the entity is a GPU and NVML can be initialized, False otherwise.
        """
        if not isinstance(entity, GPU):
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except pynvml.NVMLError:
            return False

    def get_data(self, metrics: list[Metric]) -> DataRecord:
        """
        Collects data for the specified metrics from the GPU entities.
        """
        measurements: list[Measurement | pl.DataFrame] = []
        entities: list[Entity] = []
        unique_metrics = set(metrics)

        called_methods = set()
        for metric in unique_metrics:
            if metric in self._metric_lookup:
                method = self._metric_lookup[metric]
                if method not in called_methods:
                    data_record = method()
                    measurements.extend(data_record.measurements)
                    entities.extend(data_record.entities)
                    called_methods.add(method)

        return DataRecord(entities=entities, measurements=measurements)

    def _read_device_memory_consumption(self) -> DataRecord:
        """
        Reads memory consumption for all managed GPUs.
        """
        measurements = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                now = datetime.now(UTC)

                measurements.append(
                    Measurement(
                        value=float(memory_info.used) / 1024 / 1024,
                        metric_id=GPU_DEVICE_TOTAL_MEMORY_USED.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=now,
                    )
                )
                measurements.append(
                    Measurement(
                        value=float(memory_info.free) / 1024 / 1024,
                        metric_id=GPU_DEVICE_TOTAL_MEMORY_FREE.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=now,
                    )
                )
                measurements.append(
                    Measurement(
                        value=float(memory_info.total) / 1024 / 1024,
                        metric_id=GPU_DEVICE_TOTAL_MEMORY.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=now,
                    )
                )
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=list(self.gpu_entities), measurements=measurements)

    def _read_device_energy_consumption(self) -> DataRecord:
        """
        Reads total energy consumption for all managed GPUs.
        """

        measurements = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                energy_consumption = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # mJ
                measurements.append(
                    Measurement(
                        value=energy_consumption * 0.001,  # Convert to Joules
                        metric_id=GPU_DEVICE_TOTAL_ENERGY.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=datetime.now(UTC),
                    )
                )
            except (pynvml.NVMLError, KeyError):
                pass

        if not self._timing_registered:
            ctx = TimingContext(
                source_name="nvidia_smi",
                start_time_ms=int(time.time() * 1000),  # epoch ms
                interval_seconds=self._interval if self._interval else 1.0,
                start_datetime=datetime.now(UTC),
            )
            TimingContextRegistry.instance().register(ctx)
            self._timing_registered = True

        return DataRecord(entities=list(self.gpu_entities), measurements=measurements)

    def _read_device_power(self) -> DataRecord:
        """
        Read power consumption for all managed GPUs.

        Emits GPU_DEVICE_TOTAL_POWER_INSTANT for pre-Ampere/GA100 GPUs,
        or GPU_DEVICE_TOTAL_POWER_AVERAGE for Ampere+ (except GA100).

        The metric selection is based on `gpu.has_averaged_power_api`:
        - True: GPU returns 1-second averaged power → GPU_DEVICE_TOTAL_POWER_AVERAGE
        - False: GPU returns instantaneous power → GPU_DEVICE_TOTAL_POWER_INSTANT

        Returns
        -------
        DataRecord
            Power measurements in Watts.
        """
        measurements = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                power_w = power_mw / 1000.0  # Convert to Watts

                # Select metric based on GPU generation
                if gpu.has_averaged_power_api:
                    metric_id = GPU_DEVICE_TOTAL_POWER_AVERAGE.name
                else:
                    metric_id = GPU_DEVICE_TOTAL_POWER_INSTANT.name

                measurements.append(
                    Measurement(
                        value=power_w,
                        metric_id=metric_id,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=datetime.now(UTC),
                    )
                )
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=list(self.gpu_entities), measurements=measurements)

    def _read_device_utilization(self) -> DataRecord:
        """
        Reads utilization rates for all managed GPUs.
        """
        measurements = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                device_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                now = datetime.now(UTC)

                measurements.append(
                    Measurement(
                        value=device_utilization.gpu,
                        metric_id=GPU_DEVICE_TOTAL_COMPUTE_UTILIZATION.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=now,
                    )
                )
                measurements.append(
                    Measurement(
                        value=device_utilization.memory,
                        metric_id=GPU_DEVICE_TOTAL_MEMORY_UTILIZATION.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=now,
                    )
                )
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=list(self.gpu_entities), measurements=measurements)

    def _read_device_temperature(self) -> DataRecord:
        """
        Reads temperature for all managed GPUs.
        """
        measurements = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                measurements.append(
                    Measurement(
                        value=temperature,
                        metric_id=GPU_DEVICE_TOTAL_TEMPERATURE.name,
                        physical_entity_id=gpu.id,
                        logical_entity_id=None,
                        start_ts=datetime.now(UTC),
                    )
                )
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=list(self.gpu_entities), measurements=measurements)

    def _read_pids_memory_consumption(self) -> DataRecord:
        """
        Reads memory consumption for running processes on all managed GPUs.
        """
        measurements = []
        entities = []
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                compute_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                self._logical_entity_registry.update()
                for process_info in compute_processes:
                    try:
                        pid = process_info.pid
                        pid_start_ns = get_pid_starttime_ns(pid)
                        logical_entity = self._logical_entity_registry.get_logical_entity(
                            pid=pid, start_time=pid_start_ns
                        )
                        if logical_entity is None:
                            logger.warning(f"Could not find logical entity for pid {pid}")
                            continue
                        measurements.append(
                            Measurement(
                                value=float(process_info.usedGpuMemory) / 1024 / 1024,
                                metric_id=GPU_DEVICE_PID_MEMORY.name,
                                physical_entity_id=gpu.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=datetime.now(UTC),
                            )
                        )
                        entities.append(logical_entity)
                        entities.append(gpu)
                    except (psutil.NoSuchProcess, Exception) as e:
                        logger.warning(f"Error processing pid {process_info.pid}: {e}")
                        continue
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=entities, measurements=measurements)

    def _read_pids_compute_utilization(self) -> DataRecord:
        """
        Reads compute utilization for running processes on all managed GPUs.
        """
        measurements = []
        entities = []
        now = datetime.now(UTC)
        for gpu in self.gpu_entities:
            try:
                handle = self._handles[gpu.uuid]
                process_samples = pynvml.nvmlDeviceGetProcessUtilization(handle, self._last_seen[gpu.uuid])
                self._last_seen[gpu.uuid] = int(now.timestamp())

                self._logical_entity_registry.update()
                for sample in process_samples:
                    try:
                        pid = sample.pid
                        pid_start_ns = get_pid_starttime_ns(pid)
                        logical_entity = self._logical_entity_registry.get_logical_entity(
                            pid=pid, start_time=pid_start_ns
                        )
                        if logical_entity is None:
                            logger.warning(f"Could not find logical entity for pid {pid}")
                            continue
                        measurements.append(
                            Measurement(
                                value=sample.smUtil,
                                metric_id=GPU_DEVICE_PID_COMPUTE_UTILIZATION.name,
                                physical_entity_id=gpu.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=now,
                            )
                        )
                        measurements.append(
                            Measurement(
                                value=sample.encUtil,
                                metric_id=GPU_DEVICE_PID_ENCODER_UTILIZATION.name,
                                physical_entity_id=gpu.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=now,
                            )
                        )
                        measurements.append(
                            Measurement(
                                value=sample.decUtil,
                                metric_id=GPU_DEVICE_PID_DECODER_UTILIZATION.name,
                                physical_entity_id=gpu.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=now,
                            )
                        )
                        measurements.append(
                            Measurement(
                                value=sample.memUtil,
                                metric_id=GPU_DEVICE_PID_FRAME_BUFFER_UTILIZATION.name,
                                physical_entity_id=gpu.id,
                                logical_entity_id=logical_entity.id,
                                start_ts=now,
                            )
                        )
                        entities.append(logical_entity)
                        entities.append(gpu)
                    except Exception as e:
                        logger.warning(f"Error processing pid {sample.pid}: {e}")
                        continue
            except (pynvml.NVMLError, KeyError):
                pass
        return DataRecord(entities=entities, measurements=measurements)

    def clean_up(self) -> None:
        return

    def stop(self, metrics: list[Metric]) -> DataRecord | None:
        final_data_records = self.get_data(metrics)
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as e:
            logger.error(f"Error shutting down NVML: {e}")
        return final_data_records
