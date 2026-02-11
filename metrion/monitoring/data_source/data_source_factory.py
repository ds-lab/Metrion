import importlib
import inspect
import pkgutil

import metrion.monitoring.data_source as ds
from metrion.entity.entity import Entity
from metrion.metric import Metric
from metrion.monitoring.data_source.data_source_base import DataSource


class DataSourceFactory:
    registered_sources: dict[type, dict[Metric, list[DataSource]]] = {}

    @staticmethod
    def register(source_cls: DataSource):
        """
        Registers a new data source class.

        Parameters
        ----------
        source_cls : DataSource
            The data source class to register.
        """
        for type, metrics in source_cls.get_metrics_by_entity_type().items():
            for metric in metrics:
                if type not in DataSourceFactory.registered_sources:
                    DataSourceFactory.registered_sources[type] = {}
                if metric not in DataSourceFactory.registered_sources[type]:
                    DataSourceFactory.registered_sources[type][metric] = []
                DataSourceFactory.registered_sources[type][metric].append(source_cls)

    @staticmethod
    def auto_register():
        """
        Dynamically loads and registers all DataSource subclasses from the `data_source` package.

        This method recursively walks through the `data_source` package and its subpackages,
        importing modules and registering all classes that inherit from `DataSource`.
        """

        def walk_package(package):
            """Recursively walk through a package and its subpackages, registering DataSource subclasses.

            Parameters
            ----------
            package : module
                The package to walk through.
            """
            package_name = package.__name__

            for _finder, name, ispkg in pkgutil.iter_modules(package.__path__):
                full_name = f"{package_name}.{name}"

                module = importlib.import_module(full_name)

                if ispkg:
                    walk_package(module)

                for _class_name, obj in inspect.getmembers(
                    module,
                    lambda obj, module=module: inspect.isclass(obj) and obj.__module__ == module.__name__,
                ):
                    if issubclass(obj, DataSource) and obj is not DataSource:
                        DataSourceFactory.register(obj)

        walk_package(ds)

    @staticmethod
    def get_sources_by_entity_type_and_metric_filter(
        entity: type[Entity], metrics: list[Metric]
    ) -> dict[type[DataSource], list[Metric]]:
        """
        Returns a dictionary of data sources for the given entity type and metrics.

        Parameters
        ----------
        entity : type of Entity
            The entity type to filter data sources by.
        metrics : list of Metric
            The metrics to filter data sources by.

        Returns
        -------
        dict of DataSource to list of Metric
            A dictionary mapping data source classes to the list of metrics they provide for the given entity type.
        """
        if entity not in DataSourceFactory.registered_sources:
            return {}

        sources = {}
        for metric in metrics:
            if metric in DataSourceFactory.registered_sources[entity]:
                source_cls = DataSourceFactory.registered_sources[entity][metric][0]
                if source_cls not in sources:
                    sources[source_cls] = []
                sources[source_cls].append(metric)

        return sources

    @staticmethod
    def create_sources(metrics: list[Metric], entity: Entity) -> dict[Metric, DataSource]:
        """
        Creates instances of the appropriate data sources for the given metrics and entity.

        Parameters
        ----------
        metrics : list of Metric
            The metrics for which to create data source instances.
        entity : Entity
            The entity for which the data sources are created.

        Returns
        -------
        dict of Metric to DataSource
            A dictionary mapping each metric to its corresponding data source instance.
        """
        sources = {}
        created_ds_instances = []
        for metric in metrics:
            for ds_cls in DataSourceFactory.registered_sources.values():
                if ds_cls.is_applicable(entity):
                    ds_instance = ds_cls(entity=entity)
                    if metric in ds_instance.get_available_metrics():
                        if type(ds_instance) in [type(existing_ds) for existing_ds in created_ds_instances]:
                            existing_ds = next(
                                existing_ds
                                for existing_ds in created_ds_instances
                                if isinstance(existing_ds, ds_instance.__class__)
                            )
                            sources[metric] = existing_ds
                        else:
                            sources[metric] = ds_instance
                            created_ds_instances.append(ds_instance)
        return sources
