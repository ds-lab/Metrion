from metrion.data_storage.i_data_storage import DataStorageI


class DataStorageFactory:
    def create_data_storage(self, storage_type: str, target: str) -> DataStorageI:
        """
        Create the storage backend for a storage type.

        Parameters
        ----------
        storage_type : str
            One of 'duckdb', 'sqlite', 'postgresql', 'parquet'.
        target : str
            Connection string for the database backends, output directory for parquet.
        """
        if storage_type in ("postgresql", "sqlite", "duckdb"):
            from metrion.data_storage.db_data_storage import SQLDatabaseStorage

            return SQLDatabaseStorage(target)

        if storage_type == "parquet":
            from metrion.data_storage.parquet_data_storage import ParquetDataStorage

            return ParquetDataStorage(target)

        raise ValueError(f"Unknown storage type: {storage_type}")
