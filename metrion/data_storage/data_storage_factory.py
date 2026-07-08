from metrion.data_storage.i_data_storage import DataStorageI


class DataStorageFactory:
    def __init__(self):
        super().__init__()

    def create_data_storage(self, storage_type: str, *args, **kwargs) -> DataStorageI:
        """
        Factory method to create a data storage instance based on the specified type.
        :param storage_type: Type of data storage (e.g., 'postgresql', 'sqlite').
        :return: An instance of the specified data storage type.
        """
        if storage_type == "postgresql":
            from metrion.data_storage.db_data_storage import SQLDatabaseStorage

            db_url = str(kwargs.get("db_url"))
            return SQLDatabaseStorage(db_url)
        elif storage_type == "sqlite":
            from metrion.data_storage.db_data_storage import SQLDatabaseStorage

            db_url = str(kwargs.get("db_url"))
            return SQLDatabaseStorage(db_url)
        elif storage_type == "duckdb":
            from metrion.data_storage.db_data_storage import SQLDatabaseStorage

            db_url = str(kwargs.get("db_url"))
            return SQLDatabaseStorage(db_url)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")
