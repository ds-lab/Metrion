import tempfile
from pathlib import Path

import duckdb
from loguru import logger

# Stream directories are named after the class (lowercased); a few DB tables are not.
STREAM_TABLE_NAMES = {
    "logicalentity": "logical_entity",
    "networkinterface": "network_interface",
}

# _to_df() and the DB tables disagree on a few names. The DB name wins, because the .sql
# analyses are written against it. See _logical_entity_row in db_data_storage.py.
COLUMN_RENAMES = {
    "logical_entity": {"native_parent_thread_id": "native_parent_tid"},
}

BOOKKEEPING_COLUMNS = ("_seq", "_entity_type")

# Name of the database built inside a run directory.
DUCKDB_NAME = "monitor.db"


def _columns(con, source: str) -> list[tuple[str, str]]:
    return [(row[0], row[1]) for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]


def _as_timestamp(expression: str, dtype: str) -> str:
    """
    Normalise a timestamp expression to the naive TIMESTAMP the DuckDB path stores.

    Parameters
    ----------
    expression : str
        SQL expression of the source column.
    dtype : str
        Its DuckDB type, as reported by DESCRIBE.
    """
    if dtype == "TIMESTAMP WITH TIME ZONE":
        # AT TIME ZONE 'UTC' pins the wall clock to UTC; a plain CAST would move it into
        # the session time zone instead.
        return f"CAST({expression} AT TIME ZONE 'UTC' AS TIMESTAMP)"

    if dtype.startswith("TIMESTAMP"):
        return f"CAST({expression} AS TIMESTAMP)"

    return expression


def _plain_projection(con, source: str) -> str:
    """Build the SELECT list for a stream that is copied row for row."""
    parts = []
    for name, dtype in _columns(con, source):
        column = f'"{name}"'
        parts.append(f"{_as_timestamp(column, dtype)} AS {column}")

    return ", ".join(parts)


def _merge_projection(con, source: str, table: str) -> str:
    """
    Build the "last non-empty value per column wins" projection for an entity stream.

    Mirrors the coalesce(nullif(...)) upsert of the DuckDB path: a later version only
    overrides a column it actually fills.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection used to inspect the source.
    source : str
        A read_parquet(...) expression.
    table : str
        Target table name, used to look up column renames.

    Returns
    -------
    str
        The SELECT list, without the GROUP BY.
    """
    renames = COLUMN_RENAMES.get(table, {})
    parts = ["id"]

    for name, dtype in _columns(con, source):
        if name == "id" or name in BOOKKEEPING_COLUMNS:
            continue
        condition = f'"{name}" IS NOT NULL'
        if dtype == "VARCHAR":
            condition += f" AND \"{name}\" <> ''"
        target = renames.get(name, name)
        value = f'max_by("{name}", _seq) FILTER (WHERE {condition})'
        parts.append(f'{_as_timestamp(value, dtype)} AS "{target}"')

    return ", ".join(parts)


def _entity_union(con, type_dirs: list[Path]) -> str | None:
    """Build the SELECT that feeds the shared entity table, or None if nothing qualifies."""
    selects = []
    for type_dir in type_dirs:
        source = f"read_parquet('{type_dir}/*.parquet')"
        names = {name for name, _ in _columns(con, source)}
        if "id" not in names:
            continue
        parent = '"parent_id"' if "parent_id" in names else "CAST(NULL AS VARCHAR)"
        entity_type = '"_entity_type"' if "_entity_type" in names else f"'{type_dir.name}'"
        selects.append(
            f"SELECT id, max_by({parent}, _seq) FILTER (WHERE {parent} IS NOT NULL) AS parent_id, "
            f"any_value({entity_type}) AS entity_type FROM {source} GROUP BY id"
        )

    return " UNION ALL ".join(selects) if selects else None


def _add_missing_tables(con, present: set[str]) -> None:
    """
    Copy the tables the streams did not fill out of a freshly created database.

    A run without a GPU writes no gpu stream, and an analysis joining an absent table fails
    with a catalog error instead of returning nothing. The reference also carries the metric
    rows that the DuckDB path writes on startup.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Connection to the file being built.
    present : set[str]
        Tables that the streams already produced.
    """
    from metrion.data_storage.db_data_storage import SQLDatabaseStorage

    with tempfile.TemporaryDirectory() as directory:
        reference = Path(directory) / "reference.duckdb"
        SQLDatabaseStorage(f"duckdb:///{reference}").close()

        con.execute(f"ATTACH '{reference}' AS reference (READ_ONLY)")
        try:
            missing = con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = 'reference'").fetchall()
            for (name,) in missing:
                if name not in present:
                    # Copied, not created: the reference tables carry foreign keys on entity,
                    # which a table built by CREATE TABLE AS SELECT cannot satisfy.
                    con.execute(f"CREATE TABLE {name} AS SELECT * FROM reference.{name}")
        finally:
            con.execute("DETACH reference")


def convert(parquet_root: str | Path, duckdb_path: str | Path) -> dict[str, int]:
    """
    Build a DuckDB file from the Parquet streams of a run.

    Entities are written append-only, so a stream may hold several versions of one id.
    They are merged column by column, the last non-empty value winning.

    Parameters
    ----------
    parquet_root : str | Path
        Directory that ParquetDataStorage wrote to.
    duckdb_path : str | Path
        DuckDB file to create. It is derived entirely from the streams and rebuilt from
        scratch, so an existing file at that path is replaced.

    Returns
    -------
    dict[str, int]
        Table name -> row count.
    """
    root = Path(parquet_root)
    out = Path(duckdb_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)

    con = duckdb.connect(str(out))
    counts: dict[str, int] = {}
    try:
        con.execute("SET preserve_insertion_order=false")

        measurements = root / "measurements"
        if any(measurements.glob("*.parquet")):
            source = f"read_parquet('{measurements}/*.parquet')"
            con.execute(f"CREATE OR REPLACE TABLE measurement AS SELECT {_plain_projection(con, source)} FROM {source}")
            counts["measurement"] = con.execute("SELECT count(*) FROM measurement").fetchone()[0]

        comm = root / "communication_measurements"
        if any(comm.glob("*.parquet")):
            con.execute(
                "CREATE OR REPLACE TABLE communication_measurement AS "
                "SELECT id AS measurement_id, src_endpoint, dst_endpoint "
                f"FROM read_parquet('{comm}/*.parquet')"
            )
            counts["communication_measurement"] = con.execute(
                "SELECT count(*) FROM communication_measurement"
            ).fetchone()[0]

        entities_dir = root / "entities"
        type_dirs = []
        if entities_dir.exists():
            type_dirs = sorted(d for d in entities_dir.iterdir() if d.is_dir() and any(d.glob("*.parquet")))

        for type_dir in type_dirs:
            table = STREAM_TABLE_NAMES.get(type_dir.name, type_dir.name)
            source = f"read_parquet('{type_dir}/*.parquet')"
            projection = _merge_projection(con, source, table)
            con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT {projection} FROM {source} GROUP BY id")
            counts[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

        union = _entity_union(con, type_dirs)
        if union is not None:
            con.execute(f"CREATE OR REPLACE TABLE entity AS SELECT * FROM ({union})")
            counts["entity"] = con.execute("SELECT count(*) FROM entity").fetchone()[0]

        _add_missing_tables(con, set(counts))

        con.execute("CHECKPOINT")
    finally:
        con.close()

    logger.info(f"Converted Parquet to DuckDB: {out} — {counts}")
    return counts


def ensure_duckdb(parquet_root: str | Path) -> Path:
    """
    Return the DuckDB file of a run directory, building it if it is missing or stale.

    Parameters
    ----------
    parquet_root : str | Path
        Directory that ParquetDataStorage wrote to.

    Returns
    -------
    Path
        The DuckDB file inside that directory.
    """
    root = Path(parquet_root)
    out = root / DUCKDB_NAME
    newest_stream = max((path.stat().st_mtime for path in root.rglob("*.parquet")), default=0.0)

    if out.exists() and out.stat().st_mtime >= newest_stream:
        logger.info(f"Reusing {out}")
        return out

    convert(root, out)
    return out
