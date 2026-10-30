"""
Memory-aware data I/O utilities.

Principles
----------
- Never load full large tables (bureau, installments) into pandas memory.
- Use DuckDB for SQL aggregations against raw CSV files.
- Write results as Parquet for fast subsequent reads.
- Chunked CSV reading available for medium-sized tables.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd

from credit_risk.config import CSV_ENCODING, DUCKDB_MEMORY_LIMIT, DUCKDB_THREADS
from credit_risk.logger import get_logger

logger = get_logger("credit_risk.data_io")


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with conservative resource limits."""
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}';")
    con.execute(f"SET threads={DUCKDB_THREADS};")
    return con


def read_csv_header(path: Path) -> list[str]:
    """Read only the header row from a CSV file."""
    with open(path, "r", encoding=CSV_ENCODING) as f:
        return next(csv_reader(f))


def csv_reader(file_obj):
    """Yield rows from a file object as lists (raw CSV parsing)."""
    import csv
    return csv.reader(file_obj)


def read_csv_sample(path: Path, n_rows: int = 1000) -> pd.DataFrame:
    """Read the first n_rows from a CSV file into a DataFrame."""
    logger.info("Reading %d-row sample from %s", n_rows, path.name)
    return pd.read_csv(path, nrows=n_rows, encoding=CSV_ENCODING, low_memory=False)


def read_csv_full(path: Path, **kwargs) -> pd.DataFrame:
    """
    Read an entire CSV into a DataFrame.
    Use ONLY for files that fit comfortably in memory
    (application_train ~166 MB → ~300 MB in DataFrame).
    """
    logger.info("Reading full CSV: %s", path.name)
    return pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False, **kwargs)


def read_csv_chunks(path: Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a large CSV file."""
    logger.info("Streaming %s in chunks of %d", path.name, chunksize)
    yield from pd.read_csv(
        path, encoding=CSV_ENCODING, low_memory=False, chunksize=chunksize
    )


def duckdb_query_to_df(sql: str, con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Execute a DuckDB SQL query and return the result as a DataFrame."""
    close = False
    if con is None:
        con = get_duckdb_connection()
        close = True
    try:
        return con.execute(sql).df()
    finally:
        if close:
            con.close()


def duckdb_csv_aggregate(
    csv_path: Path,
    sql_template: str,
    output_parquet: Path,
    con: duckdb.DuckDBPyConnection | None = None,
) -> pd.DataFrame:
    """
    Run a DuckDB aggregation SQL against a raw CSV file and write the result
    to a Parquet file.  The SQL template uses {csv_path} as a placeholder.

    Parameters
    ----------
    csv_path     : Path to the raw CSV file.
    sql_template : SQL string with `{csv_path}` placeholder for the file path.
    output_parquet : Where to write the aggregated Parquet file.
    con          : Optional existing connection; a new one is created otherwise.

    Returns
    -------
    DataFrame of the aggregated result.
    """
    close = False
    if con is None:
        con = get_duckdb_connection()
        close = True

    # Normalise path separators for DuckDB on Windows
    csv_str = str(csv_path).replace("\\", "/")
    sql = sql_template.format(csv_path=csv_str)

    logger.info("Running DuckDB aggregation on %s → %s", csv_path.name, output_parquet.name)
    try:
        df = con.execute(sql).df()
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_parquet, index=False)
        logger.info("Wrote %d rows × %d cols to %s", len(df), df.shape[1], output_parquet.name)
        return df
    finally:
        if close:
            con.close()


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    logger.info("Reading parquet: %s", path.name)
    return pd.read_parquet(path)


def write_json(data: dict, path: Path) -> None:
    """Write a dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Wrote JSON: %s", path.name)


def read_json(path: Path) -> dict:
    """Read a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
