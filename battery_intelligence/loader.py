"""
battery_intelligence/loader.py
Low-memory parquet loaders using DuckDB.

DuckDB reads only the columns you request and skips entire Hive partitions
when date filters are applied — RAM usage stays proportional to your query
result, not the full dataset size.

Usage
-----
    from battery_intelligence.loader import load_master, load_gold_cells, load_silver_cells, load_raw

    master = load_master(DATA_DIR, columns=['timestamp', 'soc_pct', 'avg_battery_temp_c'])
    gold   = load_gold_cells(DATA_DIR, columns=['timestamp', 'M1_C25_delta', 'M1_C25_is_outlier'])
    silver = load_silver_cells(DATA_DIR, columns=['timestamp', 'M1_C1', 'M1_C2'])
    raw    = load_raw(DATA_DIR, columns=['timestamp', 'last_trip_kwh', 'total_kwh_consumed'])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd


def _glob(path: Path) -> str:
    return str(path).replace("\\", "/")


def load_master(
    data_dir: Path,
    columns: Optional[list[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load master dataset. Reads only requested columns from disk.

    Parameters
    ----------
    data_dir : Path  Root data directory (DATA_DIR from configs.settings).
    columns  : list  Column names to load. None = all columns.
    start_date, end_date : str  ISO date strings e.g. '2026-04-01' for partition pruning.
    """
    path = _glob(data_dir / "analysis" / "master" / "**" / "*.parquet")
    col_sql = ", ".join(columns) if columns else "*"

    where_clauses = []
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp < '{end_date}'")
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    df = duckdb.query(f"""
        SELECT {col_sql}
        FROM read_parquet('{path}', hive_partitioning=true)
        {where}
        ORDER BY timestamp
    """).df()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_gold_cells(
    data_dir: Path,
    columns: Optional[list[str]] = None,
    vehicle_id: str = "EV01",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load gold cell_features. Reads only requested columns from disk."""
    path = _glob(data_dir / "gold_cells" / "cell_features" / "**" / "*.parquet")
    col_sql = ", ".join(columns) if columns else "*"

    where_clauses = [f"vehicle_id = '{vehicle_id}'"]
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp < '{end_date}'")
    where = "WHERE " + " AND ".join(where_clauses)

    df = duckdb.query(f"""
        SELECT {col_sql}
        FROM read_parquet('{path}', hive_partitioning=true)
        {where}
        ORDER BY timestamp
    """).df()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_silver_cells(
    data_dir: Path,
    columns: Optional[list[str]] = None,
    vehicle_id: str = "EV01",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load silver_cells. Reads only requested columns from disk."""
    path = _glob(data_dir / "silver_cells" / "**" / "*.parquet")
    col_sql = ", ".join(columns) if columns else "*"

    where_clauses = [f"vehicle_id = '{vehicle_id}'"]
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp < '{end_date}'")
    where = "WHERE " + " AND ".join(where_clauses)

    df = duckdb.query(f"""
        SELECT {col_sql}
        FROM read_parquet('{path}', hive_partitioning=true)
        {where}
        ORDER BY timestamp
    """).df()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_raw(
    data_dir: Path,
    columns: Optional[list[str]] = None,
    vehicle_id: str = "EV01",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw_parquet signals. Strips UTC timezone so merges with master work."""
    path = _glob(data_dir / "raw_parquet" / "**" / "*.parquet")

    base_cols = list(columns) if columns else None
    # Always need vehicle_id for filtering even if caller didn't ask for it
    if base_cols and "vehicle_id" not in base_cols:
        fetch_cols = ["vehicle_id"] + base_cols
        drop_vehicle = True
    else:
        fetch_cols = base_cols
        drop_vehicle = False

    col_sql = ", ".join(fetch_cols) if fetch_cols else "*"

    where_clauses = [f"vehicle_id = '{vehicle_id}'"]
    if start_date:
        where_clauses.append(f"timestamp >= '{start_date}'")
    if end_date:
        where_clauses.append(f"timestamp < '{end_date}'")
    where = "WHERE " + " AND ".join(where_clauses)

    df = duckdb.query(f"""
        SELECT {col_sql}
        FROM read_parquet('{path}', hive_partitioning=true)
        {where}
        ORDER BY timestamp
    """).df()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(None)

    if drop_vehicle and "vehicle_id" in df.columns:
        df = df.drop(columns=["vehicle_id"])

    return df.drop_duplicates(subset="timestamp")
