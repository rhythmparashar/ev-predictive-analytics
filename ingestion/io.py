"""
IO utilities:
- Find raw CSV files under data/raw/dt=YYYY-MM-DD/
- Read them
- Write parquet outputs with atomic replace (prevents partial/corrupted files)
- Build partition paths consistently
"""

from __future__ import annotations

import csv
import io as _io
import re
from pathlib import Path
import pandas as pd

VEHICLE_RE = re.compile(r"vehicle_id=([^./]+)\.csv$", re.IGNORECASE)


def dt_dir(base: Path, dt: str) -> Path:
    """Return partition folder path: base/dt=YYYY-MM-DD"""
    return base / f"dt={dt}"


def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def list_raw_csvs(raw_dir: Path, dt: str) -> list[Path]:
    """List raw CSV files for a given date partition."""
    d = dt_dir(raw_dir, dt)
    if not d.exists():
        return []
    return sorted(d.glob("vehicle_id=*.csv"))


def vehicle_id_from_filename(path: Path) -> str:
    """Extract vehicle_id from file name pattern vehicle_id=EV01.csv"""
    m = VEHICLE_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse vehicle_id from filename: {path.name}")
    return m.group(1)


def read_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV with tolerance for rows that have more fields than the header.

    Extra fields in a row are silently truncated to match the column count.
    This handles files where a split script appended rows with additional
    columns (e.g. dt, vehicle_id) onto an existing file — those extra columns
    are dropped, keeping the row's valid data intact.

    Uses Python's csv.reader so quoted fields containing commas (e.g. a
    timestamp like "09/03/2026, 09:57:57") are handled correctly.
    """
    rows: list[list[str]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()
        ncols = len(header)
        rows.append(header)
        for row in reader:
            rows.append(row[:ncols])

    buf = _io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    buf.seek(0)
    return pd.read_csv(buf, low_memory=False)


def parquet_path(base: Path, dt: str, vehicle_id: str) -> Path:
    """Build parquet output path: base/dt=.../vehicle_id=...parquet"""
    return dt_dir(base, dt) / f"vehicle_id={vehicle_id}.parquet"


def fault_csv_path(raw_fault_dir: Path, dt: str, vehicle_id: str) -> Path:
    """Build fault CSV input path."""
    return dt_dir(raw_fault_dir, dt) / f"vehicle_id={vehicle_id}.csv"


def write_parquet_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """
    Atomic write:
    write to tmp then replace, preventing half-written files on crash.
    """
    ensure_dir(out_path.parent)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)