"""
File I/O utilities for Battery Intelligence Engine.
All paths come from configs/settings.py or are passed as arguments.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd


def load_master_dataset(master_dir: Path) -> pd.DataFrame:
    """
    Load all Hive-partitioned parquet files from data/analysis/master/.
    Reads all dt= subdirectories recursively.

    Returns concatenated DataFrame sorted by timestamp.
    """
    files = sorted(master_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {master_dir}")

    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as exc:
            print(f"  [WARN] Skipped {f.name}: {exc}")

    if not parts:
        raise ValueError(f"All partitions failed to load from {master_dir}")

    df = pd.concat(parts, ignore_index=True)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to parquet using write-to-tmp + rename for atomicity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    try:
        df.to_parquet(tmp, index=False)
        shutil.move(str(tmp), str(path))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data: dict | list, path: Path) -> None:
    """Write dict or list to JSON with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)


def write_markdown(text: str, path: Path) -> None:
    """Write markdown report text to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
