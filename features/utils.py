#features/utils.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_sorted_1hz(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("Expected column 'timestamp' in silver data.")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def add_missing_columns(df: pd.DataFrame, cols: Iterable[str], fill_value=pd.NA) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def stable_column_order(df: pd.DataFrame, first: List[str]) -> pd.DataFrame:
    first_present = [c for c in first if c in df.columns]
    rest = sorted([c for c in df.columns if c not in set(first_present)])
    return df[first_present + rest]


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == "bool":
        return s.fillna(False)
    return s.fillna(0).astype("int64").astype("bool")


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp.{path.name}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    tmp.replace(path)


def try_import_atomic_parquet_writer():
    """
    Prefer ingestion.io.write_parquet_atomic if available.
    Fallback: write to tmp then rename.
    """
    try:
        from ingestion.io import write_parquet_atomic  # type: ignore
        return write_parquet_atomic
    except Exception:
        def _fallback_write_parquet_atomic(df: pd.DataFrame, out_path: str | Path) -> None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.parent / f".tmp.{out_path.name}"
            df.to_parquet(tmp, index=False)
            tmp.replace(out_path)
        return _fallback_write_parquet_atomic


def atomic_dir_tmp(final_dir: Path) -> Path:
    """
    Create a temp sibling directory for atomic dataset writes.
    Caller writes into returned tmp dir, then calls atomic_dir_commit().
    """
    final_dir = Path(final_dir)
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_dir.parent / f".tmp.{final_dir.name}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    return tmp


def atomic_dir_commit(tmp_dir: Path, final_dir: Path) -> None:
    """
    Atomically replace final_dir with tmp_dir (best-effort).
    """
    tmp_dir = Path(tmp_dir)
    final_dir = Path(final_dir)

    if final_dir.exists():
        shutil.rmtree(final_dir)

    tmp_dir.replace(final_dir)
