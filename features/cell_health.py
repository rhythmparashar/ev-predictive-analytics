# features/cell_health.py
"""
Cell-level health features — per timestamp.

Computes for every cell in every module:
  - deviation from module mean        (M1_C1_delta ... M5_C36_delta)
  - z-score within module             (M1_C1_zscore ... M5_C36_zscore)
  - is_outlier binary flag            (M1_C1_is_outlier ... M5_C36_is_outlier)

Input:  silver_cells dataframe (one row per timestamp)
Output: cell_features dataframe (same rows, feature columns added)

Does NOT modify silver_cells.
Called by cell_pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# -------------------------------------------------
# YAML loader
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Column helpers
# -------------------------------------------------

def voltage_columns_for_module(m: int, cells_per_module: int) -> list[str]:
    return [f"M{m}_C{c}" for c in range(1, cells_per_module + 1)]


def all_voltage_columns(
    modules: list[int],
    cells_per_module: int,
) -> dict[int, list[str]]:
    """
    Returns dict: module_number → list of cell voltage column names.
    e.g. {1: ['M1_C1', ..., 'M1_C36'], 2: [...], ...}
    """
    return {
        m: voltage_columns_for_module(m, cells_per_module)
        for m in modules
    }


# -------------------------------------------------
# Per-module cell features
# -------------------------------------------------

def compute_module_cell_features(
    df: pd.DataFrame,
    cell_cols: list[str],
    outlier_zscore_threshold: float,
) -> pd.DataFrame:
    """
    For one module, compute delta, zscore, is_outlier per cell.

    Parameters
    ----------
    df                       : silver_cells dataframe
    cell_cols                : list of cell voltage column names for this module
    outlier_zscore_threshold : z-score above which cell is flagged as outlier

    Returns
    -------
    DataFrame with only the new feature columns (same index as df).
    """

    present_cols = [c for c in cell_cols if c in df.columns]

    if not present_cols:
        return pd.DataFrame(index=df.index)

    module_data = df[present_cols].astype(float)

    # Module mean and std per row (axis=1 = across cells)
    module_mean = module_data.mean(axis=1)
    module_std = module_data.std(axis=1, ddof=1)

    # Avoid division by zero when all cells identical
    module_std_safe = module_std.replace(0, np.nan)

    feature_map: dict[str, pd.Series] = {}

    for col in present_cols:
        cell_v = module_data[col]

        # Delta from module mean
        delta = cell_v - module_mean
        feature_map[f"{col}_delta"] = delta

        # Z-score within module
        zscore = delta / module_std_safe
        feature_map[f"{col}_zscore"] = zscore

        # Outlier flag
        feature_map[f"{col}_is_outlier"] = (
            zscore.abs() > outlier_zscore_threshold
        ).astype("int8")

    return pd.DataFrame(feature_map, index=df.index)


# -------------------------------------------------
# Main entry point
# -------------------------------------------------

def compute_cell_features(
    silver: pd.DataFrame,
    voltage_schema_path: Path,
    cell_cfg_path: Path,
) -> pd.DataFrame:
    """
    Compute per-cell deviation, z-score, and outlier features
    for all modules.

    Parameters
    ----------
    silver              : already-filtered silver_cells dataframe
    voltage_schema_path : schema/cell_voltage_schema.yaml
    cell_cfg_path       : configs/cell.yaml

    Returns
    -------
    cell_features dataframe:
        timestamp, vehicle_id, cell_quality_flag,
        M1_C1_delta, M1_C1_zscore, M1_C1_is_outlier,
        ... (repeated for all 180 cells)
    """

    v_schema = load_yaml(voltage_schema_path)
    cfg = load_yaml(cell_cfg_path)

    modules = v_schema["modules"]
    cells_per_module = v_schema["cells_per_module"]
    outlier_zscore = float(cfg["voltage"]["outlier_zscore"])

    module_cols = all_voltage_columns(modules, cells_per_module)

    df = silver.copy()

    identity_cols = ["timestamp", "vehicle_id", "cell_quality_flag"]
    identity_cols = [c for c in identity_cols if c in df.columns]

    if df.empty:
        return pd.DataFrame(columns=identity_cols)

    # -------------------------------------------------
    # Compute per-module cell features
    # -------------------------------------------------

    feature_parts = []

    for m in modules:
        cell_cols = module_cols[m]
        module_features = compute_module_cell_features(
            df=df,
            cell_cols=cell_cols,
            outlier_zscore_threshold=outlier_zscore,
        )
        feature_parts.append(module_features)

    # Concatenate all module features horizontally
    all_features = pd.concat(feature_parts, axis=1)

    # -------------------------------------------------
    # Assemble output
    # -------------------------------------------------

    out = pd.concat(
        [
            df[identity_cols].reset_index(drop=True),
            all_features.reset_index(drop=True),
        ],
        axis=1,
    )

    out = out.sort_values("timestamp").reset_index(drop=True)

    return out