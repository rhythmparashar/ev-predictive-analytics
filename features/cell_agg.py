# features/cell_agg.py
"""
Module and pack level aggregations — per timestamp.

Computes from silver_cells:

Module features (per module, per timestamp):
  module_{m}_voltage_mean       mean cell voltage across module
  module_{m}_voltage_std        spread within module
  module_{m}_voltage_range      max - min within module
  module_{m}_lowest_cell        column name of lowest voltage cell
  module_{m}_temp_mean          mean temp across module sensors
  module_{m}_temp_range         max - min temp within module
  module_{m}_hottest_sensor     column name of hottest temp sensor

Pack features (per timestamp):
  pack_voltage_range            max - min across all 180 cells
  pack_voltage_std              std across all 180 cells
  pack_temp_range               max - min across all 90 sensors
  worst_cell_id                 column name of lowest voltage cell in pack
  hottest_module_id             module number with highest mean temp
  imbalance_flag                1 if pack_voltage_range > threshold

Input:  silver_cells dataframe (already quality filtered)
Output: module_features dataframe, pack_features dataframe

Does NOT modify silver_cells.
Called by cell_pipeline.py.
"""

from __future__ import annotations

from pathlib import Path

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

def voltage_cols_for_module(m: int, cells_per_module: int) -> list[str]:
    return [f"M{m}_C{c}" for c in range(1, cells_per_module + 1)]


def temp_cols_for_module(m: int, sensors_per_module: int) -> list[str]:
    return [f"M{m}_T{s}" for s in range(1, sensors_per_module + 1)]


# -------------------------------------------------
# Module features
# -------------------------------------------------

def compute_module_features(
    df: pd.DataFrame,
    modules: list[int],
    cells_per_module: int,
    sensors_per_module: int,
) -> pd.DataFrame:
    """
    Compute per-module voltage and temperature aggregations.

    Returns dataframe with same index as df.
    """

    out = pd.DataFrame(index=df.index)

    for m in modules:
        v_cols = [c for c in voltage_cols_for_module(m, cells_per_module) if c in df.columns]
        t_cols = [c for c in temp_cols_for_module(m, sensors_per_module) if c in df.columns]

        # -------------------------------------------------
        # Voltage aggregations
        # -------------------------------------------------

        if v_cols:
            v_data = df[v_cols].astype(float)

            out[f"module_{m}_voltage_mean"] = v_data.mean(axis=1)
            out[f"module_{m}_voltage_std"] = v_data.std(axis=1, ddof=1)
            out[f"module_{m}_voltage_range"] = v_data.max(axis=1) - v_data.min(axis=1)

            lowest_cell = v_data.idxmin(axis=1)
            no_valid_v = v_data.notna().sum(axis=1) == 0
            lowest_cell = lowest_cell.where(~no_valid_v, pd.NA)
            out[f"module_{m}_lowest_cell"] = lowest_cell

        # -------------------------------------------------
        # Temperature aggregations
        # -------------------------------------------------

        if t_cols:
            t_data = df[t_cols].astype(float)

            out[f"module_{m}_temp_mean"] = t_data.mean(axis=1)
            out[f"module_{m}_temp_range"] = t_data.max(axis=1) - t_data.min(axis=1)

            hottest_sensor = t_data.idxmax(axis=1)
            no_valid_t = t_data.notna().sum(axis=1) == 0
            hottest_sensor = hottest_sensor.where(~no_valid_t, pd.NA)
            out[f"module_{m}_hottest_sensor"] = hottest_sensor

    return out


# -------------------------------------------------
# Pack features
# -------------------------------------------------

def compute_pack_features(
    df: pd.DataFrame,
    module_features: pd.DataFrame,
    modules: list[int],
    cells_per_module: int,
    sensors_per_module: int,
    imbalance_alert_mv: float,
) -> pd.DataFrame:
    """
    Compute pack-level aggregations.

    Returns dataframe with same index as df.
    """

    out = pd.DataFrame(index=df.index)

    # All voltage columns present in df
    all_v_cols = [
        f"M{m}_C{c}"
        for m in modules
        for c in range(1, cells_per_module + 1)
        if f"M{m}_C{c}" in df.columns
    ]

    if all_v_cols:
        v_data = df[all_v_cols].astype(float)

        out["pack_voltage_range"] = v_data.max(axis=1) - v_data.min(axis=1)
        out["pack_voltage_std"] = v_data.std(axis=1, ddof=1)

        worst_cell = v_data.idxmin(axis=1)
        no_valid_v = v_data.notna().sum(axis=1) == 0
        worst_cell = worst_cell.where(~no_valid_v, pd.NA)
        out["worst_cell_id"] = worst_cell

        # Imbalance flag — pack_voltage_range > alert threshold (convert mV → V)
        alert_v = imbalance_alert_mv / 1000.0
        out["imbalance_flag"] = (out["pack_voltage_range"] > alert_v).astype("int8")

    # All temperature columns present in df
    all_t_cols = [
        f"M{m}_T{s}"
        for m in modules
        for s in range(1, sensors_per_module + 1)
        if f"M{m}_T{s}" in df.columns
    ]

    if all_t_cols:
        t_data = df[all_t_cols].astype(float)
        out["pack_temp_range"] = t_data.max(axis=1) - t_data.min(axis=1)

    # Hottest module from module temp means
    module_temp_mean_cols = [
        f"module_{m}_temp_mean"
        for m in modules
        if f"module_{m}_temp_mean" in module_features.columns
    ]

    if module_temp_mean_cols:
        temp_means = module_features[module_temp_mean_cols]
        hottest_col = temp_means.idxmax(axis=1)
        no_valid_tm = temp_means.notna().sum(axis=1) == 0
        hottest_module_id = hottest_col.str.extract(r"module_(\d+)_temp_mean")[0].astype("Int64")
        hottest_module_id = hottest_module_id.where(~no_valid_tm, pd.NA)
        out["hottest_module_id"] = hottest_module_id

    return out


# -------------------------------------------------
# Main entry point
# -------------------------------------------------

def compute_agg_features(
    silver: pd.DataFrame,
    voltage_schema_path: Path,
    temp_schema_path: Path,
    cell_cfg_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute module and pack aggregation features from silver_cells.

    Parameters
    ----------
    silver              : already quality-filtered silver_cells dataframe
    voltage_schema_path : schema/cell_voltage_schema.yaml
    temp_schema_path    : schema/cell_temp_schema.yaml
    cell_cfg_path       : configs/cell.yaml

    Returns
    -------
    (module_features_df, pack_features_df)
    Both have same row count as input silver.
    Both include timestamp + vehicle_id as identity columns.
    """

    v_schema = load_yaml(voltage_schema_path)
    t_schema = load_yaml(temp_schema_path)
    cfg = load_yaml(cell_cfg_path)

    modules = v_schema["modules"]
    cells_per_module = v_schema["cells_per_module"]
    sensors_per_module = t_schema["sensors_per_module"]
    imbalance_alert_mv = float(cfg["voltage"]["imbalance_alert_mv"])

    df = silver.copy()

    identity_cols = ["timestamp", "vehicle_id", "cell_quality_flag"]
    identity_cols = [c for c in identity_cols if c in df.columns]

    if df.empty:
        empty = pd.DataFrame(columns=identity_cols)
        return empty.copy(), empty.copy()

    # -------------------------------------------------
    # Module features
    # -------------------------------------------------

    module_feat = compute_module_features(
        df=df,
        modules=modules,
        cells_per_module=cells_per_module,
        sensors_per_module=sensors_per_module,
    )

    module_out = pd.concat(
        [
            df[identity_cols].reset_index(drop=True),
            module_feat.reset_index(drop=True),
        ],
        axis=1,
    ).sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------
    # Pack features
    # -------------------------------------------------

    pack_feat = compute_pack_features(
        df=df,
        module_features=module_feat,
        modules=modules,
        cells_per_module=cells_per_module,
        sensors_per_module=sensors_per_module,
        imbalance_alert_mv=imbalance_alert_mv,
    )

    pack_out = pd.concat(
        [
            df[identity_cols].reset_index(drop=True),
            pack_feat.reset_index(drop=True),
        ],
        axis=1,
    ).sort_values("timestamp").reset_index(drop=True)

    return module_out, pack_out