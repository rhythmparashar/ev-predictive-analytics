# ingestion/cell_validators.py
"""
Cell-specific validation for raw cell voltage and temperature CSVs.

Responsibilities:
1) Schema validation — expected columns present
2) Timestamp parsing — to UTC datetime
3) Null classification — CELL_FFILLED / MODULE_GAP / PACK_GAP per row
4) Forward fill — where null rules allow
5) Range checks — soft and hard bounds per signal type
6) Flatline detection — cell stuck at constant value
7) Returns CellValidationResult(df, report)

Does NOT modify silver_cells directly.
Called by cell_ingest.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


# -------------------------------------------------
# Output container
# -------------------------------------------------

@dataclass
class CellValidationResult:
    df: pd.DataFrame
    report: dict


# -------------------------------------------------
# YAML loader
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Column generation helpers
# -------------------------------------------------

def voltage_columns(schema: dict) -> list[str]:
    """Generate all M{m}_C{c} column names from schema."""
    cols = []
    pattern = schema["column_pattern"]
    for m in schema["modules"]:
        for c in range(1, schema["cells_per_module"] + 1):
            cols.append(pattern.format(m=m, c=c))
    return cols


def temp_columns(schema: dict) -> list[str]:
    """Generate all M{m}_T{s} column names from schema."""
    cols = []
    pattern = schema["column_pattern"]
    for m in schema["modules"]:
        for s in range(1, schema["sensors_per_module"] + 1):
            cols.append(pattern.format(m=m, s=s))
    return cols


def module_columns(all_cols: list[str], module: int) -> list[str]:
    """Filter columns belonging to a specific module number."""
    prefix = f"M{module}_"
    return [c for c in all_cols if c.startswith(prefix)]


# -------------------------------------------------
# Main validator
# -------------------------------------------------

def validate_cells(
    df: pd.DataFrame,
    data_type: str,                  # "voltage" or "temp"
    voltage_schema_path: Path,
    temp_schema_path: Path,
    cell_quality_flags_path: Path,
    cell_cfg_path: Path,
    vehicle_id: str,
) -> CellValidationResult:
    """
    Validate raw cell voltage or temp dataframe.

    Parameters
    ----------
    df          : raw dataframe as loaded from CSV
    data_type   : "voltage" or "temp"
    """

    assert data_type in ("voltage", "temp"), \
        f"data_type must be 'voltage' or 'temp', got {data_type!r}"

    v_schema = load_yaml(voltage_schema_path)
    t_schema = load_yaml(temp_schema_path)
    qf = load_yaml(cell_quality_flags_path)
    cfg = load_yaml(cell_cfg_path)

    schema = v_schema if data_type == "voltage" else t_schema
    null_cfg = cfg["null_handling"]

    # Select expected columns for this data type
    if data_type == "voltage":
        signal_cols = voltage_columns(v_schema)
        ranges = v_schema["ranges"]
        flatline_w = int(v_schema["flatline_window_s"])
    else:
        signal_cols = temp_columns(t_schema)
        ranges = t_schema["ranges"]
        flatline_w = int(t_schema["flatline_window_s"])

    ts_col = schema["timestamp_col"]   # "recorded_at"

    # -------------------------------------------------
    # 1) Timestamp parsing
    # -------------------------------------------------

    if ts_col not in df.columns:
        raise ValueError(
            f"cell_validators: missing timestamp column '{ts_col}'"
        )

    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    bad_ts = df[ts_col].isna()
    if bad_ts.any():
        # Drop rows with unparseable timestamps — cannot join to silver
        df = df[~bad_ts].copy()

    # Rename to canonical 'timestamp'
    df = df.rename(columns={ts_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -------------------------------------------------
    # 2) Add vehicle_id and cell_quality_flag
    # -------------------------------------------------

    df["vehicle_id"] = vehicle_id
    df["cell_quality_flag"] = 0

    # -------------------------------------------------
    # 3) Keep only expected signal columns
    #    (drop any extra columns from CSV)
    # -------------------------------------------------

    present_signal_cols = [c for c in signal_cols if c in df.columns]
    missing_signal_cols = [c for c in signal_cols if c not in df.columns]

    # Add missing columns as NaN so downstream code has consistent shape
    for c in missing_signal_cols:
        df[c] = pd.NA

    # Convert all signal columns to numeric
    for c in present_signal_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------------------------
    # 4) Null classification + fill per row
    # -------------------------------------------------

    dropout_max = int(null_cfg["cell_group_dropout_max"])   # 8
    module_gap_min = int(null_cfg["module_gap_min"])        # 9
    pack_gap_thr = int(null_cfg["pack_gap_threshold"])      # 170

    null_counts = df[signal_cols].isnull().sum(axis=1)

    # Precompute full forward fill once — used by both MODULE_GAP and CELL_FFILLED
    # This ensures fills can see previous valid rows outside the masked subset
    filled = df[signal_cols].ffill()

    # PACK_GAP — no fill, flag only
    pack_gap_mask = null_counts >= pack_gap_thr
    df.loc[pack_gap_mask, "cell_quality_flag"] |= int(qf["PACK_GAP"])

    # MODULE_GAP — partial module dropout, fill where possible using precomputed ffill
    module_gap_mask = (
        (null_counts >= module_gap_min) &
        (null_counts < pack_gap_thr)
    )
    if module_gap_mask.any():
        df.loc[module_gap_mask, signal_cols] = (
            df.loc[module_gap_mask, signal_cols]
            .fillna(filled.loc[module_gap_mask])
        )
        df.loc[module_gap_mask, "cell_quality_flag"] |= int(qf["MODULE_GAP"])

    # CELL_FFILLED — single cell group dropout, fill using precomputed ffill
    cell_ffill_mask = (null_counts > 0) & (null_counts <= dropout_max)
    if cell_ffill_mask.any():
        df.loc[cell_ffill_mask, signal_cols] = (
            df.loc[cell_ffill_mask, signal_cols]
            .fillna(filled.loc[cell_ffill_mask])
        )
        df.loc[cell_ffill_mask, "cell_quality_flag"] |= int(qf["CELL_FFILLED"])

    # -------------------------------------------------
    # 5) Range checks
    # -------------------------------------------------

    soft_lo, soft_hi = ranges["soft"]
    hard_lo, hard_hi = ranges["hard"]

    soft_breaches = 0
    hard_breaches = 0
    soft_by_col = {}
    hard_by_col = {}

    for col in signal_cols:
        s = df[col]

        # Soft
        soft_mask = (s < soft_lo) | (s > soft_hi)
        cnt = int(soft_mask.sum())
        if cnt:
            soft_breaches += cnt
            soft_by_col[col] = cnt
            df.loc[soft_mask, "cell_quality_flag"] |= int(qf["CELL_SOFT_BREACH"])

        # Hard
        hard_mask = (s < hard_lo) | (s > hard_hi)
        cnt = int(hard_mask.sum())
        if cnt:
            hard_breaches += cnt
            hard_by_col[col] = cnt
            df.loc[hard_mask, "cell_quality_flag"] |= int(qf["CELL_HARD_BREACH"])

    # -------------------------------------------------
    # 6) Flatline detection
    #    Flag cells where value unchanged for flatline_w consecutive rows
    #
    #    NOTE: flatline_w is treated as row count assuming ~1Hz sampling.
    #    If sampling rate changes, convert to time-based window.
    #
    #    Vectorised across all signal columns simultaneously.
    # -------------------------------------------------

    # Forward fill before std computation so NaNs don't break windows
    filled_for_flatline = df[signal_cols].ffill()

    # Compute rolling std across all columns at once
    rolling_std_matrix = filled_for_flatline.rolling(
        window=flatline_w,
        min_periods=flatline_w
    ).std()

    # Any column with std == 0 at this row is flatlined
    flatline_matrix = rolling_std_matrix == 0

    # Row-level flag — True if any cell is flatlined at this timestamp
    flatline_any_row = flatline_matrix.any(axis=1)

    # Count of flatlined cell-timestamps
    flatline_count = int(flatline_matrix.fillna(False).values.sum())

    # Which columns had any flatline at all
    flatline_cols = [
        c for c in signal_cols
        if flatline_matrix[c].fillna(False).any()
    ]

    if flatline_any_row.any():
        df.loc[flatline_any_row, "cell_quality_flag"] |= int(qf["CELL_FLATLINE"])

    # -------------------------------------------------
    # 7) Build report
    # -------------------------------------------------

    total_null_rows = int((null_counts > 0).sum())
    pack_gap_rows = int(pack_gap_mask.sum())
    module_gap_rows = int(module_gap_mask.sum())
    cell_ffill_rows = int(cell_ffill_mask.sum())

    report = {
        "vehicle_id": vehicle_id,
        "data_type": data_type,
        "row_count": int(len(df)),
        "bad_timestamps_dropped": int(bad_ts.sum()),
        "missing_signal_cols": missing_signal_cols,
        "null_rows": {
            "total": total_null_rows,
            "pack_gap": pack_gap_rows,
            "module_gap": module_gap_rows,
            "cell_ffill": cell_ffill_rows,
        },
        "range_breaches": {
            "soft_total": soft_breaches,
            "hard_total": hard_breaches,
            "soft_top10": dict(
                sorted(soft_by_col.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "hard_top10": dict(
                sorted(hard_by_col.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
        },
        "flatline": {
            "total_flagged_rows": flatline_count,
            "affected_cols": flatline_cols[:20],
        },
        "null_rates": {
            c: float(df[c].isna().mean())
            for c in signal_cols
        },
    }

    # Final column order: timestamp, vehicle_id, signals, cell_quality_flag
    out_cols = (
        ["timestamp", "vehicle_id"]
        + signal_cols
        + ["cell_quality_flag"]
    )
    df = df[out_cols]

    return CellValidationResult(df=df, report=report)