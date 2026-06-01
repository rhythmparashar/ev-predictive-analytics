"""
Battery Health Score computation.

Reads a master dataset DataFrame and returns it with added columns:
  thermal_penalty, voltage_imbalance_penalty, temperature_imbalance_penalty,
  current_stress_penalty, weak_module_penalty, data_quality_penalty, fault_penalty,
  battery_health_score, battery_health_status,
  voltage_imbalance_mv, temp_imbalance_c,
  weakest_module_by_voltage, weakest_module_by_temp,
  top_issue, recommended_action, business_impact

All penalty computations are vectorized (no Python loops over rows).
Voltage range columns in the master dataset are in VOLTS — multiplied ×1000 for mV comparisons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from battery_intelligence.config import (
    CELL_FLAG,
    HEALTH_STATUS,
    MODULES,
    PENALTY_VALUES as PV,
    THRESHOLDS as TH,
)
from battery_intelligence.recommendations import assign_top_issue_and_recommendation


def _thermal_penalty(df: pd.DataFrame) -> np.ndarray:
    """Penalty from max battery temperature."""
    if "max_battery_temp_c" in df.columns:
        temp = df["max_battery_temp_c"].fillna(0.0).values
    elif "avg_battery_temp_c" in df.columns:
        temp = df["avg_battery_temp_c"].fillna(0.0).values
    else:
        return np.zeros(len(df))

    t = TH["thermal"]
    return np.select(
        [temp >= t["critical"], temp >= t["high"], temp >= t["warn"]],
        [PV["thermal_critical"], PV["thermal_high"], PV["thermal_warn"]],
        default=0.0,
    )


def _voltage_imbalance_penalty(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Penalty from pack voltage imbalance. Returns (penalty, imbalance_mv)."""
    if "pack_voltage_range" in df.columns:
        imbalance_mv = df["pack_voltage_range"].fillna(0.0).values * 1000.0
    elif "cell_spread_v" in df.columns:
        imbalance_mv = df["cell_spread_v"].fillna(0.0).values * 1000.0
    else:
        return np.zeros(len(df)), np.zeros(len(df))

    t = TH["voltage_imbalance_mv"]
    penalty = np.select(
        [imbalance_mv >= t["high"], imbalance_mv >= t["medium"], imbalance_mv >= t["low"]],
        [PV["voltage_imbalance_high"], PV["voltage_imbalance_medium"], PV["voltage_imbalance_low"]],
        default=0.0,
    )
    return penalty, imbalance_mv


def _temp_imbalance_penalty(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Penalty from thermal gradient across battery pack. Returns (penalty, temp_imbalance_c)."""
    if "pack_temp_range" in df.columns:
        temp_diff = df["pack_temp_range"].fillna(0.0).values
    elif "max_battery_temp_c" in df.columns and "min_battery_temp_c" in df.columns:
        temp_diff = (
            df["max_battery_temp_c"].fillna(0.0) - df["min_battery_temp_c"].fillna(0.0)
        ).values
    else:
        return np.zeros(len(df)), np.zeros(len(df))

    t = TH["temp_imbalance"]
    penalty = np.select(
        [temp_diff >= t["high"], temp_diff >= t["medium"], temp_diff >= t["low"]],
        [PV["temp_imbalance_high"], PV["temp_imbalance_medium"], PV["temp_imbalance_low"]],
        default=0.0,
    )
    return penalty, temp_diff


def _current_stress_penalty(df: pd.DataFrame) -> np.ndarray:
    """Penalty from high discharge/charge current."""
    if "abs_current_a" in df.columns:
        current = df["abs_current_a"].fillna(0.0).values
    elif "battery_current_a" in df.columns:
        current = df["battery_current_a"].abs().fillna(0.0).values
    else:
        return np.zeros(len(df))

    t = TH["current_stress"]
    return np.select(
        [current >= t["high"], current >= t["medium"], current >= t["low"]],
        [PV["current_high"], PV["current_medium"], PV["current_low"]],
        default=0.0,
    )


def _weak_module_penalty(df: pd.DataFrame) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Penalty when one module is significantly worse than others.
    Returns (penalty, weakest_module_by_voltage_id, weakest_module_by_temp_id).
    """
    v_cols = [f"module_{i}_voltage_range" for i in MODULES if f"module_{i}_voltage_range" in df.columns]
    t_cols = [f"module_{i}_temp_range" for i in MODULES if f"module_{i}_temp_range" in df.columns]

    tw = TH["weak_module_voltage_mv"]
    tt = TH["weak_module_temp"]

    if v_cols:
        v_filled = df[v_cols].fillna(0.0)
        max_v_mv = v_filled.max(axis=1) * 1000.0
        v_pen = np.select(
            [max_v_mv > tw["alert"], max_v_mv > tw["warn"]],
            [PV["weak_module_voltage_alert"], PV["weak_module_voltage_warn"]],
            default=0.0,
        )
        weakest_v = (
            v_filled.idxmax(axis=1)
            .str.extract(r"module_(\d+)_voltage_range", expand=False)
            .rename("weakest_module_by_voltage")
        )
    else:
        v_pen = np.zeros(len(df))
        weakest_v = pd.Series("unknown", index=df.index, name="weakest_module_by_voltage")

    if t_cols:
        t_filled = df[t_cols].fillna(0.0)
        max_t = t_filled.max(axis=1)
        t_pen = np.select(
            [max_t > tt["alert"], max_t > tt["warn"]],
            [PV["weak_module_temp_alert"], PV["weak_module_temp_warn"]],
            default=0.0,
        )
        weakest_t = (
            t_filled.idxmax(axis=1)
            .str.extract(r"module_(\d+)_temp_range", expand=False)
            .rename("weakest_module_by_temp")
        )
    else:
        t_pen = np.zeros(len(df))
        weakest_t = pd.Series("unknown", index=df.index, name="weakest_module_by_temp")

    combined = np.minimum(v_pen + t_pen, PV["weak_module_cap"])
    return combined, weakest_v, weakest_t


def _data_quality_penalty(df: pd.DataFrame) -> np.ndarray:
    """Penalty from data quality flags."""
    penalty = np.zeros(len(df))

    if "is_clean" in df.columns:
        not_clean = (~df["is_clean"].fillna(True)).values
        penalty = np.where(not_clean, penalty + PV["data_quality_not_clean"], penalty)

    if "cell_quality_flag" in df.columns:
        cqf = df["cell_quality_flag"].fillna(0).astype(int).values
        penalty = np.where(cqf & CELL_FLAG["PACK_GAP"], penalty + PV["data_quality_pack_gap"], penalty)
        penalty = np.where(cqf & CELL_FLAG["CELL_HARD_BREACH"], penalty + PV["data_quality_hard_breach"], penalty)
        penalty = np.where(cqf & CELL_FLAG["MODULE_GAP"], penalty + PV["data_quality_module_gap"], penalty)
        penalty = np.where(cqf & CELL_FLAG["CELL_SOFT_BREACH"], penalty + PV["data_quality_soft_breach"], penalty)
        penalty = np.where(cqf & CELL_FLAG["CELL_OUTLIER"], penalty + PV["data_quality_outlier"], penalty)

    if "vcu_quality_flag" in df.columns:
        vqf = df["vcu_quality_flag"].fillna(0).astype(int).values
        penalty = np.where(vqf != 0, penalty + PV["data_quality_vcu_flag"], penalty)

    return penalty


def _fault_penalty(df: pd.DataFrame) -> np.ndarray:
    """Penalty when any fault is active."""
    if "fault_any" not in df.columns:
        return np.zeros(len(df))
    fault = df["fault_any"].fillna(0).astype(int).values
    return np.where(fault != 0, PV["fault"], 0.0)


def _score_to_status(score: np.ndarray) -> np.ndarray:
    """Map numeric score to health status string."""
    return np.select(
        [
            score >= HEALTH_STATUS["EXCELLENT"],
            score >= HEALTH_STATUS["HEALTHY"],
            score >= HEALTH_STATUS["WARNING"],
            score >= HEALTH_STATUS["CRITICAL"],
        ],
        ["EXCELLENT", "HEALTHY", "WARNING", "CRITICAL"],
        default="SEVERE",
    )


def compute_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add battery health score and all penalty columns to df.

    Parameters
    ----------
    df : pd.DataFrame
        Master dataset (one row per timestamp per vehicle).

    Returns
    -------
    pd.DataFrame
        Input df with added columns for penalties, score, status, and diagnostics.
    """
    out = df.copy()

    thermal_pen = _thermal_penalty(out)
    vi_pen, imbalance_mv = _voltage_imbalance_penalty(out)
    ti_pen, temp_imbalance_c = _temp_imbalance_penalty(out)
    cs_pen = _current_stress_penalty(out)
    wm_pen, weakest_v, weakest_t = _weak_module_penalty(out)
    dq_pen = _data_quality_penalty(out)
    fa_pen = _fault_penalty(out)

    out["thermal_penalty"] = thermal_pen.astype("int16")
    out["voltage_imbalance_penalty"] = vi_pen.astype("int16")
    out["temperature_imbalance_penalty"] = ti_pen.astype("int16")
    out["current_stress_penalty"] = cs_pen.astype("int16")
    out["weak_module_penalty"] = wm_pen.astype("int16")
    out["data_quality_penalty"] = dq_pen.astype("int16")
    out["fault_penalty"] = fa_pen.astype("int16")

    out["voltage_imbalance_mv"] = imbalance_mv.round(2)
    out["temp_imbalance_c"] = temp_imbalance_c.round(2)
    out["weakest_module_by_voltage"] = weakest_v.values
    out["weakest_module_by_temp"] = weakest_t.values

    raw_score = (
        100
        - thermal_pen
        - vi_pen
        - ti_pen
        - cs_pen
        - wm_pen
        - dq_pen
        - fa_pen
    )
    score = np.clip(raw_score, 0, 100).astype("float32")

    # NO_DATA: rows where no penalty source had any usable column
    has_data = (
        ("max_battery_temp_c" in df.columns or "avg_battery_temp_c" in df.columns)
        or ("pack_voltage_range" in df.columns or "cell_spread_v" in df.columns)
        or ("abs_current_a" in df.columns)
    )
    if not has_data:
        score[:] = np.nan

    out["battery_health_score"] = score
    status = _score_to_status(score)
    if not has_data:
        status[:] = "NO_DATA"
    out["battery_health_status"] = status

    out = assign_top_issue_and_recommendation(out)

    return out
