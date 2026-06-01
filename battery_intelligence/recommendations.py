"""
Top-issue classification and recommendation mapping.
Called from health_score.py after all penalties are computed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Priority order: earlier = higher priority for top_issue selection
_ISSUE_RULES: list[tuple[str, str, str, str]] = [
    # (condition_key, top_issue, recommended_action, business_impact)
    (
        "fault_critical",
        "Active battery fault detected",
        "Prioritise immediate service inspection. Take vehicle offline if fault persists.",
        "High — fault-driven failures risk unplanned downtime and warranty claims.",
    ),
    (
        "thermal_critical",
        "Battery temperature critically high",
        "Reduce load immediately and inspect cooling system. Allow battery to cool before next trip.",
        "High — thermal runaway risk if unaddressed. May trigger BMS protection shutdown.",
    ),
    (
        "voltage_imbalance_critical",
        "Critical cell voltage imbalance",
        "Schedule battery balancing service. Inspect weakest cell group and BMS balancing circuit.",
        "High — severe imbalance reduces usable capacity and accelerates cell degradation.",
    ),
    (
        "temp_imbalance_critical",
        "Critical thermal gradient across modules",
        "Inspect module-level cooling paths. Check for blocked vents or failed thermal interface.",
        "High — persistent hot-spot accelerates localised degradation and reduces pack life.",
    ),
    (
        "thermal_high",
        "Battery temperature elevated",
        "Reduce vehicle load. Monitor cooling system performance and ambient temperature.",
        "Medium — continued high temperature accelerates electrolyte degradation.",
    ),
    (
        "current_extreme",
        "Extreme current stress on battery",
        "Review operator driving pattern and vehicle load. Implement current limiting if available.",
        "High — sustained extreme current reduces cycle life and may trigger overcurrent faults.",
    ),
    (
        "temp_and_current",
        "High current combined with elevated temperature",
        "Reduce load and monitor temperature. Both stressors together significantly accelerate wear.",
        "High — combined thermal and electrical stress is the most damaging operating condition.",
    ),
    (
        "voltage_imbalance_warning",
        "Cell voltage imbalance increasing",
        "Monitor trend over next 3 days. Schedule balancing if imbalance continues to rise.",
        "Medium — early imbalance reduces SOC estimation accuracy and effective capacity.",
    ),
    (
        "temp_imbalance_warning",
        "Thermal gradient detected across modules",
        "Inspect module cooling path and thermal sensor readings. Review hottest module.",
        "Medium — thermal gradient signals uneven ageing if sustained over multiple days.",
    ),
    (
        "current_high",
        "High current stress detected",
        "Review vehicle load and operator driving pattern.",
        "Medium — reduces cycle life if high-current operation is habitual.",
    ),
    (
        "weak_module",
        "Weak battery module detected",
        "Flag module for next service. Inspect lowest cell voltage and hottest sensor in that module.",
        "Medium — weak module limits overall pack capacity and may trigger premature BMS cutoff.",
    ),
    (
        "data_quality",
        "BMS or cell data quality issue",
        "Check BMS and CAN communication. Inspect cell wiring harness and sensor connections.",
        "Low-Medium — data gaps prevent accurate health monitoring and may mask real issues.",
    ),
    (
        "thermal_warn",
        "Battery temperature slightly elevated",
        "Monitor cooling system. No immediate action required.",
        "Low — mild temperature rise within acceptable range.",
    ),
    (
        "normal",
        "Battery operating normally",
        "Continue normal monitoring. No action required.",
        "None — battery is within all healthy operating thresholds.",
    ),
]

_ISSUE_MAP: dict[str, dict[str, str]] = {
    row[0]: {"top_issue": row[1], "recommended_action": row[2], "business_impact": row[3]}
    for row in _ISSUE_RULES
}


def _classify_condition(df: pd.DataFrame) -> np.ndarray:
    """Return condition key array (object dtype) for each row using vectorised priority logic."""
    n = len(df)

    fa = df["fault_penalty"].values if "fault_penalty" in df.columns else np.zeros(n)
    tp = df["thermal_penalty"].values if "thermal_penalty" in df.columns else np.zeros(n)
    vi = df["voltage_imbalance_penalty"].values if "voltage_imbalance_penalty" in df.columns else np.zeros(n)
    ti = df["temperature_imbalance_penalty"].values if "temperature_imbalance_penalty" in df.columns else np.zeros(n)
    cs = df["current_stress_penalty"].values if "current_stress_penalty" in df.columns else np.zeros(n)
    wm = df["weak_module_penalty"].values if "weak_module_penalty" in df.columns else np.zeros(n)
    dq = df["data_quality_penalty"].values if "data_quality_penalty" in df.columns else np.zeros(n)

    abs_current = (
        df["abs_current_a"].fillna(0.0).values
        if "abs_current_a" in df.columns
        else np.zeros(n)
    )
    max_temp = (
        df["max_battery_temp_c"].fillna(0.0).values
        if "max_battery_temp_c" in df.columns
        else np.zeros(n)
    )

    conditions = [
        fa >= 25,                                          # fault_critical
        tp >= 35,                                          # thermal_critical
        vi >= 35,                                          # voltage_imbalance_critical
        ti >= 35,                                          # temp_imbalance_critical
        tp >= 20,                                          # thermal_high
        cs >= 25,                                          # current_extreme
        (abs_current >= 80) & (max_temp >= 43),           # temp_and_current
        vi >= 20,                                          # voltage_imbalance_warning
        ti >= 20,                                          # temp_imbalance_warning
        cs >= 15,                                          # current_high
        wm > 0,                                            # weak_module
        dq > 0,                                            # data_quality
        tp > 0,                                            # thermal_warn
    ]
    keys = [
        "fault_critical",
        "thermal_critical",
        "voltage_imbalance_critical",
        "temp_imbalance_critical",
        "thermal_high",
        "current_extreme",
        "temp_and_current",
        "voltage_imbalance_warning",
        "temp_imbalance_warning",
        "current_high",
        "weak_module",
        "data_quality",
        "thermal_warn",
    ]
    return np.select(conditions, keys, default="normal")


def assign_top_issue_and_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add top_issue, recommended_action, business_impact columns to df.
    Called after all penalty columns have been added.
    """
    keys = _classify_condition(df)

    df = df.copy()
    df["top_issue"] = [_ISSUE_MAP[k]["top_issue"] for k in keys]
    df["recommended_action"] = [_ISSUE_MAP[k]["recommended_action"] for k in keys]
    df["business_impact"] = [_ISSUE_MAP[k]["business_impact"] for k in keys]
    return df
