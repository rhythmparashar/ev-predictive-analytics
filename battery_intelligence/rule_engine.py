"""
Rule-based alert engine for Battery Intelligence Module 1.

evaluate_rules(df) → long-format DataFrame of triggered alerts.
Each row = one alert instance (one rule fired at one timestamp for one vehicle).
Only timestamps where at least one rule fires are included.

Rules:
  1  BATTERY_TEMP_WARNING
  2  BATTERY_TEMP_CRITICAL
  3  VOLTAGE_IMBALANCE_WARNING
  4  VOLTAGE_IMBALANCE_CRITICAL
  5  TEMP_IMBALANCE_WARNING
  6  TEMP_IMBALANCE_CRITICAL
  7  HIGH_CURRENT_STRESS
  8  HIGH_CURRENT_WITH_HIGH_TEMP
  9  WEAK_MODULE_VOLTAGE
  10 WEAK_MODULE_THERMAL
  11 BMS_OR_CELL_DATA_QUALITY_ISSUE
  12 BATTERY_FAULT_PRESENT
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from battery_intelligence.config import CELL_FLAG, MODULES, RULE_THRESHOLDS as RT


# ---------------------------------------------------------------------------
# Rule definitions
# Each rule is (rule_id, severity, title, description, possible_cause,
#               recommended_action, business_impact, condition_fn, evidence_fn)
# condition_fn: (df) -> bool array
# evidence_fn:  (df) -> str Series
# ---------------------------------------------------------------------------

def _safe(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(fill)
    return pd.Series(fill, index=df.index)


def _v_imbalance_mv(df: pd.DataFrame) -> pd.Series:
    if "pack_voltage_range" in df.columns:
        return _safe(df, "pack_voltage_range") * 1000.0
    return _safe(df, "cell_spread_v") * 1000.0


def _temp_diff(df: pd.DataFrame) -> pd.Series:
    if "pack_temp_range" in df.columns:
        return _safe(df, "pack_temp_range")
    return (_safe(df, "max_battery_temp_c") - _safe(df, "min_battery_temp_c", 0.0)).clip(lower=0)


_RULES: list[dict] = [
    {
        "rule_id": "BATTERY_TEMP_WARNING",
        "severity": "WARNING",
        "title": "Battery Temperature Warning",
        "description": "Battery temperature has exceeded the warning threshold.",
        "possible_cause": "High ambient temperature, sustained heavy load, or reduced cooling capacity.",
        "recommended_action": "Reduce vehicle load. Monitor cooling system. Allow battery to cool if possible.",
        "business_impact": "Medium — sustained high temperature accelerates electrolyte degradation.",
        "condition": lambda df: (
            (_safe(df, "max_battery_temp_c") >= RT["temp_warning_c"])
            & (_safe(df, "max_battery_temp_c") < RT["temp_critical_c"])
        ).values,
        "evidence": lambda df: (
            "max_battery_temp_c=" + _safe(df, "max_battery_temp_c").round(1).astype(str)
            + "°C (threshold=" + str(RT["temp_warning_c"]) + "°C)"
        ),
    },
    {
        "rule_id": "BATTERY_TEMP_CRITICAL",
        "severity": "CRITICAL",
        "title": "Battery Temperature Critical",
        "description": "Battery temperature has exceeded the critical threshold. Risk of thermal runaway.",
        "possible_cause": "Cooling system failure, extreme ambient temperature, or sustained high-power operation.",
        "recommended_action": "Reduce load immediately. Inspect cooling system. Take vehicle offline if temperature continues rising.",
        "business_impact": "High — thermal runaway risk. May trigger BMS protection shutdown and unplanned downtime.",
        "condition": lambda df: (
            _safe(df, "max_battery_temp_c") >= RT["temp_critical_c"]
        ).values,
        "evidence": lambda df: (
            "max_battery_temp_c=" + _safe(df, "max_battery_temp_c").round(1).astype(str)
            + "°C (threshold=" + str(RT["temp_critical_c"]) + "°C)"
        ),
    },
    {
        "rule_id": "VOLTAGE_IMBALANCE_WARNING",
        "severity": "WARNING",
        "title": "Cell Voltage Imbalance Warning",
        "description": "Pack voltage range has exceeded the warning threshold, indicating cell imbalance.",
        "possible_cause": "Uneven cell ageing, BMS balancing lag, or a weak cell group.",
        "recommended_action": "Monitor trend over next 3–5 days. Schedule balancing if imbalance persists.",
        "business_impact": "Medium — reduces SOC estimation accuracy and effective pack capacity.",
        "condition": lambda df: (
            (_v_imbalance_mv(df) >= RT["voltage_imbalance_warning_mv"])
            & (_v_imbalance_mv(df) < RT["voltage_imbalance_critical_mv"])
        ).values,
        "evidence": lambda df: (
            "voltage_imbalance=" + _v_imbalance_mv(df).round(1).astype(str)
            + "mV (threshold=" + str(RT["voltage_imbalance_warning_mv"]) + "mV)"
        ),
    },
    {
        "rule_id": "VOLTAGE_IMBALANCE_CRITICAL",
        "severity": "CRITICAL",
        "title": "Cell Voltage Imbalance Critical",
        "description": "Pack voltage range is critically high. Severe cell imbalance detected.",
        "possible_cause": "Degraded cell(s), BMS balancing failure, or connection resistance issue.",
        "recommended_action": "Schedule battery balancing service. Inspect weakest cell group and BMS balancing circuit.",
        "business_impact": "High — severe imbalance reduces usable capacity and accelerates cell degradation.",
        "condition": lambda df: (
            _v_imbalance_mv(df) >= RT["voltage_imbalance_critical_mv"]
        ).values,
        "evidence": lambda df: (
            "voltage_imbalance=" + _v_imbalance_mv(df).round(1).astype(str)
            + "mV (threshold=" + str(RT["voltage_imbalance_critical_mv"]) + "mV)"
        ),
    },
    {
        "rule_id": "TEMP_IMBALANCE_WARNING",
        "severity": "WARNING",
        "title": "Thermal Gradient Warning",
        "description": "Temperature difference across battery pack modules has exceeded the warning threshold.",
        "possible_cause": "Uneven cooling, partially blocked cooling channel, or varying module load.",
        "recommended_action": "Inspect module cooling path. Check for blocked vents or degraded thermal interface material.",
        "business_impact": "Medium — thermal gradient signals uneven ageing if sustained over multiple days.",
        "condition": lambda df: (
            (_temp_diff(df) >= RT["temp_imbalance_warning_c"])
            & (_temp_diff(df) < RT["temp_imbalance_critical_c"])
        ).values,
        "evidence": lambda df: (
            "temp_imbalance=" + _temp_diff(df).round(1).astype(str)
            + "°C (threshold=" + str(RT["temp_imbalance_warning_c"]) + "°C)"
        ),
    },
    {
        "rule_id": "TEMP_IMBALANCE_CRITICAL",
        "severity": "CRITICAL",
        "title": "Thermal Gradient Critical",
        "description": "Severe temperature gradient across battery modules. Hot-spot conditions present.",
        "possible_cause": "Cooling system failure in one or more modules, or severely degraded cell group.",
        "recommended_action": "Inspect all module cooling paths immediately. Take vehicle offline pending inspection.",
        "business_impact": "High — persistent hot-spot accelerates localised degradation and reduces pack life.",
        "condition": lambda df: (
            _temp_diff(df) >= RT["temp_imbalance_critical_c"]
        ).values,
        "evidence": lambda df: (
            "temp_imbalance=" + _temp_diff(df).round(1).astype(str)
            + "°C (threshold=" + str(RT["temp_imbalance_critical_c"]) + "°C)"
        ),
    },
    {
        "rule_id": "HIGH_CURRENT_STRESS",
        "severity": "WARNING",
        "title": "High Current Stress",
        "description": "Battery is operating under high current, increasing electrical stress.",
        "possible_cause": "Heavy acceleration, high vehicle load, or aggressive driving pattern.",
        "recommended_action": "Review vehicle load and operator driving pattern. Consider current limiting.",
        "business_impact": "Medium — sustained high current reduces cycle life.",
        "condition": lambda df: (
            _safe(df, "abs_current_a") >= RT["high_current_a"]
        ).values,
        "evidence": lambda df: (
            "abs_current_a=" + _safe(df, "abs_current_a").round(1).astype(str)
            + "A (threshold=" + str(RT["high_current_a"]) + "A)"
        ),
    },
    {
        "rule_id": "HIGH_CURRENT_WITH_HIGH_TEMP",
        "severity": "CRITICAL",
        "title": "High Current Under Elevated Temperature",
        "description": "High current load is occurring while battery temperature is already elevated.",
        "possible_cause": "Aggressive operation in high ambient temperature or after sustained heavy use.",
        "recommended_action": "Reduce load immediately. Allow battery to cool. Both stressors together significantly accelerate wear.",
        "business_impact": "High — combined thermal and electrical stress is the most damaging operating condition.",
        "condition": lambda df: (
            (_safe(df, "abs_current_a") >= RT["combined_current_a"])
            & (_safe(df, "max_battery_temp_c") >= RT["combined_temp_c"])
        ).values,
        "evidence": lambda df: (
            "abs_current_a=" + _safe(df, "abs_current_a").round(1).astype(str)
            + "A, max_battery_temp_c=" + _safe(df, "max_battery_temp_c").round(1).astype(str) + "°C"
        ),
    },
    {
        "rule_id": "WEAK_MODULE_VOLTAGE",
        "severity": "WARNING",
        "title": "Weak Module — Voltage Imbalance",
        "description": "One or more battery modules shows significantly higher voltage spread than others.",
        "possible_cause": "Degraded cell(s) within that module, or connection resistance issue.",
        "recommended_action": "Inspect module with highest voltage range. Check lowest cell voltage and BMS balancing for that module.",
        "business_impact": "Medium — weak module limits overall pack capacity and may trigger premature BMS cutoff.",
        "condition": lambda df: (
            pd.DataFrame(
                {f"module_{i}_voltage_range": _safe(df, f"module_{i}_voltage_range") * 1000
                 for i in MODULES if f"module_{i}_voltage_range" in df.columns}
            ).max(axis=1) >= RT["weak_module_voltage_mv"]
            if any(f"module_{i}_voltage_range" in df.columns for i in MODULES)
            else pd.Series(False, index=df.index)
        ).values,
        "evidence": lambda df: (
            "max_module_voltage_range=" + pd.DataFrame(
                {f"module_{i}_voltage_range": _safe(df, f"module_{i}_voltage_range") * 1000
                 for i in MODULES if f"module_{i}_voltage_range" in df.columns}
            ).max(axis=1).round(1).astype(str) + "mV"
            if any(f"module_{i}_voltage_range" in df.columns for i in MODULES)
            else pd.Series("n/a", index=df.index)
        ),
    },
    {
        "rule_id": "WEAK_MODULE_THERMAL",
        "severity": "WARNING",
        "title": "Weak Module — Thermal Hotspot",
        "description": "One or more battery modules has a significantly higher temperature range than others.",
        "possible_cause": "Degraded cooling for that module, or a cell with elevated internal resistance.",
        "recommended_action": "Inspect cooling path for hottest module. Check hottest temperature sensor reading.",
        "business_impact": "Medium — thermal hotspot accelerates localised degradation.",
        "condition": lambda df: (
            pd.DataFrame(
                {f"module_{i}_temp_range": _safe(df, f"module_{i}_temp_range")
                 for i in MODULES if f"module_{i}_temp_range" in df.columns}
            ).max(axis=1) >= RT["weak_module_temp_c"]
            if any(f"module_{i}_temp_range" in df.columns for i in MODULES)
            else pd.Series(False, index=df.index)
        ).values,
        "evidence": lambda df: (
            "max_module_temp_range=" + pd.DataFrame(
                {f"module_{i}_temp_range": _safe(df, f"module_{i}_temp_range")
                 for i in MODULES if f"module_{i}_temp_range" in df.columns}
            ).max(axis=1).round(1).astype(str) + "°C"
            if any(f"module_{i}_temp_range" in df.columns for i in MODULES)
            else pd.Series("n/a", index=df.index)
        ),
    },
    {
        "rule_id": "BMS_OR_CELL_DATA_QUALITY_ISSUE",
        "severity": "WARNING",
        "title": "BMS or Cell Data Quality Issue",
        "description": "Data quality flags indicate missing, gap-filled, or out-of-range cell/BMS data.",
        "possible_cause": "CAN communication drop, faulty cell sensor, or wiring harness issue.",
        "recommended_action": "Check BMS and CAN bus communication logs. Inspect cell wiring harness and sensor connections.",
        "business_impact": "Low-Medium — data gaps prevent accurate health monitoring and may mask real issues.",
        "condition": lambda df: (
            (
                (df["cell_quality_flag"].fillna(0).astype(int) & (
                    CELL_FLAG["PACK_GAP"] | CELL_FLAG["MODULE_GAP"]
                    | CELL_FLAG["CELL_HARD_BREACH"] | CELL_FLAG["CELL_OUTLIER"]
                )) != 0
                if "cell_quality_flag" in df.columns
                else pd.Series(False, index=df.index)
            ) | (
                df["vcu_quality_flag"].fillna(0).astype(int) != 0
                if "vcu_quality_flag" in df.columns
                else pd.Series(False, index=df.index)
            )
        ).values,
        "evidence": lambda df: (
            "cell_quality_flag=" + _safe(df, "cell_quality_flag").astype(int).astype(str)
            + ", vcu_quality_flag=" + _safe(df, "vcu_quality_flag").astype(int).astype(str)
        ),
    },
    {
        "rule_id": "BATTERY_FAULT_PRESENT",
        "severity": "CRITICAL",
        "title": "Active Battery Fault",
        "description": "BMS has recorded one or more active fault conditions.",
        "possible_cause": "Hardware fault, sensor failure, overcurrent, or overvoltage event.",
        "recommended_action": "Prioritise immediate service inspection. Review fault log. Take vehicle offline if fault persists.",
        "business_impact": "High — fault-driven failures risk unplanned downtime and warranty claims.",
        "condition": lambda df: (
            _safe(df, "fault_any").astype(int) != 0
        ).values,
        "evidence": lambda df: (
            "fault_any=1"
        ),
    },
]


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate all 12 rules against df.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame of triggered alerts.
        Columns: timestamp, vehicle_id, rule_id, severity, title, description,
                 evidence, possible_cause, recommended_action, business_impact.
        Only rows where the rule condition is True are included.
        Empty DataFrame with the same schema if no rules fire.
    """
    result_chunks: list[pd.DataFrame] = []

    base_cols = [c for c in ("timestamp", "vehicle_id", "trip_id") if c in df.columns]

    for rule in _RULES:
        mask = rule["condition"](df)
        if not mask.any():
            continue

        triggered = df.loc[mask, base_cols].copy()
        triggered["component"] = "battery"
        triggered["rule_id"] = rule["rule_id"]
        triggered["severity"] = rule["severity"]
        triggered["title"] = rule["title"]
        triggered["description"] = rule["description"]
        triggered["possible_cause"] = rule["possible_cause"]
        triggered["recommended_action"] = rule["recommended_action"]
        triggered["business_impact"] = rule["business_impact"]

        ev = rule["evidence"](df)
        if isinstance(ev, str):
            triggered["evidence"] = ev
        else:
            triggered["evidence"] = ev[mask].values

        result_chunks.append(triggered)

    if not result_chunks:
        return pd.DataFrame(columns=[
            "timestamp", "vehicle_id", "trip_id", "component",
            "rule_id", "severity", "title", "description",
            "evidence", "possible_cause", "recommended_action", "business_impact",
        ])

    return pd.concat(result_chunks, ignore_index=True)
