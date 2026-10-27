import pandas as pd
import pytest

from battery_intelligence.rule_engine import evaluate_rules
from battery_intelligence.config import RULE_THRESHOLDS as RT


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_row(**kwargs) -> pd.DataFrame:
    """A single row that triggers no rules. Override any field via kwargs."""
    base = {
        "timestamp":          pd.Timestamp("2026-01-06 08:00:00"),
        "vehicle_id":         "EV01",
        "max_battery_temp_c": 30.0,       # below 45°C warning
        "pack_voltage_range": 0.05,        # 50 mV, below 90 mV warning
        "pack_temp_range":    0.0,         # explicit 0 → avoids max-min fallback firing TEMP_IMBALANCE
        "abs_current_a":      50.0,        # below 100 A
        "fault_any":          0,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def fired(df: pd.DataFrame) -> set:
    return set(evaluate_rules(df)["rule_id"].tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_rules_fire_on_safe_data():
    assert len(evaluate_rules(safe_row())) == 0


def test_empty_result_has_expected_columns():
    result = evaluate_rules(safe_row())
    for col in ("rule_id", "severity", "title", "description", "evidence"):
        assert col in result.columns


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE
# ═══════════════════════════════════════════════════════════════════════════════

def test_temp_warning_fires_at_threshold():
    assert "BATTERY_TEMP_WARNING" in fired(safe_row(max_battery_temp_c=RT["temp_warning_c"]))


def test_temp_warning_not_fired_below_threshold():
    assert "BATTERY_TEMP_WARNING" not in fired(safe_row(max_battery_temp_c=RT["temp_warning_c"] - 0.1))


def test_temp_critical_fires_at_threshold():
    assert "BATTERY_TEMP_CRITICAL" in fired(safe_row(max_battery_temp_c=RT["temp_critical_c"]))


def test_temp_warning_not_fired_at_critical_temp():
    # WARNING condition is (>= temp_warning AND < temp_critical) — must be False at critical
    assert "BATTERY_TEMP_WARNING" not in fired(safe_row(max_battery_temp_c=RT["temp_critical_c"]))


# ═══════════════════════════════════════════════════════════════════════════════
# VOLTAGE IMBALANCE
# ═══════════════════════════════════════════════════════════════════════════════

def test_voltage_imbalance_warning_fires():
    # pack_voltage_range is in Volts in the master dataset; rule multiplies × 1000 → mV
    df = safe_row(pack_voltage_range=RT["voltage_imbalance_warning_mv"] / 1000.0)
    assert "VOLTAGE_IMBALANCE_WARNING" in fired(df)


def test_voltage_imbalance_warning_not_fired_below_threshold():
    df = safe_row(pack_voltage_range=(RT["voltage_imbalance_warning_mv"] - 1) / 1000.0)
    assert "VOLTAGE_IMBALANCE_WARNING" not in fired(df)


def test_voltage_imbalance_critical_fires():
    df = safe_row(pack_voltage_range=RT["voltage_imbalance_critical_mv"] / 1000.0)
    assert "VOLTAGE_IMBALANCE_CRITICAL" in fired(df)


def test_voltage_imbalance_warning_not_fired_at_critical():
    # WARNING condition is (>= warning AND < critical) — must be False at critical
    df = safe_row(pack_voltage_range=RT["voltage_imbalance_critical_mv"] / 1000.0)
    assert "VOLTAGE_IMBALANCE_WARNING" not in fired(df)


# ═══════════════════════════════════════════════════════════════════════════════
# CURRENT STRESS
# ═══════════════════════════════════════════════════════════════════════════════

def test_high_current_fires_at_threshold():
    assert "HIGH_CURRENT_STRESS" in fired(safe_row(abs_current_a=RT["high_current_a"]))


def test_high_current_not_fired_below_threshold():
    assert "HIGH_CURRENT_STRESS" not in fired(safe_row(abs_current_a=RT["high_current_a"] - 0.1))


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED STRESS
# ═══════════════════════════════════════════════════════════════════════════════

def test_combined_stress_fires_when_both_conditions_met():
    df = safe_row(abs_current_a=RT["combined_current_a"], max_battery_temp_c=RT["combined_temp_c"])
    assert "HIGH_CURRENT_WITH_HIGH_TEMP" in fired(df)


def test_combined_stress_not_fired_with_only_high_current():
    df = safe_row(abs_current_a=RT["combined_current_a"], max_battery_temp_c=30.0)
    assert "HIGH_CURRENT_WITH_HIGH_TEMP" not in fired(df)


def test_combined_stress_not_fired_with_only_high_temp():
    df = safe_row(abs_current_a=10.0, max_battery_temp_c=RT["combined_temp_c"])
    assert "HIGH_CURRENT_WITH_HIGH_TEMP" not in fired(df)


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT
# ═══════════════════════════════════════════════════════════════════════════════

def test_fault_fires_when_active():
    assert "BATTERY_FAULT_PRESENT" in fired(safe_row(fault_any=1))


def test_fault_not_fired_when_zero():
    assert "BATTERY_FAULT_PRESENT" not in fired(safe_row(fault_any=0))


# ═══════════════════════════════════════════════════════════════════════════════
# SEVERITY AND MULTI-RULE
# ═══════════════════════════════════════════════════════════════════════════════

def test_severity_of_temp_warning_is_warning():
    result = evaluate_rules(safe_row(max_battery_temp_c=RT["temp_warning_c"]))
    row = result[result["rule_id"] == "BATTERY_TEMP_WARNING"].iloc[0]
    assert row["severity"] == "WARNING"


def test_severity_of_temp_critical_is_critical():
    result = evaluate_rules(safe_row(max_battery_temp_c=RT["temp_critical_c"]))
    row = result[result["rule_id"] == "BATTERY_TEMP_CRITICAL"].iloc[0]
    assert row["severity"] == "CRITICAL"


def test_multiple_rules_fire_simultaneously():
    # Critical temp + active fault → both rules fire
    df = safe_row(max_battery_temp_c=RT["temp_critical_c"], fault_any=1)
    rules = fired(df)
    assert "BATTERY_TEMP_CRITICAL" in rules
    assert "BATTERY_FAULT_PRESENT" in rules
