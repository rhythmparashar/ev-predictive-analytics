import numpy as np
import pandas as pd
import pytest

from battery_intelligence.health_score import compute_health_scores


# ── helpers ───────────────────────────────────────────────────────────────────

def make_df(**kwargs) -> pd.DataFrame:
    """Build a single-row DataFrame with only the columns you pass in."""
    return pd.DataFrame({k: [v] for k, v in kwargs.items()})


# ── thermal penalty ───────────────────────────────────────────────────────────

def test_thermal_penalty_no_stress():
    df = make_df(max_battery_temp_c=20.0)
    out = compute_health_scores(df)
    assert out["thermal_penalty"].iloc[0] == 0


def test_thermal_penalty_just_below_warn_is_zero():
    # 44.9°C is below the 45°C warn threshold — must be zero
    # This catches any accidental lowering of the warn threshold
    df = make_df(max_battery_temp_c=44.9)
    out = compute_health_scores(df)
    assert out["thermal_penalty"].iloc[0] == 0


def test_thermal_penalty_warn_boundary():
    # Exactly at warn threshold (45°C) → 10-point penalty
    df = make_df(max_battery_temp_c=45.0)
    out = compute_health_scores(df)
    assert out["thermal_penalty"].iloc[0] == 10


def test_thermal_penalty_high_boundary():
    # Exactly at high threshold (50°C) → 20-point penalty
    df = make_df(max_battery_temp_c=50.0)
    out = compute_health_scores(df)
    assert out["thermal_penalty"].iloc[0] == 20


def test_thermal_penalty_critical_boundary():
    # Exactly at critical threshold (55°C) → 35-point penalty
    df = make_df(max_battery_temp_c=55.0)
    out = compute_health_scores(df)
    assert out["thermal_penalty"].iloc[0] == 35


# ── score clamping ────────────────────────────────────────────────────────────

def test_score_never_below_zero():
    # Stack every possible penalty — score must stay >= 0
    df = make_df(
        max_battery_temp_c=60.0,        # critical thermal
        pack_voltage_range=0.15,         # high voltage imbalance (150 mV)
        max_battery_temp_c2=None,
        abs_current_a=200.0,             # high current stress
        fault_any=1,
    )
    # Use only the columns compute_health_scores actually reads
    df2 = pd.DataFrame({
        "max_battery_temp_c": [60.0],
        "pack_voltage_range": [0.15],
        "abs_current_a": [200.0],
        "fault_any": [1],
    })
    out = compute_health_scores(df2)
    assert out["battery_health_score"].iloc[0] >= 0


def test_score_never_above_hundred():
    # Perfect data — score must stay <= 100
    df = make_df(max_battery_temp_c=20.0, abs_current_a=10.0, fault_any=0)
    out = compute_health_scores(df)
    assert out["battery_health_score"].iloc[0] <= 100


# ── health status mapping ─────────────────────────────────────────────────────

def test_status_excellent():
    # No penalties → score = 100 → EXCELLENT
    df = make_df(max_battery_temp_c=20.0)
    out = compute_health_scores(df)
    assert out["battery_health_status"].iloc[0] == "EXCELLENT"


def test_status_warning():
    # 35-point thermal penalty → score = 65 → WARNING (60–74)
    df = make_df(max_battery_temp_c=55.0)
    out = compute_health_scores(df)
    assert out["battery_health_status"].iloc[0] == "WARNING"


def test_status_critical():
    # 35 thermal + 25 fault = 60-point deduction → score = 40 → CRITICAL boundary
    df = make_df(max_battery_temp_c=55.0, fault_any=1)
    out = compute_health_scores(df)
    assert out["battery_health_score"].iloc[0] == pytest.approx(40.0)
    assert out["battery_health_status"].iloc[0] == "CRITICAL"


def test_status_no_data():
    # No recognised columns → score NaN, status NO_DATA
    df = pd.DataFrame({"vehicle_id": ["EV01"]})
    out = compute_health_scores(df)
    assert out["battery_health_status"].iloc[0] == "NO_DATA"


# ── fault penalty ─────────────────────────────────────────────────────────────

def test_fault_penalty_active():
    df = make_df(max_battery_temp_c=20.0, fault_any=1)
    out = compute_health_scores(df)
    assert out["fault_penalty"].iloc[0] == 25


def test_fault_penalty_inactive():
    df = make_df(max_battery_temp_c=20.0, fault_any=0)
    out = compute_health_scores(df)
    assert out["fault_penalty"].iloc[0] == 0


# ── voltage imbalance penalty ─────────────────────────────────────────────────

def test_voltage_imbalance_no_penalty():
    # 30 mV — below low threshold (50 mV)
    df = make_df(pack_voltage_range=0.030)
    out = compute_health_scores(df)
    assert out["voltage_imbalance_penalty"].iloc[0] == 0


def test_voltage_imbalance_high_penalty():
    # 130 mV — above high threshold (120 mV) → 35-point penalty
    df = make_df(pack_voltage_range=0.130)
    out = compute_health_scores(df)
    assert out["voltage_imbalance_penalty"].iloc[0] == 35


# ── current stress penalty ────────────────────────────────────────────────────

def test_current_stress_no_penalty():
    df = make_df(abs_current_a=30.0)
    out = compute_health_scores(df)
    assert out["current_stress_penalty"].iloc[0] == 0


def test_current_stress_high_penalty():
    # >= 150 A → 25-point penalty
    df = make_df(abs_current_a=160.0)
    out = compute_health_scores(df)
    assert out["current_stress_penalty"].iloc[0] == 25
