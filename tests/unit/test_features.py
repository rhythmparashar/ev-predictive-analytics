import numpy as np
import pandas as pd
import pytest

from features.lags import lag_features
from features.physics import physics_features, _safe_ratio
from features.rolling import rolling_features


# ── helpers ───────────────────────────────────────────────────────────────────

def make_trip_df(n: int = 10, soc_start: float = 80.0, trip_id: str = "T1") -> pd.DataFrame:
    """Single-trip DataFrame with a simple SOC ramp-down."""
    return pd.DataFrame({
        "trip_id": trip_id,
        "soc_pct": [soc_start - i for i in range(n)],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def test_lag_first_rows_are_nan():
    # lag-3 means rows 0,1,2 have no prior value — must be NaN
    df = make_trip_df(10)
    out = lag_features(df, signals=["soc_pct"], lags_s=[3])
    assert out["soc_pct_lag_3s"].iloc[:3].isna().all()


def test_lag_correct_value():
    # row 5 with lag-3 must equal the value at row 2
    df = make_trip_df(10)
    out = lag_features(df, signals=["soc_pct"], lags_s=[3])
    assert out["soc_pct_lag_3s"].iloc[5] == pytest.approx(df["soc_pct"].iloc[2])


def test_lag_respects_trip_boundary():
    # Two trips concatenated — lag must not bleed from trip 1 into trip 2
    trip1 = make_trip_df(5, soc_start=90.0, trip_id="T1")
    trip2 = make_trip_df(5, soc_start=50.0, trip_id="T2")
    df = pd.concat([trip1, trip2], ignore_index=True)

    out = lag_features(df, signals=["soc_pct"], lags_s=[2])

    # First two rows of trip2 (index 5, 6) must be NaN — not carry values from trip1
    trip2_rows = out[out["trip_id"] == "T2"]
    assert trip2_rows["soc_pct_lag_2s"].iloc[:2].isna().all()


def test_lag_missing_signal_silently_skipped():
    # Requesting a column that doesn't exist should not crash
    df = make_trip_df(5)
    out = lag_features(df, signals=["nonexistent_col"], lags_s=[1])
    assert "nonexistent_col_lag_1s" not in out.columns


def test_lag_missing_group_col_raises():
    df = pd.DataFrame({"soc_pct": [1, 2, 3]})  # no trip_id column
    with pytest.raises(ValueError, match="trip_id"):
        lag_features(df, signals=["soc_pct"], lags_s=[1])


# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def test_elec_power_proxy_formula():
    # P = V * I / 1000
    df = pd.DataFrame({"stack_voltage_v": [400.0], "battery_current_a": [100.0]})
    out = physics_features(df)
    assert out["elec_power_kw_proxy"].iloc[0] == pytest.approx(40.0)


def test_cell_voltage_delta():
    df = pd.DataFrame({"max_cell_voltage_v": [3.8], "min_cell_voltage_v": [3.5]})
    out = physics_features(df)
    assert out["cell_voltage_delta_v"].iloc[0] == pytest.approx(0.3)


def test_battery_temp_delta():
    df = pd.DataFrame({"max_battery_temp_c": [45.0], "min_battery_temp_c": [30.0]})
    out = physics_features(df)
    assert out["battery_temp_delta_c"].iloc[0] == pytest.approx(15.0)


def test_mech_power_proxy_formula():
    # P_mech = torque * (rpm * 2π/60) / 1000
    rpm, torque = 1000.0, 100.0
    expected = torque * (rpm * 2 * np.pi / 60) / 1000.0
    df = pd.DataFrame({
        "motor_torque_value_nm": [torque],
        "motor_speed_rpm": [rpm],
        "stack_voltage_v": [400.0],
        "battery_current_a": [50.0],
    })
    out = physics_features(df)
    assert out["mech_power_kw_proxy"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_missing_columns_produce_no_output():
    # DataFrame with unrelated columns — no physics features should be added
    df = pd.DataFrame({"vehicle_id": ["EV01"], "trip_id": ["T1"]})
    out = physics_features(df)
    assert "elec_power_kw_proxy" not in out.columns
    assert "cell_voltage_delta_v" not in out.columns


# ── _safe_ratio ───────────────────────────────────────────────────────────────

def test_safe_ratio_normal_case():
    num = pd.Series([10.0])
    den = pd.Series([2.0])
    result = _safe_ratio(num, den, min_abs_den=1e-3, clip_abs=10.0)
    assert result.iloc[0] == pytest.approx(5.0)


def test_safe_ratio_clips_extreme_values():
    # 100 / 0.1 = 1000, but clip_abs=10 → result must be 10
    num = pd.Series([100.0])
    den = pd.Series([0.1])
    result = _safe_ratio(num, den, min_abs_den=1e-3, clip_abs=10.0)
    assert result.iloc[0] == pytest.approx(10.0)


def test_safe_ratio_tiny_denominator_returns_nan():
    # denominator below min_abs_den → NaN (safe divide)
    num = pd.Series([5.0])
    den = pd.Series([0.0])
    result = _safe_ratio(num, den, min_abs_den=1e-3, clip_abs=10.0)
    assert result.isna().iloc[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def test_rolling_mean_correct_value():
    # window=3: row 2 mean = mean(0,1,2) = mean(10,20,30) = 20
    df = pd.DataFrame({"trip_id": "T1", "soc_pct": [10.0, 20.0, 30.0, 40.0, 50.0]})
    out = rolling_features(df, signals=["soc_pct"], windows_s=[3], aggs=["mean"])
    assert out["soc_pct_roll_mean_3s"].iloc[2] == pytest.approx(20.0)


def test_rolling_first_rows_are_nan():
    # window=3 with min_periods=3 (default) → rows 0 and 1 must be NaN
    df = make_trip_df(10)
    out = rolling_features(df, signals=["soc_pct"], windows_s=[3], aggs=["mean"])
    assert out["soc_pct_roll_mean_3s"].iloc[:2].isna().all()


def test_rolling_std_uses_ddof_zero():
    # ddof=0 (population std): std([10,20,30]) = sqrt(mean([(-10)^2, 0, 10^2])) = sqrt(200/3)
    df = pd.DataFrame({"trip_id": "T1", "soc_pct": [10.0, 20.0, 30.0]})
    out = rolling_features(df, signals=["soc_pct"], windows_s=[3], aggs=["std"])
    expected = np.std([10.0, 20.0, 30.0], ddof=0)
    assert out["soc_pct_roll_std_3s"].iloc[2] == pytest.approx(expected)


def test_rolling_min_max():
    df = pd.DataFrame({"trip_id": "T1", "soc_pct": [50.0, 30.0, 70.0, 20.0, 60.0]})
    out = rolling_features(df, signals=["soc_pct"], windows_s=[3], aggs=["min", "max"])
    # window ending at row 3: [30, 70, 20] → min=20, max=70
    assert out["soc_pct_roll_min_3s"].iloc[3] == pytest.approx(20.0)
    assert out["soc_pct_roll_max_3s"].iloc[3] == pytest.approx(70.0)


def test_rolling_missing_signal_silently_skipped():
    df = make_trip_df(5)
    out = rolling_features(df, signals=["does_not_exist"], windows_s=[3], aggs=["mean"])
    assert "does_not_exist_roll_mean_3s" not in out.columns
