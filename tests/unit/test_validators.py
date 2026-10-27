import pandas as pd
import pytest

from ingestion.validators import validate


# Quality flag bitmasks (from schema/quality_flags.yaml)
SOFT_RANGE_BREACH = 8
HARD_RANGE_BREACH = 16
TIME_ANOMALY      = 32


# ── helpers ───────────────────────────────────────────────────────────────────

def make_df(**kwargs) -> pd.DataFrame:
    """Minimal valid telemetry row. Override any field via kwargs."""
    base = {
        "timestamp":  "2026-01-06 08:00:00",
        "vehicle_id": "EV01",
        "soc_pct":    75.0,
    }
    base.update(kwargs)
    return pd.DataFrame([base])


def run(df, schema_path, ranges_path, quality_flags_path):
    return validate(df, schema_path, ranges_path, quality_flags_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_required_column_raises(schema_path, ranges_path, quality_flags_path):
    df = pd.DataFrame([{"timestamp": "2026-01-06 08:00:00", "vehicle_id": "EV01"}])
    with pytest.raises(ValueError, match="soc_pct"):
        validate(df, schema_path, ranges_path, quality_flags_path)


def test_valid_data_passes(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(), schema_path, ranges_path, quality_flags_path)
    assert len(result.df) == 1
    assert result.report["bad_timestamps"] == 0
    assert result.report["hard_range_breaches"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TIMESTAMP HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def test_bad_timestamp_row_is_dropped(schema_path, ranges_path, quality_flags_path):
    df = pd.DataFrame([
        {"timestamp": "2026-01-06 08:00:00", "vehicle_id": "EV01", "soc_pct": 75.0},
        {"timestamp": "not-a-date",           "vehicle_id": "EV01", "soc_pct": 74.0},
    ])
    result = run(df, schema_path, ranges_path, quality_flags_path)
    assert len(result.df) == 1
    assert result.report["bad_timestamps"] == 1


def test_time_reversal_sets_quality_flag(schema_path, ranges_path, quality_flags_path):
    # Row 2 has an earlier timestamp than row 1 — reversal
    df = pd.DataFrame([
        {"timestamp": "2026-01-06 08:00:05", "vehicle_id": "EV01", "soc_pct": 75.0},
        {"timestamp": "2026-01-06 08:00:00", "vehicle_id": "EV01", "soc_pct": 74.0},
    ])
    result = run(df, schema_path, ranges_path, quality_flags_path)
    assert result.report["time_reversal_rows"] == 1
    # The reversed row (index 1) must have the TIME_ANOMALY bit set
    assert result.df["quality_flag"].iloc[1] & TIME_ANOMALY


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def test_zero_stack_voltage_becomes_nan(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(stack_voltage_v=0.0), schema_path, ranges_path, quality_flags_path)
    assert pd.isna(result.df["stack_voltage_v"].iloc[0])


def test_nonzero_stack_voltage_kept(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(stack_voltage_v=400.0), schema_path, ranges_path, quality_flags_path)
    assert result.df["stack_voltage_v"].iloc[0] == pytest.approx(400.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RANGE CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def test_soft_breach_sets_flag(schema_path, ranges_path, quality_flags_path):
    # soc_pct=101 is outside soft range [0,100] but inside hard range [-1,101]
    result = run(make_df(soc_pct=101.0), schema_path, ranges_path, quality_flags_path)
    assert result.df["quality_flag"].iloc[0] & SOFT_RANGE_BREACH
    assert result.report["soft_range_breaches"] == 1


def test_soft_breach_keeps_value(schema_path, ranges_path, quality_flags_path):
    # Soft breach — value is flagged but NOT nulled
    result = run(make_df(soc_pct=101.0), schema_path, ranges_path, quality_flags_path)
    assert result.df["soc_pct"].iloc[0] == pytest.approx(101.0)


def test_hard_breach_nulls_value(schema_path, ranges_path, quality_flags_path):
    # soc_pct=200 is outside hard range [-1,101] → value must become NaN
    result = run(make_df(soc_pct=200.0), schema_path, ranges_path, quality_flags_path)
    assert pd.isna(result.df["soc_pct"].iloc[0])


def test_hard_breach_sets_flag(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(soc_pct=200.0), schema_path, ranges_path, quality_flags_path)
    assert result.df["quality_flag"].iloc[0] & HARD_RANGE_BREACH
    assert result.report["hard_range_breaches"] == 1


def test_clean_value_has_no_flags(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(soc_pct=75.0), schema_path, ranges_path, quality_flags_path)
    assert result.df["quality_flag"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

def test_report_has_expected_keys(schema_path, ranges_path, quality_flags_path):
    result = run(make_df(), schema_path, ranges_path, quality_flags_path)
    for key in ("row_count_after_validation", "bad_timestamps",
                "soft_range_breaches", "hard_range_breaches", "null_rates"):
        assert key in result.report


def test_report_row_count_matches_df(schema_path, ranges_path, quality_flags_path):
    df = pd.DataFrame([
        {"timestamp": "2026-01-06 08:00:00", "vehicle_id": "EV01", "soc_pct": 75.0},
        {"timestamp": "2026-01-06 08:00:01", "vehicle_id": "EV01", "soc_pct": 74.0},
    ])
    result = run(df, schema_path, ranges_path, quality_flags_path)
    assert result.report["row_count_after_validation"] == len(result.df)
