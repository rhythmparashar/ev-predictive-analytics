import pandas as pd
import pytest

from ingestion.resampler import aggregate_duplicates_per_timestamp, resample_1hz


# ── helpers ───────────────────────────────────────────────────────────────────

def ts(sec: int) -> pd.Timestamp:
    return pd.Timestamp(f"2026-01-06 08:00:{sec:02d}", tz="UTC")


# ═══════════════════════════════════════════════════════════════════════════════
# aggregate_duplicates_per_timestamp
# ═══════════════════════════════════════════════════════════════════════════════

def test_no_duplicates_returns_same_row_count():
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(1)],
        "vehicle_id": ["EV01", "EV01"],
        "soc_pct":    [80.0, 79.0],
    })
    result = aggregate_duplicates_per_timestamp(df)
    assert len(result) == 2


def test_numeric_columns_averaged_on_duplicate_timestamp():
    df = pd.DataFrame({
        "timestamp": [ts(0), ts(0)],
        "soc_pct":   [80.0, 60.0],
    })
    result = aggregate_duplicates_per_timestamp(df)
    assert len(result) == 1
    assert result["soc_pct"].iloc[0] == pytest.approx(70.0)


def test_quality_flag_is_bitwise_or_on_duplicate_timestamp():
    # bit 0 (1) | bit 1 (2) = 3
    df = pd.DataFrame({
        "timestamp":    [ts(0), ts(0)],
        "quality_flag": [1, 2],
    })
    result = aggregate_duplicates_per_timestamp(df)
    assert result["quality_flag"].iloc[0] == 3


def test_non_numeric_column_takes_last_non_null():
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(0)],
        "vehicle_id": ["EV01", "EV02"],
    })
    result = aggregate_duplicates_per_timestamp(df)
    assert result["vehicle_id"].iloc[0] == "EV02"


def test_missing_timestamp_column_raises():
    df = pd.DataFrame({"soc_pct": [80.0, 79.0]})
    with pytest.raises(ValueError, match="timestamp"):
        aggregate_duplicates_per_timestamp(df)


# ═══════════════════════════════════════════════════════════════════════════════
# resample_1hz
# ═══════════════════════════════════════════════════════════════════════════════

def test_resample_fills_to_1hz(resample_cfg, signal_classes_path, quality_flags_path):
    # Two rows 5 seconds apart → 6 rows at 1 Hz (t=0,1,2,3,4,5)
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(5)],
        "vehicle_id": ["EV01", "EV01"],
        "soc_pct":    [80.0, 75.0],
    })
    result = resample_1hz(df, resample_cfg, signal_classes_path, quality_flags_path)
    assert len(result) == 6


def test_resample_gap_inserted_flag_on_inserted_rows(resample_cfg, signal_classes_path, quality_flags_path):
    # Rows t=1,2,3,4 are inserted; GAP_INSERTED = 4
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(5)],
        "vehicle_id": ["EV01", "EV01"],
        "soc_pct":    [80.0, 75.0],
    })
    result = resample_1hz(df, resample_cfg, signal_classes_path, quality_flags_path)
    GAP_INSERTED = 4
    inserted = result.iloc[1:5]
    assert (inserted["quality_flag"] & GAP_INSERTED).all()


def test_resample_original_rows_have_no_gap_flag(resample_cfg, signal_classes_path, quality_flags_path):
    # t=0 and t=5 are original rows — must NOT have GAP_INSERTED bit
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(5)],
        "vehicle_id": ["EV01", "EV01"],
        "soc_pct":    [80.0, 75.0],
    })
    result = resample_1hz(df, resample_cfg, signal_classes_path, quality_flags_path)
    GAP_INSERTED = 4
    assert not (result.iloc[0]["quality_flag"] & GAP_INSERTED)
    assert not (result.iloc[5]["quality_flag"] & GAP_INSERTED)


def test_resample_vehicle_id_propagated_to_inserted_rows(resample_cfg, signal_classes_path, quality_flags_path):
    df = pd.DataFrame({
        "timestamp":  [ts(0), ts(3)],
        "vehicle_id": ["EV01", "EV01"],
        "soc_pct":    [80.0, 77.0],
    })
    result = resample_1hz(df, resample_cfg, signal_classes_path, quality_flags_path)
    assert (result["vehicle_id"] == "EV01").all()
