import pandas as pd
import pytest

from ingestion.trip_segmentor import add_trip_id


# ── helpers ───────────────────────────────────────────────────────────────────

GAP_INSERTED = 4   # matches schema/quality_flags.yaml


def make_rows(n: int, gap: bool = False, vehicle_id: str = "EV01") -> pd.DataFrame:
    """n rows for one vehicle. gap=True means quality_flag has GAP_INSERTED bit set."""
    base = pd.Timestamp("2026-01-06 08:00:00")
    return pd.DataFrame({
        "timestamp":    [base + pd.Timedelta(seconds=i) for i in range(n)],
        "vehicle_id":   vehicle_id,
        "quality_flag": GAP_INSERTED if gap else 0,
    })


def run_segmentor(df: pd.DataFrame, trip_cfg_path, quality_flags_path, tmp_path) -> pd.DataFrame:
    state = tmp_path / "trip_state.parquet"
    return add_trip_id(df, trip_cfg_path, quality_flags_path, state)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_quality_flag_raises(trip_cfg_path, quality_flags_path, tmp_path):
    df = pd.DataFrame({
        "timestamp":  [pd.Timestamp("2026-01-06 08:00:00")],
        "vehicle_id": "EV01",
    })
    with pytest.raises(ValueError, match="quality_flag"):
        run_segmentor(df, trip_cfg_path, quality_flags_path, tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# VALID TRIP
# ═══════════════════════════════════════════════════════════════════════════════

def test_contiguous_present_rows_form_single_trip(trip_cfg_path, quality_flags_path, tmp_path):
    # 100 consecutive present rows → all assigned to one trip_id
    df = make_rows(100, gap=False)
    result = run_segmentor(df, trip_cfg_path, quality_flags_path, tmp_path)
    assert result["trip_id"].notna().all()
    assert result["trip_id"].nunique() == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SHORT TRIP FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def test_short_trip_below_min_duration_is_nulled(trip_cfg_path, quality_flags_path, tmp_path):
    # min_trip_duration_s=60 in trip.yaml; 10 rows < 60 → all trip_id should be NA
    df = make_rows(10, gap=False)
    result = run_segmentor(df, trip_cfg_path, quality_flags_path, tmp_path)
    assert result["trip_id"].isna().all()


# ═══════════════════════════════════════════════════════════════════════════════
# GAP HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def test_short_gap_does_not_break_trip(trip_cfg_path, quality_flags_path, tmp_path):
    # 70 present + 5 gap + 70 present → gap < 300 (gap_break_s) → all one trip
    df = pd.concat([
        make_rows(70, gap=False),
        make_rows(5,  gap=True),
        make_rows(70, gap=False),
    ], ignore_index=True)
    # Re-assign timestamps to be sequential
    base = pd.Timestamp("2026-01-06 08:00:00")
    df["timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(len(df))]

    result = run_segmentor(df, trip_cfg_path, quality_flags_path, tmp_path)
    present_rows = result[result["quality_flag"] & GAP_INSERTED == 0]
    assert present_rows["trip_id"].nunique() == 1


def test_long_gap_breaks_trip_into_two(trip_cfg_path, quality_flags_path, tmp_path):
    # 70 present + 300 gap + 70 present → gap >= 300 → two separate trips
    df = pd.concat([
        make_rows(70,  gap=False),
        make_rows(300, gap=True),
        make_rows(70,  gap=False),
    ], ignore_index=True)
    base = pd.Timestamp("2026-01-06 08:00:00")
    df["timestamp"] = [base + pd.Timedelta(seconds=i) for i in range(len(df))]

    result = run_segmentor(df, trip_cfg_path, quality_flags_path, tmp_path)
    # Only count distinct trip_ids from the present rows
    present_trips = result[result["quality_flag"] & GAP_INSERTED == 0]["trip_id"].dropna().unique()
    assert len(present_trips) == 2
