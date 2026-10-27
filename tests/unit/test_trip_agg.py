import pandas as pd
import pytest

from features.trip_agg import trip_aggregations


# ── helpers ───────────────────────────────────────────────────────────────────

def make_trip(
    trip_id: str,
    n: int,
    vehicle_id: str = "EV01",
    soc_start: float = 80.0,
    fault_any: int = 0,
) -> pd.DataFrame:
    base_ts = pd.Timestamp("2026-01-06 08:00:00")
    return pd.DataFrame({
        "trip_id":    trip_id,
        "vehicle_id": vehicle_id,
        "timestamp":  [base_ts + pd.Timedelta(seconds=i) for i in range(n)],
        "soc":        [soc_start - i * 0.1 for i in range(n)],
        "fault_any":  fault_any,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED COLUMNS
# ═══════════════════════════════════════════════════════════════════════════════

def test_missing_timestamp_raises():
    df = pd.DataFrame({"trip_id": ["T1"], "vehicle_id": ["EV01"]})
    with pytest.raises(ValueError, match="timestamp"):
        trip_aggregations(df)


def test_missing_vehicle_id_raises():
    df = pd.DataFrame({"trip_id": ["T1"], "timestamp": [pd.Timestamp("2026-01-01")]})
    with pytest.raises(ValueError, match="vehicle_id"):
        trip_aggregations(df)


# ═══════════════════════════════════════════════════════════════════════════════
# MIN ROWS FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def test_short_trip_below_min_rows_excluded():
    df = make_trip("T1", n=10)   # default min_rows=60
    assert len(trip_aggregations(df)) == 0


def test_trip_at_min_rows_included():
    df = make_trip("T1", n=60)
    assert len(trip_aggregations(df)) == 1


def test_custom_min_rows_respected():
    df = make_trip("T1", n=10)
    result = trip_aggregations(df, min_rows=5)
    assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# DURATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_duration_correct():
    # 120 rows at 1-second intervals → first=t0, last=t119 → duration=119s
    df = make_trip("T1", n=120)
    result = trip_aggregations(df)
    assert result.iloc[0]["duration_s"] == 119


# ═══════════════════════════════════════════════════════════════════════════════
# SOC
# ═══════════════════════════════════════════════════════════════════════════════

def test_soc_start_and_end_correct():
    df = make_trip("T1", n=100, soc_start=80.0)
    result = trip_aggregations(df)
    assert result.iloc[0]["soc_start"] == pytest.approx(80.0)
    assert result.iloc[0]["soc_end"] == pytest.approx(80.0 - 99 * 0.1)


def test_soc_delta_correct():
    df = make_trip("T1", n=100, soc_start=80.0)
    result = trip_aggregations(df)
    expected_delta = (80.0 - 99 * 0.1) - 80.0
    assert result.iloc[0]["soc_delta"] == pytest.approx(expected_delta)


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT SECONDS
# ═══════════════════════════════════════════════════════════════════════════════

def test_fault_seconds_summed():
    df = make_trip("T1", n=100, fault_any=1)
    result = trip_aggregations(df)
    assert result.iloc[0]["fault_seconds"] == 100


def test_no_fault_seconds_when_inactive():
    df = make_trip("T1", n=100, fault_any=0)
    result = trip_aggregations(df)
    assert result.iloc[0]["fault_seconds"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPLE TRIPS
# ═══════════════════════════════════════════════════════════════════════════════

def test_multiple_trips_produce_separate_rows():
    df = pd.concat([make_trip("T1", n=100), make_trip("T2", n=100)], ignore_index=True)
    result = trip_aggregations(df)
    assert len(result) == 2
    assert set(result["trip_id"]) == {"T1", "T2"}


def test_short_trip_excluded_long_trip_kept():
    df = pd.concat([make_trip("T1", n=10), make_trip("T2", n=100)], ignore_index=True)
    result = trip_aggregations(df)
    assert len(result) == 1
    assert result.iloc[0]["trip_id"] == "T2"
