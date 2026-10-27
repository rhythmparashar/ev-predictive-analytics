import pandas as pd
import pytest

from ingestion.faults import _sanitize_fault_code, add_fault_flags


# ── helpers ───────────────────────────────────────────────────────────────────

def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_telemetry(timestamps) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp":  [ts(t) for t in timestamps],
        "vehicle_id": "EV01",
    })


def make_fault(code: str, activated_at: str, fixed_at: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        "code":         code,
        "activated_at": ts(activated_at),
        "fixed_at":     ts(fixed_at) if fixed_at else pd.NaT,
    }])


# ═══════════════════════════════════════════════════════════════════════════════
# _sanitize_fault_code
# ═══════════════════════════════════════════════════════════════════════════════

def test_sanitize_lowercases_code():
    assert _sanitize_fault_code("FAULT_ONE") == "fault_one"


def test_sanitize_replaces_spaces_with_underscore():
    assert _sanitize_fault_code("fault one two") == "fault_one_two"


def test_sanitize_strips_leading_trailing_underscores():
    assert _sanitize_fault_code("_fault_") == "fault"


def test_sanitize_collapses_multiple_separators():
    assert _sanitize_fault_code("fault--two  three") == "fault_two_three"


# ═══════════════════════════════════════════════════════════════════════════════
# add_fault_flags — empty faults
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_faults_sets_all_to_zero():
    df = make_telemetry(["2026-01-06 08:00:00", "2026-01-06 08:00:01"])
    faults = pd.DataFrame(columns=["code", "activated_at", "fixed_at"])
    result = add_fault_flags(df, faults)
    assert (result["fault_any"] == 0).all()


# ═══════════════════════════════════════════════════════════════════════════════
# add_fault_flags — active interval
# ═══════════════════════════════════════════════════════════════════════════════

def test_fault_marks_timestamps_within_active_interval():
    # fault active from 08:00:01 to 08:00:03
    df = make_telemetry([
        "2026-01-06 08:00:00",
        "2026-01-06 08:00:01",
        "2026-01-06 08:00:02",
        "2026-01-06 08:00:04",
    ])
    faults = make_fault("hardware_overvoltage_fault", "2026-01-06 08:00:01", "2026-01-06 08:00:03")
    result = add_fault_flags(df, faults)
    # t=00 before fault → 0; t=01,02 inside → 1; t=04 after → 0
    assert result["fault_any"].iloc[0] == 0
    assert result["fault_any"].iloc[1] == 1
    assert result["fault_any"].iloc[2] == 1
    assert result["fault_any"].iloc[3] == 0


def test_fault_column_created_for_known_code():
    df = make_telemetry(["2026-01-06 08:00:00"])
    faults = make_fault("hardware_overvoltage_fault", "2026-01-06 07:59:00", "2026-01-06 08:00:30")
    result = add_fault_flags(df, faults)
    assert "fault_hardware_overvoltage_fault" in result.columns


def test_fault_any_set_when_any_code_active():
    df = make_telemetry(["2026-01-06 08:00:00"])
    faults = make_fault("busbar_undervoltage_fault", "2026-01-06 07:59:00", "2026-01-06 08:00:30")
    result = add_fault_flags(df, faults)
    assert result["fault_any"].iloc[0] == 1


def test_fault_not_active_after_fixed_at():
    # Fault ended at 08:00:05; telemetry row at 08:00:06 must NOT be active.
    # (The implementation is inclusive at fixed_at — the row AT fixed_at is still marked active.)
    df = make_telemetry(["2026-01-06 08:00:06"])
    faults = make_fault("bus_overvoltage_fault", "2026-01-06 08:00:00", "2026-01-06 08:00:05")
    result = add_fault_flags(df, faults)
    assert result["fault_any"].iloc[0] == 0
