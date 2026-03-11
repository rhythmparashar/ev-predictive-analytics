"""
Validation Stage (Raw → Silver preparation)

Purpose
-------
This module validates and cleans raw telemetry before resampling.

It performs:

1) Schema validation
   - Required columns must exist

2) Timestamp parsing
   - Converts to UTC datetime
   - Flags invalid timestamps
   - Flags time reversal anomalies
   - Drops rows with invalid timestamps after flagging/reporting

3) Duplicate timestamp reporting
   - Expected for 2 Hz telemetry with second-level timestamps
   - NOT treated as anomaly

4) Numeric conversion
   - Converts declared numeric signals safely using schema

5) Placeholder value cleaning
   - Example: stack_voltage_v == 0 → treated as missing

6) Range checks
   - Soft breaches → flagged but allowed
   - Hard breaches → flagged and value nulled

7) Null-rate reporting

8) Quality flag bitmask handling

Output
------
Returns:
    ValidationResult:
        df      → cleaned dataframe
        report  → validation statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml
import pandas as pd


@dataclass
class ValidationResult:
    df: pd.DataFrame
    report: dict


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _numeric_columns_from_schema(schema: dict) -> list[str]:
    cols = schema.get("columns", {})
    out: list[str] = []
    for col, dtype in cols.items():
        if dtype in {"float64", "int64", "Int64", "float32", "int32"}:
            out.append(col)
    return out


def validate(
    df: pd.DataFrame,
    schema_path: Path,
    ranges_path: Path,
    quality_flags_path: Path,
) -> ValidationResult:
    schema = load_yaml(schema_path)
    ranges = load_yaml(ranges_path)
    qf = load_yaml(quality_flags_path)

    df = df.copy()

    required = schema.get("required", [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "quality_flag" not in df.columns:
        df["quality_flag"] = 0
    df["quality_flag"] = df["quality_flag"].fillna(0).astype("int64")

    # -------------------------------------------------
    # Timestamp parsing + anomaly detection
    # -------------------------------------------------

    raw_ts = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=True,
        utc=True,
    )

    bad_ts = raw_ts.isna()

    # Detect timestamp reversal in original file order
    ts_diff = raw_ts.diff()
    reversed_ts = ts_diff.lt(pd.Timedelta(0)).fillna(False)

    if bad_ts.any():
        df.loc[bad_ts, "quality_flag"] |= int(qf["TIME_ANOMALY"])

    if reversed_ts.any():
        df.loc[reversed_ts, "quality_flag"] |= int(qf["TIME_ANOMALY"])

    df["timestamp"] = raw_ts

    # Duplicate timestamps are expected at 2 Hz with second precision
    duplicate_count = int(df["timestamp"].duplicated(keep="first").sum())

    # Drop rows that cannot participate in a time-series pipeline
    dropped_bad_timestamp_rows = int(bad_ts.sum())
    if dropped_bad_timestamp_rows > 0:
        df = df.loc[df["timestamp"].notna()].copy()

    # -------------------------------------------------
    # Numeric conversion from schema
    # -------------------------------------------------

    numeric_cols = _numeric_columns_from_schema(schema)
    for col in numeric_cols:
        if col in df.columns and col not in {"quality_flag"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------
    # Placeholder values -> NaN
    # Optional future move: schema/placeholders.yaml
    # -------------------------------------------------

    zero_as_missing = ["stack_voltage_v"]

    for col in zero_as_missing:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            df[col] = s.mask(s == 0)

    # -------------------------------------------------
    # Range checks
    # Soft -> flag only
    # Hard -> flag + null that signal value
    # -------------------------------------------------

    hard_breaches = 0
    soft_breaches = 0

    soft_by_col: dict[str, int] = {}
    hard_by_col: dict[str, int] = {}

    for col, lim in ranges.items():
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        soft_lo, soft_hi = lim.get("soft", [None, None])
        hard_lo, hard_hi = lim.get("hard", [None, None])

        if soft_lo is not None and soft_hi is not None:
            soft_mask = s.notna() & ((s < soft_lo) | (s > soft_hi))
            cnt = int(soft_mask.sum())
            soft_breaches += cnt
            soft_by_col[col] = cnt
            if cnt:
                df.loc[soft_mask, "quality_flag"] |= int(qf["SOFT_RANGE_BREACH"])

        if hard_lo is not None and hard_hi is not None:
            hard_mask = s.notna() & ((s < hard_lo) | (s > hard_hi))
            cnt = int(hard_mask.sum())
            hard_breaches += cnt
            hard_by_col[col] = cnt
            if cnt:
                df.loc[hard_mask, "quality_flag"] |= int(qf["HARD_RANGE_BREACH"])
                df.loc[hard_mask, col] = pd.NA

    report = {
        "row_count_after_validation": int(len(df)),
        "missing_required_columns": missing,
        "duplicate_timestamps": duplicate_count,  # informational
        "bad_timestamps": dropped_bad_timestamp_rows,
        "time_reversal_rows": int(reversed_ts.sum()),
        "soft_range_breaches": soft_breaches,
        "hard_range_breaches": hard_breaches,
        "soft_breaches_by_col": dict(
            sorted(soft_by_col.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "hard_breaches_by_col": dict(
            sorted(hard_by_col.items(), key=lambda x: x[1], reverse=True)[:10]
        ),
        "null_rates": {c: float(df[c].isna().mean()) for c in df.columns},
    }

    return ValidationResult(df=df, report=report)