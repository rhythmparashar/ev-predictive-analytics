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

3) Duplicate timestamp reporting
   - Expected for 2 Hz telemetry with second-level timestamps
   - NOT treated as anomaly

4) Numeric conversion
   - Converts signals to numeric safely

5) Placeholder value cleaning
   - Example: stack_voltage_v == 0 → treated as missing

6) Range checks
   - Soft breaches → flagged but allowed
   - Hard breaches → flagged for exclusion later

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


# -------------------------
# Output container
# -------------------------

@dataclass
class ValidationResult:
    df: pd.DataFrame
    report: dict


# -------------------------
# YAML Loader
# -------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------
# Validation Pipeline
# -------------------------

def validate(
    df: pd.DataFrame,
    schema_path: Path,
    ranges_path: Path,
    quality_flags_path: Path
) -> ValidationResult:

    schema = load_yaml(schema_path)
    ranges = load_yaml(ranges_path)
    qf = load_yaml(quality_flags_path)

    # -------------------------------------------------
    # 1) Required column validation
    # -------------------------------------------------

    required = schema.get("required", [])

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")


    # -------------------------------------------------
    # 2) Ensure quality_flag column
    # -------------------------------------------------

    if "quality_flag" not in df.columns:
        df["quality_flag"] = 0

    df["quality_flag"] = df["quality_flag"].fillna(0).astype("int64")


    # -------------------------------------------------
    # 3) Timestamp parsing
    # -------------------------------------------------

    # Raw format example:
    # 15/01/2026, 11:15:05

    ts = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        dayfirst=True,
        utc=True
    )

    bad_ts = ts.isna()

    if bad_ts.any():
        df.loc[bad_ts, "quality_flag"] |= int(qf["TIME_ANOMALY"])

    df["timestamp"] = ts


    # -------------------------------------------------
    # 4) Duplicate timestamps (EXPECTED for 2Hz)
    # -------------------------------------------------

    # These occur because raw telemetry is 2Hz but
    # timestamps only have second precision.

    dup = df["timestamp"].duplicated(keep="first")

    duplicate_count = int(dup.sum())


    # -------------------------------------------------
    # 5) Convert numeric columns safely
    # -------------------------------------------------

    for col in df.columns:

        if col in ["timestamp", "vehicle_id", "battery_status",
                   "motor_rotation_direction", "motor_operation_mode"]:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")


    # -------------------------------------------------
    # 6) Replace known placeholder values with NaN
    # -------------------------------------------------

    # Many EV telemetry systems output 0 when sensor
    # is unavailable. These must be treated as missing.

    ZERO_AS_MISSING = [
        "stack_voltage_v"
    ]

    for col in ZERO_AS_MISSING:

        if col in df.columns:

            s = pd.to_numeric(df[col], errors="coerce")

            df[col] = s.mask(s == 0, other=pd.NA)


    # -------------------------------------------------
    # 7) Range checks
    # -------------------------------------------------

    hard_breaches = 0
    soft_breaches = 0

    soft_by_col = {}
    hard_by_col = {}

    for col, lim in ranges.items():

        if col not in df.columns:
            continue

        s = df[col]

        soft_lo, soft_hi = lim.get("soft", [None, None])
        hard_lo, hard_hi = lim.get("hard", [None, None])


        # Soft breaches

        if soft_lo is not None and soft_hi is not None:

            soft_mask = (s < soft_lo) | (s > soft_hi)

            cnt = int(soft_mask.sum())

            soft_breaches += cnt

            soft_by_col[col] = cnt

            df.loc[soft_mask, "quality_flag"] |= int(qf["SOFT_RANGE_BREACH"])


        # Hard breaches

        if hard_lo is not None and hard_hi is not None:

            hard_mask = (s < hard_lo) | (s > hard_hi)

            cnt = int(hard_mask.sum())

            hard_breaches += cnt

            hard_by_col[col] = cnt

            df.loc[hard_mask, "quality_flag"] |= int(qf["HARD_RANGE_BREACH"])


    # -------------------------------------------------
    # 8) Validation report
    # -------------------------------------------------

    report = {

        "row_count": int(len(df)),

        "missing_required_columns": missing,

        # Informational only (expected for 2Hz)
        "duplicate_timestamps": duplicate_count,

        "bad_timestamps": int(bad_ts.sum()),

        "soft_range_breaches": soft_breaches,

        "hard_range_breaches": hard_breaches,

        "soft_breaches_by_col":
            dict(sorted(
                soft_by_col.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),

        "hard_breaches_by_col":
            dict(sorted(
                hard_by_col.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),

        "null_rates": {
            c: float(df[c].isna().mean())
            for c in df.columns
        }

    }


    return ValidationResult(
        df=df,
        report=report
    )