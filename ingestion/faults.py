"""
Fault ingestion and flagging.

Adds binary columns to silver telemetry:

fault_busbar_undervoltage_fault
fault_bus_overvoltage_fault
...

Also adds:
fault_any  (1 if any fault active)
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


# Canonical known faults from current domain knowledge.
# New unseen codes will still be added dynamically after sanitization.
KNOWN_FAULT_CODES = [
    "busbar_undervoltage_fault",
    "bus_overvoltage_fault",
    "hardware_overvoltage_fault",
    "total_hardware_failure",
    "ac_hall_failure",
    "module_over_temperature_warning",
    "temperature_difference_failure",
    "low_voltage_undervoltage_fault",
    "software_overcurrent_fault",
]


def _sanitize_fault_code(code: object) -> str:
    s = str(code).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def load_fault_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["activated_at", "fixed_at", "code"])

    faults = pd.read_csv(path)

    faults = faults.rename(
        columns={
            "Activated At": "activated_at",
            "Fixed At": "fixed_at",
            "Code": "code",
        }
    )

    for col in ["activated_at", "fixed_at", "code"]:
        if col not in faults.columns:
            faults[col] = pd.NA

    faults["activated_at"] = pd.to_datetime(
        faults["activated_at"],
        dayfirst=True,
        errors="coerce",
        utc=True,
    )

    faults["fixed_at"] = pd.to_datetime(
        faults["fixed_at"],
        dayfirst=True,
        errors="coerce",
        utc=True,
    )

    faults["code"] = faults["code"].map(_sanitize_fault_code)

    faults = faults.dropna(subset=["activated_at", "code"]).sort_values("activated_at").reset_index(drop=True)
    return faults


def _all_fault_codes(faults: pd.DataFrame) -> list[str]:
    dynamic = []
    if not faults.empty and "code" in faults.columns:
        dynamic = sorted([c for c in faults["code"].dropna().unique().tolist() if c])
    return sorted(set(KNOWN_FAULT_CODES).union(dynamic))


def add_fault_flags(df: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    all_codes = _all_fault_codes(faults)
    fault_cols = [f"fault_{code}" for code in all_codes]

    for col in fault_cols:
        if col not in df.columns:
            df[col] = np.int8(0)

    if faults.empty:
        df["fault_any"] = np.int8(0)
        return df

    ts = df["timestamp"].to_numpy(dtype="datetime64[ns]")
    if len(ts) == 0:
        df["fault_any"] = np.int8(0)
        return df

    telemetry_end = df["timestamp"].max().to_datetime64()

    # Open faults remain active until end of telemetry
    faults = faults.copy()
    faults["fixed_at"] = faults["fixed_at"].fillna(pd.Timestamp(df["timestamp"].max()))

    for code, f in faults.groupby("code", sort=True):
        col = f"fault_{code}"
        starts = f["activated_at"].to_numpy(dtype="datetime64[ns]")
        ends = f["fixed_at"].to_numpy(dtype="datetime64[ns]")

        # Clamp open-ended / future-ended faults to telemetry range
        ends = np.minimum(ends, telemetry_end)

        diff = np.zeros(len(ts) + 1, dtype=np.int32)

        start_idx = np.searchsorted(ts, starts, side="left")
        end_idx = np.searchsorted(ts, ends, side="right")

        np.add.at(diff, start_idx, 1)
        np.add.at(diff, end_idx, -1)

        active = (np.cumsum(diff[:-1]) > 0).astype("int8")
        df[col] = active

    df["fault_any"] = df[fault_cols].max(axis=1).astype("int8")
    return df