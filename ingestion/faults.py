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
import pandas as pd
from pathlib import Path


# -------------------------------------------------
# Load fault CSV
# -------------------------------------------------

def load_fault_csv(path: Path) -> pd.DataFrame:

    if not path.exists():
        return pd.DataFrame()

    faults = pd.read_csv(path)

    faults = faults.rename(columns={
        "Activated At": "activated_at",
        "Fixed At": "fixed_at",
        "Code": "code"
    })

    faults["activated_at"] = pd.to_datetime(
        faults["activated_at"],
        dayfirst=True,
        utc=True
    )

    faults["fixed_at"] = pd.to_datetime(
        faults["fixed_at"],
        dayfirst=True,
        utc=True
    )

    faults = faults.sort_values("activated_at")

    return faults


# -------------------------------------------------
# Add fault flags
# -------------------------------------------------

def add_fault_flags(df: pd.DataFrame, faults: pd.DataFrame) -> pd.DataFrame:

    if faults.empty:
        return df

    df = df.copy()

    # Unique fault codes
    codes = sorted(faults["code"].unique())

    # Create columns
    for code in codes:

        col = f"fault_{code}"

        df[col] = 0

        f = faults[faults["code"] == code]

        for _, row in f.iterrows():

            mask = (
                (df["timestamp"] >= row["activated_at"])
                &
                (df["timestamp"] <= row["fixed_at"])
            )

            df.loc[mask, col] = 1


    # -------------------------------------------------
    # fault_any column
    # -------------------------------------------------

    fault_cols = [c for c in df.columns if c.startswith("fault_")]

    df["fault_any"] = df[fault_cols].max(axis=1)

    return df