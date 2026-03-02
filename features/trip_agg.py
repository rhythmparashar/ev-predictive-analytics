# features/trip_agg.py
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def trip_aggregations(df: pd.DataFrame, min_rows: int = 60) -> pd.DataFrame:
    """
    One row per trip_id. Assumes df includes timestamp, trip_id, vehicle_id.
    """
    required = ["timestamp", "trip_id", "vehicle_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trip_aggregations missing required columns: {missing}")

    g = df.groupby("trip_id", sort=False)

    rows = []
    for trip_id, t in g:
        if len(t) < min_rows:
            continue
        t = t.sort_values("timestamp", kind="mergesort")

        rec: Dict[str, object] = {
            "trip_id": trip_id,
            "vehicle_id": t["vehicle_id"].iloc[0],
            "trip_start_ts": t["timestamp"].iloc[0],
            "trip_end_ts": t["timestamp"].iloc[-1],
            "duration_s": int((t["timestamp"].iloc[-1] - t["timestamp"].iloc[0]).total_seconds()),
            "rows": int(len(t)),
        }

        # SOC start/end
        if "soc" in t.columns:
            rec["soc_start"] = float(t["soc"].iloc[0]) if pd.notna(t["soc"].iloc[0]) else None
            rec["soc_end"] = float(t["soc"].iloc[-1]) if pd.notna(t["soc"].iloc[-1]) else None
            if rec["soc_start"] is not None and rec["soc_end"] is not None:
                rec["soc_delta"] = float(rec["soc_end"]) - float(rec["soc_start"])

        # Fault seconds
        if "fault_any" in t.columns:
            rec["fault_seconds"] = int(t["fault_any"].fillna(0).astype("int64").sum())

        # Temperature max
        for col in ["motor_temp_c", "module_temp_max_c", "module_temp_delta_c"]:
            if col in t.columns:
                rec[f"{col}_max"] = float(t[col].max(skipna=True)) if t[col].notna().any() else None

        # Energy proxy if power exists
        if "dc_power_kw" in t.columns:
            # kWh = sum(kW) / 3600 for 1 Hz
            rec["energy_kwh_proxy"] = float(t["dc_power_kw"].fillna(0.0).sum() / 3600.0)

        rows.append(rec)

    return pd.DataFrame(rows)