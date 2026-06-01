# ingestion/cell_ingest.py
"""
Cell pipeline entry point — one vehicle, one date partition.

Flow:
  raw_cell_voltage CSV  ──┐
                          ├── validate ──> merge on timestamp ──> silver_cells
  raw_cell_temp CSV     ──┘

Steps:
  1) Load raw cell voltage CSV
  2) Validate voltage (null handling, range checks, flatline)
  3) Load raw cell temp CSV
  4) Validate temp (null handling, range checks, flatline)
  5) Merge voltage + temp on timestamp (inner join)
  6) Write silver_cells (atomic)
  7) Return manifest dict

Silver_cells contains:
  timestamp, vehicle_id,
  M1_C1 ... M5_C36  (180 voltage cols),
  M1_T1 ... M5_T18  (90 temp cols),
  cell_quality_flag
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ingestion.io import (
    vehicle_id_from_filename,
    parquet_path,
    write_parquet_atomic,
)
from ingestion.cell_validators import validate_cells


# -------------------------------------------------
# List cell CSVs (voltage or temp folder)
# -------------------------------------------------

def list_cell_csvs(raw_dir: Path, dt: str) -> list[Path]:
    """List cell CSVs for a given date partition."""
    d = raw_dir / f"dt={dt}"
    if not d.exists():
        return []
    return sorted(d.glob("vehicle_id=*.csv"))


# -------------------------------------------------
# Load raw cell CSV
# -------------------------------------------------

def read_cell_csv(path: Path) -> pd.DataFrame:
    """
    Read cell voltage or temp CSV.
    No casting here — casting happens in cell_validators.
    """
    return pd.read_csv(path)


# -------------------------------------------------
# Merge voltage + temp silver frames
# -------------------------------------------------

def merge_voltage_temp(
    voltage_df: pd.DataFrame,
    temp_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner join voltage and temp on timestamp + vehicle_id.

    Both frames have already been validated and have canonical columns:
      timestamp, vehicle_id, signal_cols..., cell_quality_flag

    cell_quality_flag is combined with bitwise OR across both sources.
    """

    # Separate quality flags before merge
    v_flag = voltage_df[["timestamp", "vehicle_id", "cell_quality_flag"]].rename(
        columns={"cell_quality_flag": "_v_flag"}
    )
    t_flag = temp_df[["timestamp", "vehicle_id", "cell_quality_flag"]].rename(
        columns={"cell_quality_flag": "_t_flag"}
    )

    # Drop quality flag from both before merge to avoid collision
    voltage_df = voltage_df.drop(columns=["cell_quality_flag"])
    temp_df = temp_df.drop(columns=["cell_quality_flag"])

    # Inner join — only rows present in both
    merged = pd.merge(
        voltage_df,
        temp_df,
        on=["timestamp", "vehicle_id"],
        how="inner",
    )

    # Reattach and combine quality flags with bitwise OR
    merged = merged.merge(v_flag, on=["timestamp", "vehicle_id"], how="left")
    merged = merged.merge(t_flag, on=["timestamp", "vehicle_id"], how="left")

    merged["_v_flag"] = merged["_v_flag"].fillna(0).astype("int64")
    merged["_t_flag"] = merged["_t_flag"].fillna(0).astype("int64")
    merged["cell_quality_flag"] = merged["_v_flag"] | merged["_t_flag"]

    merged = merged.drop(columns=["_v_flag", "_t_flag"])

    # Sort by timestamp
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


# -------------------------------------------------
# Main ingest
# -------------------------------------------------

def run_cell_ingest_for_day(
    dt: str,
    raw_cell_voltage_dir: Path,
    raw_cell_temp_dir: Path,
    silver_cells_dir: Path,
    voltage_schema_path: Path,
    temp_schema_path: Path,
    cell_quality_flags_path: Path,
    cell_cfg_path: Path,
    reports_dir: Path,
    vehicle_id_filter: str | None = None,
) -> dict:
    """
    Run cell ingest for all vehicles on a given date.

    Returns manifest dict.
    """

    # Find voltage CSVs for this date — drives vehicle list
    voltage_csvs = list_cell_csvs(raw_cell_voltage_dir, dt)

    if vehicle_id_filter:
        voltage_csvs = [c for c in voltage_csvs if vehicle_id_from_filename(c) == vehicle_id_filter]

    if not voltage_csvs:
        raise FileNotFoundError(
            f"No cell voltage CSVs found under "
            f"{raw_cell_voltage_dir}/dt={dt}/ (vehicle_id_filter={vehicle_id_filter})"
        )

    manifest = {
        "dt": dt,
        "pipeline": "cell_ingest",
        "vehicles_processed": [],
        "per_vehicle": {},
        "status": "success",
    }

    for voltage_path in voltage_csvs:
        vehicle_id = vehicle_id_from_filename(voltage_path)

        # -------------------------------------------------
        # Locate matching temp CSV
        # -------------------------------------------------

        temp_path = (
            raw_cell_temp_dir
            / f"dt={dt}"
            / f"vehicle_id={vehicle_id}.csv"
        )

        if not temp_path.exists():
            manifest["per_vehicle"][vehicle_id] = {
                "status": "skipped",
                "reason": f"No temp CSV found at {temp_path}",
            }
            continue

        # -------------------------------------------------
        # Load
        # -------------------------------------------------

        voltage_raw = read_cell_csv(voltage_path)
        temp_raw = read_cell_csv(temp_path)

        # -------------------------------------------------
        # Validate voltage
        # -------------------------------------------------

        v_result = validate_cells(
            df=voltage_raw,
            data_type="voltage",
            voltage_schema_path=voltage_schema_path,
            temp_schema_path=temp_schema_path,
            cell_quality_flags_path=cell_quality_flags_path,
            cell_cfg_path=cell_cfg_path,
            vehicle_id=vehicle_id,
        )

        # -------------------------------------------------
        # Validate temp
        # -------------------------------------------------

        t_result = validate_cells(
            df=temp_raw,
            data_type="temp",
            voltage_schema_path=voltage_schema_path,
            temp_schema_path=temp_schema_path,
            cell_quality_flags_path=cell_quality_flags_path,
            cell_cfg_path=cell_cfg_path,
            vehicle_id=vehicle_id,
        )

        # -------------------------------------------------
        # Merge voltage + temp → silver_cells
        # -------------------------------------------------

        silver = merge_voltage_temp(
            voltage_df=v_result.df,
            temp_df=t_result.df,
        )

        rows_before_merge = min(len(v_result.df), len(t_result.df))
        rows_after_merge = len(silver)
        rows_dropped = rows_before_merge - rows_after_merge

        # -------------------------------------------------
        # Write silver_cells (atomic)
        # -------------------------------------------------

        silver_out = parquet_path(
            silver_cells_dir,
            dt,
            vehicle_id,
        )

        write_parquet_atomic(silver, silver_out)

        # -------------------------------------------------
        # Manifest
        # -------------------------------------------------

        manifest["vehicles_processed"].append(vehicle_id)

        manifest["per_vehicle"][vehicle_id] = {
            "status": "ok",
            "silver_cells_path": str(silver_out),
            "voltage_rows": int(len(v_result.df)),
            "temp_rows": int(len(t_result.df)),
            "silver_rows": int(len(silver)),
            "rows_dropped_merge": int(rows_dropped),
            "data_start_ts": str(silver["timestamp"].min()) if len(silver) else None,
            "data_end_ts": str(silver["timestamp"].max()) if len(silver) else None,
            "voltage_validation": v_result.report,
            "temp_validation": t_result.report,
            "quality_flag_counts": {
                "CELL_FFILLED": int(((silver["cell_quality_flag"] & 1) != 0).sum()),
                "MODULE_GAP": int(((silver["cell_quality_flag"] & 2) != 0).sum()),
                "PACK_GAP": int(((silver["cell_quality_flag"] & 4) != 0).sum()),
                "CELL_SOFT_BREACH": int(((silver["cell_quality_flag"] & 8) != 0).sum()),
                "CELL_HARD_BREACH": int(((silver["cell_quality_flag"] & 16) != 0).sum()),
                "CELL_FLATLINE": int(((silver["cell_quality_flag"] & 32) != 0).sum()),
                "CELL_OUTLIER": int(((silver["cell_quality_flag"] & 64) != 0).sum()),
            },
        }

    # -------------------------------------------------
    # Write manifest JSON
    # -------------------------------------------------

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"dt={dt}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[OK] cell_ingest dt={dt} "
        f"vehicles={len(manifest['vehicles_processed'])} "
        f"report={report_path}"
    )

    return manifest