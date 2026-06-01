# features/cell_pipeline.py
"""
Cell Gold pipeline entry point — one vehicle, one date partition.

Flow:
  silver_cells
      ↓
  Quality filter (GOLD_SAFE_MASK)
      ↓
  cell_features    (per-cell delta, zscore, is_outlier)
      ↓
  module_features  (per-module voltage + temp aggregations)
      ↓
  pack_features    (pack-level aggregations)
      ↓
  Write gold_cells/cell_features/
  Write gold_cells/module_features/
  Write gold_cells/pack_features/
  Write manifest JSON

Silver_cells is never modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ingestion.io import write_parquet_atomic, parquet_path
from features.cell_health import compute_cell_features
from features.cell_agg import compute_agg_features


# -------------------------------------------------
# YAML loader
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Quality filter
# -------------------------------------------------

def apply_gold_filter(
    df: pd.DataFrame,
    gold_safe_mask: int,
) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that fail GOLD_SAFE_MASK.
    Returns (filtered_df, n_dropped).
    """
    if "cell_quality_flag" not in df.columns:
        return df.copy(), 0

    safe = (df["cell_quality_flag"] & gold_safe_mask) == 0
    dropped = int((~safe).sum())
    return df[safe].copy().reset_index(drop=True), dropped


# -------------------------------------------------
# Load silver_cells for one vehicle one date
# -------------------------------------------------

def load_silver_cells(
    silver_cells_dir: Path,
    dt: str,
    vehicle_id: str,
) -> pd.DataFrame:
    path = parquet_path(silver_cells_dir, dt, vehicle_id)
    if not path.exists():
        raise FileNotFoundError(
            f"silver_cells not found: {path}\n"
            f"Run: python run_cells.py ingest --dt {dt} first."
        )
    return pd.read_parquet(path)


# -------------------------------------------------
# List available vehicles for a date
# -------------------------------------------------

def list_silver_cell_vehicles(
    silver_cells_dir: Path,
    dt: str,
) -> list[str]:
    d = silver_cells_dir / f"dt={dt}"
    if not d.exists():
        return []
    return [
        p.stem.replace("vehicle_id=", "")
        for p in sorted(d.glob("vehicle_id=*.parquet"))
    ]


# -------------------------------------------------
# Write gold output
# -------------------------------------------------

def write_gold(
    df: pd.DataFrame,
    base_dir: Path,
    dt: str,
    vehicle_id: str,
) -> Path:
    """
    Write a gold_cells dataframe partition atomically.
    Returns output path.
    """
    out = parquet_path(base_dir, dt, vehicle_id)
    write_parquet_atomic(df, out)
    return out


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def run_cell_gold_for_day(
    dt: str,
    silver_cells_dir: Path,
    gold_cell_features_dir: Path,
    gold_module_features_dir: Path,
    gold_pack_features_dir: Path,
    voltage_schema_path: Path,
    temp_schema_path: Path,
    cell_cfg_path: Path,
    cell_quality_flags_path: Path,
    reports_dir: Path,
    vehicle_id_filter: str | None = None,
) -> dict:
    """
    Run cell gold pipeline for all vehicles on a given date.

    Returns manifest dict.
    """

    qf = load_yaml(cell_quality_flags_path)
    cfg = load_yaml(cell_cfg_path)

    gold_safe_mask = int(qf["GOLD_SAFE_MASK"])
    min_rows_for_gold = int(cfg["health_report"]["min_rows_for_report"])

    vehicles = list_silver_cell_vehicles(silver_cells_dir, dt)
    if vehicle_id_filter:
        vehicles = [v for v in vehicles if v == vehicle_id_filter]

    if not vehicles:
        raise FileNotFoundError(
            f"No silver_cells found for dt={dt} under {silver_cells_dir}.\n"
            f"Run: python run_cells.py ingest --dt {dt} first."
        )

    manifest = {
        "dt": dt,
        "pipeline": "cell_gold",
        "vehicles_processed": [],
        "per_vehicle": {},
        "status": "success",
    }

    for vehicle_id in vehicles:
        # -------------------------------------------------
        # Load silver_cells
        # -------------------------------------------------

        silver = load_silver_cells(silver_cells_dir, dt, vehicle_id)
        silver_rows = len(silver)

        # -------------------------------------------------
        # Quality filter
        # -------------------------------------------------

        filtered, rows_dropped = apply_gold_filter(silver, gold_safe_mask)

        if len(filtered) < min_rows_for_gold:
            manifest["per_vehicle"][vehicle_id] = {
                "status": "skipped",
                "reason": (
                    f"Only {len(filtered)} rows after quality filter "
                    f"(min={min_rows_for_gold})"
                ),
                "silver_rows": silver_rows,
                "rows_dropped": rows_dropped,
            }
            continue

        # -------------------------------------------------
        # Cell features
        # -------------------------------------------------

        cell_feat = compute_cell_features(
            silver=filtered,
            voltage_schema_path=voltage_schema_path,
            cell_cfg_path=cell_cfg_path,
        )

        # -------------------------------------------------
        # Module + pack features
        # -------------------------------------------------

        module_feat, pack_feat = compute_agg_features(
            silver=filtered,
            voltage_schema_path=voltage_schema_path,
            temp_schema_path=temp_schema_path,
            cell_cfg_path=cell_cfg_path,
        )

        # -------------------------------------------------
        # Write gold outputs
        # -------------------------------------------------

        cell_path = write_gold(
            cell_feat,
            gold_cell_features_dir,
            dt,
            vehicle_id,
        )

        module_path = write_gold(
            module_feat,
            gold_module_features_dir,
            dt,
            vehicle_id,
        )

        pack_path = write_gold(
            pack_feat,
            gold_pack_features_dir,
            dt,
            vehicle_id,
        )

        # -------------------------------------------------
        # Manifest
        # -------------------------------------------------

        manifest["vehicles_processed"].append(vehicle_id)

        manifest["per_vehicle"][vehicle_id] = {
            "status": "ok",
            "silver_rows": silver_rows,
            "rows_dropped_filter": rows_dropped,
            "gold_rows": int(len(filtered)),
            "cell_features_rows": int(len(cell_feat)),
            "module_features_rows": int(len(module_feat)),
            "pack_features_rows": int(len(pack_feat)),
            "cell_features_path": str(cell_path),
            "module_features_path": str(module_path),
            "pack_features_path": str(pack_path),
            "imbalance_flag_count": int(
                pack_feat["imbalance_flag"].sum()
            ) if "imbalance_flag" in pack_feat.columns else 0,
        }

    # -------------------------------------------------
    # Write manifest JSON
    # -------------------------------------------------

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"dt={dt}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[gold_cells] dt={dt} "
        f"vehicles={len(manifest['vehicles_processed'])} "
        f"report={report_path}"
    )

    return manifest