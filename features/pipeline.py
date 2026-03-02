from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pyarrow.parquet as pq  # <-- NEW

from features.lags import lag_features
from features.physics import physics_features
from features.rolling import rolling_features
from features.trip_agg import trip_aggregations
from features.utils import (
    ensure_sorted_1hz,
    load_yaml,
    stable_column_order,
    try_import_atomic_parquet_writer,
    write_json_atomic,
    atomic_dir_tmp,
    atomic_dir_commit,
)

# ===============================
# Paths
# ===============================

@dataclass(frozen=True)
class GoldPaths:
    window_dir: Path
    trip_dir: Path
    daily_dir: Path
    report_dir: Path


def _get_paths() -> GoldPaths:
    try:
        from configs.settings import GOLD_DIR, REPORTS_DIR
        gold_dir = Path(GOLD_DIR)
        reports_dir = Path(REPORTS_DIR)
    except Exception:
        gold_dir = Path("data/gold")
        reports_dir = Path("data/reports")

    return GoldPaths(
        window_dir=gold_dir / "window_features",
        trip_dir=gold_dir / "trip_features",
        daily_dir=gold_dir / "daily_stats",
        report_dir=reports_dir / "gold",
    )


def _silver_path(dt: str, vehicle_id: str) -> Path:
    try:
        from configs.settings import SILVER_DIR
        silver_dir = Path(SILVER_DIR)
    except Exception:
        silver_dir = Path("data/silver")

    return silver_dir / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"


def _partition_path(base: Path, dt: str, vehicle_id: str) -> Path:
    return base / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"


def _window_out_dir(base: Path, dt: str, vehicle_id: str) -> Path:
    return base / f"dt={dt}" / f"vehicle_id={vehicle_id}"


# ===============================
# Main Pipeline
# ===============================

def build_gold_for_vehicle_day(dt: str, vehicle_id: str) -> Dict[str, Any]:
    cfg = load_yaml(Path("configs/gold.yaml"))
    paths = _get_paths()
    write_parquet_atomic = try_import_atomic_parquet_writer()

    silver_path = _silver_path(dt, vehicle_id)
    if not silver_path.exists():
        raise FileNotFoundError(f"Silver parquet not found: {silver_path}")

    # ---------------------------
    # Feature Config
    # ---------------------------
    base_signals = list(cfg.get("base_signals", []))
    windows_s = list(cfg.get("rolling_windows_s", [30, 60, 300, 600]))
    aggs = list(cfg.get("rolling_aggs", ["mean", "std", "min", "max"]))
    lags_s = list(cfg.get("lag_seconds", [60, 300, 600]))

    target_cfg = cfg.get("target", {}) or {}
    target_name = str(target_cfg.get("name", "y_soc_t_plus_300s"))
    horizon_s = int(target_cfg.get("horizon_s", 300))
    src_col = str(target_cfg.get("source_col", "soc_pct"))

    drop_mask = int(cfg.get("quality_filter", {}).get("drop_mask", 48))
    trip_min_rows = int(cfg.get("trip", {}).get("min_rows", 60))
    daily_min_rows = int(cfg.get("daily", {}).get("min_rows", 60))

    # ---------------------------
    # Load Silver (COLUMN PROJECTION, SCHEMA-SAFE)
    # ---------------------------
    desired_cols = ["timestamp", "vehicle_id", "trip_id", "quality_flag", "fault_any"]
    desired_cols += base_signals
    desired_cols = list(dict.fromkeys([c for c in desired_cols if c]))

    schema_cols = set(pq.read_schema(silver_path).names)
    read_cols = [c for c in desired_cols if c in schema_cols]

    df = pd.read_parquet(silver_path, columns=read_cols)

    # If no fault file existed that day, silver may not have fault_any
    if "fault_any" not in df.columns:
        df["fault_any"] = 0

    df = ensure_sorted_1hz(df)

    required = ["vehicle_id", "trip_id", "quality_flag", "timestamp"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Silver missing required column '{col}'")

    if src_col not in df.columns:
        raise ValueError(f"Target source column '{src_col}' not found in silver (after projection).")

    # ---------------------------
    # Downcast to reduce RAM
    # ---------------------------
    float64_cols = df.select_dtypes(include=["float64"]).columns
    if len(float64_cols) > 0:
        df[float64_cols] = df[float64_cols].astype("float32")

    df["quality_flag"] = df["quality_flag"].fillna(0).astype("uint32")

    before_rows = len(df)

    # ---------------------------
    # Quality Filter
    # ---------------------------
    mask = (df["quality_flag"] & drop_mask) == 0
    df = df.loc[mask]
    df = df.loc[df["trip_id"].notna()]

    after_rows = len(df)
    if after_rows == 0:
        raise RuntimeError("All rows removed by quality filter.")

    # ---------------------------
    # Stream Features per Trip -> write dataset parts
    # ---------------------------
    out_window_final = _window_out_dir(paths.window_dir, dt, vehicle_id)
    out_window_tmp = atomic_dir_tmp(out_window_final)

    part_idx = 0
    total_window_rows = 0
    trip_frames = []
    seen_trips = 0

    for trip_id, g in df.groupby("trip_id", sort=False):
        g = ensure_sorted_1hz(g)
        if len(g) == 0:
            continue

        g_feat = rolling_features(g, signals=base_signals, windows_s=windows_s, aggs=aggs)
        g_feat = physics_features(g_feat)
        g_feat = lag_features(g_feat, signals=base_signals, lags_s=lags_s)

        future_soc = g_feat[src_col].shift(-horizon_s)
        g_feat[target_name] = future_soc
        g_feat["label_available"] = future_soc.notna().astype("int64")

        delta = (future_soc - g_feat[src_col]).abs()
        valid_rows = delta.isna() | (delta <= 25)
        g_feat = g_feat.loc[valid_rows]

        g_feat = stable_column_order(
            g_feat,
            first=[
                "timestamp",
                "vehicle_id",
                "trip_id",
                "quality_flag",
                "fault_any",
                src_col,
                target_name,
                "label_available",
            ],
        )

        part_idx += 1
        part_path = out_window_tmp / f"part-{part_idx:06d}.parquet"
        g_feat.to_parquet(part_path, index=False)

        total_window_rows += int(len(g_feat))
        seen_trips += 1

        df_trip_one = trip_aggregations(g_feat, min_rows=trip_min_rows)
        if not df_trip_one.empty:
            trip_frames.append(df_trip_one)

    atomic_dir_commit(out_window_tmp, out_window_final)

    # ---------------------------
    # Trip Features
    # ---------------------------
    df_trip = pd.concat(trip_frames, ignore_index=True) if trip_frames else pd.DataFrame()

    out_trip = _partition_path(paths.trip_dir, dt, vehicle_id)
    write_parquet_atomic(df_trip, out_trip)

    # ---------------------------
    # Daily Stats
    # ---------------------------
    daily = {
        "dt": dt,
        "vehicle_id": vehicle_id,
        "silver_rows": int(before_rows),
        "rows_after_quality_filter": int(after_rows),
        "dropped_rows": int(before_rows - after_rows),
        "drop_pct": float((before_rows - after_rows) / max(1, before_rows)),
        "window_feature_rows": int(total_window_rows),
        "trip_count": int(df_trip["trip_id"].nunique()) if not df_trip.empty else 0,
        "has_enough_rows": int(total_window_rows >= daily_min_rows),
        "trip_groups_processed": int(seen_trips),
        "window_parts_written": int(part_idx),
    }

    df_daily = pd.DataFrame([daily])
    out_daily = _partition_path(paths.daily_dir, dt, vehicle_id)
    write_parquet_atomic(df_daily, out_daily)

    # ---------------------------
    # Manifest
    # ---------------------------
    manifest: Dict[str, Any] = {
        "dt": dt,
        "vehicle_id": vehicle_id,
        "inputs": {"silver_path": str(silver_path)},
        "outputs": {
            "window_features_dir": str(out_window_final),
            "trip_features": str(out_trip),
            "daily_stats": str(out_daily),
        },
        "counts": {
            "silver_rows": int(before_rows),
            "rows_after_quality_filter": int(after_rows),
            "window_rows": int(total_window_rows),
            "trip_rows": int(len(df_trip)),
            "window_parts": int(part_idx),
            "trip_groups_processed": int(seen_trips),
        },
    }

    report_path = paths.report_dir / f"dt={dt}" / f"vehicle_id={vehicle_id}.json"
    write_json_atomic(report_path, manifest)

    return manifest