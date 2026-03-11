from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq


# ── Paths ──────────────────────────────────────────────────────────────────────

def _gold_dir() -> Path:
    try:
        from configs.settings import GOLD_DIR
        return Path(GOLD_DIR) / "window_features"
    except Exception:
        return Path("data/gold/window_features")


def _parquet_files(dt: str, vehicle_id: str) -> List[Path]:
    base = _gold_dir() / f"dt={dt}" / f"vehicle_id={vehicle_id}"
    return sorted(base.glob("*.parquet"))


# ── Derived features ───────────────────────────────────────────────────────────

def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features not stored in Gold.
    Requires battery_current_a and motor_speed_rpm.
    """
    out = df.copy()

    required = ["battery_current_a", "motor_speed_rpm"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Cannot compute derived charging features; missing columns: {missing}"
        )

    out["is_charging_current"] = (out["battery_current_a"] > 0).astype("int8")
    out["is_parked_charging"] = (
        (out["battery_current_a"] > 5)
        & (out["motor_speed_rpm"].abs() < 50)
    ).astype("int8")

    return out


def _schema_cols(parquet_path: Path) -> set[str]:
    return set(pq.read_schema(parquet_path).names)


# ── Core loader ────────────────────────────────────────────────────────────────

def load_gold_dates(
    dates: List[str],
    vehicles: List[str],
    features: List[str],
    target: str,
    drop_quality_mask: int = 52,
    label_col: str = "label_available",
    label_value: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate Gold parquet files for the given dates + vehicles.

    Returns a single DataFrame with features + target + trip_id + timestamp.
    Applies quality filter, label filter, and derived feature engineering.
    """

    # These are needed either directly or for deriving charging features
    meta_cols = [
        "timestamp",
        "trip_id",
        "label_available",
        "quality_flag",
        "battery_current_a",
        "motor_speed_rpm",
        "soc_pct",
    ]

    read_cols_requested = list(dict.fromkeys(features + [target] + meta_cols))
    frames = []

    for dt in dates:
        for vehicle in vehicles:
            files = _parquet_files(dt, vehicle)

            if not files:
                if verbose:
                    print(f"  WARNING: no gold files for {dt} / {vehicle}")
                continue

            file_frames = []
            for f in files:
                schema = _schema_cols(f)
                cols = [c for c in read_cols_requested if c in schema]
                if not cols:
                    continue
                file_frames.append(pd.read_parquet(f, columns=cols))

            if not file_frames:
                if verbose:
                    print(f"  WARNING: no readable parquet columns for {dt} / {vehicle}")
                continue

            df = pd.concat(file_frames, ignore_index=True)

            if label_col in df.columns:
                df = df[df[label_col] == label_value]

            if "quality_flag" in df.columns:
                df = df[(df["quality_flag"] & drop_quality_mask) == 0]

            if verbose:
                trip_count = int(df["trip_id"].nunique()) if "trip_id" in df.columns else 0
                print(f"  {dt}/{vehicle}: {len(df):,} rows, {trip_count} trips")

            frames.append(df)

    if not frames:
        raise RuntimeError("No Gold data found for the given dates/vehicles.")

    combined = pd.concat(frames, ignore_index=True)

    # Derived charging features
    need_derived = any(f in {"is_charging_current", "is_parked_charging"} for f in features)
    if need_derived:
        combined = _add_derived_features(combined)

    missing_features = [f for f in features if f not in combined.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature columns in loaded Gold data: {missing_features}"
        )

    if target not in combined.columns:
        raise ValueError(f"Target column '{target}' missing in loaded Gold data.")

    combined = combined.dropna(subset=features + [target])

    if verbose:
        print(
            f"  ── total: {len(combined):,} rows, "
            f"{combined['trip_id'].nunique()} trips"
        )

    return combined


# ── Micro-trip filter ──────────────────────────────────────────────────────────

def filter_micro_trips(
    df: pd.DataFrame,
    min_rows: int = 60,
    min_soc_range: float = 3.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove trips that are too short or have negligible SOC change."""

    rows = df.groupby("trip_id").size()
    soc = df["soc_pct"]
    soc_range = soc.groupby(df["trip_id"]).max() - soc.groupby(df["trip_id"]).min()

    stats = pd.concat([rows, soc_range], axis=1)
    stats.columns = ["rows", "soc_range"]

    valid = stats[
        (stats["rows"] >= min_rows) & (stats["soc_range"] >= min_soc_range)
    ].index

    if verbose:
        skipped = stats[~stats.index.isin(valid)]
        if len(skipped) > 0:
            print(f"\n  Micro-trips removed ({len(skipped)}):")
            for tid, row in skipped.iterrows():
                print(
                    f"    {tid}  rows={int(row['rows'])}  "
                    f"soc_range={row['soc_range']:.1f}%"
                )

    return df[df["trip_id"].isin(valid)].copy()


# ── Data fingerprint ───────────────────────────────────────────────────────────

def fingerprint_gold(
    dates: List[str],
    vehicles: List[str],
) -> Dict[str, str]:
    """
    SHA-256 hash of every parquet file used for training.
    Stored in the run folder so you can verify data provenance later.
    """
    hashes: Dict[str, str] = {}
    for dt in dates:
        for vehicle in vehicles:
            for f in _parquet_files(dt, vehicle):
                key = f"dt={dt}/vehicle_id={vehicle}/{f.name}"
                hashes[key] = _sha256(f)
    return hashes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"