# build_master_dataset.py
# Build analysis-ready master dataset by joining:
#   - VCU silver
#   - cell pack_features
#   - cell module_features
#
# Output:
#   data/analysis/master/dt={partition}/vehicle_id=EV01.parquet
#
# Important:
# - Uses shared folder partitions as the build unit
# - Does NOT split chunked partitions into individual calendar days
# - Deduplicates timestamp+vehicle_id keys before merge
# - Logs duplicate-removal counts and merge retention

from __future__ import annotations

import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd


# ── Config ──────────────────────────────────────────────────────────────
VCU_SILVER = Path("data/silver")
CELL_PACK = Path("data/gold_cells/pack_features")
CELL_MODULE = Path("data/gold_cells/module_features")
OUT_DIR = Path("data/analysis/master")

_parser = argparse.ArgumentParser()
_parser.add_argument("--vehicle-id", default="EV01", help="Vehicle to build master for (e.g. EV02)")
VEHICLE_ID = _parser.parse_args().vehicle_id

# Update if your flag bit definitions differ
PACK_GAP = 4
CELL_HARD_BREACH = 16
SAFE_MASK = PACK_GAP | CELL_HARD_BREACH

IDLE_CURRENT_A = 10
HEAVY_CURRENT_A = 100

VCU_COLS = [
    "timestamp",
    "vehicle_id",
    "trip_id",
    "soc_pct",
    "battery_current_a",
    "stack_voltage_v",
    "output_power_kw",
    "max_cell_voltage_v",
    "min_cell_voltage_v",
    "avg_cell_voltage_v",
    "avg_battery_temp_c",
    "max_battery_temp_c",
    "min_battery_temp_c",
    "motor_speed_rpm",
    "motor_torque_value_nm",
    "motor_temperature_c",
    "battery_status",
    "fault_any",
    "quality_flag",
]

PACK_COLS = [
    "timestamp",
    "vehicle_id",
    "cell_quality_flag",
    "pack_voltage_range",
    "pack_voltage_std",
    "worst_cell_id",
    "imbalance_flag",
    "pack_temp_range",
    "hottest_module_id",
]

MODULE_COLS = [
    "timestamp",
    "vehicle_id",
    "cell_quality_flag",
    "module_1_voltage_mean",
    "module_1_voltage_std",
    "module_1_voltage_range",
    "module_1_lowest_cell",
    "module_1_temp_mean",
    "module_1_temp_range",
    "module_1_hottest_sensor",
    "module_2_voltage_mean",
    "module_2_voltage_std",
    "module_2_voltage_range",
    "module_2_lowest_cell",
    "module_2_temp_mean",
    "module_2_temp_range",
    "module_2_hottest_sensor",
    "module_3_voltage_mean",
    "module_3_voltage_std",
    "module_3_voltage_range",
    "module_3_lowest_cell",
    "module_3_temp_mean",
    "module_3_temp_range",
    "module_3_hottest_sensor",
    "module_4_voltage_mean",
    "module_4_voltage_std",
    "module_4_voltage_range",
    "module_4_lowest_cell",
    "module_4_temp_mean",
    "module_4_temp_range",
    "module_4_hottest_sensor",
    "module_5_voltage_mean",
    "module_5_voltage_std",
    "module_5_voltage_range",
    "module_5_lowest_cell",
    "module_5_temp_mean",
    "module_5_temp_range",
    "module_5_hottest_sensor",
]


# ── Helpers ─────────────────────────────────────────────────────────────
def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.tz_localize(None)
    return out


def current_direction(series: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            series < -IDLE_CURRENT_A,
            series > IDLE_CURRENT_A,
        ],
        [
            "charge",
            "discharge",
        ],
        default="idle",
    )
    return pd.Categorical(
        values,
        categories=["charge", "idle", "discharge"],
        ordered=True,
    )


def operating_regime(series: pd.Series) -> pd.Categorical:
    values = np.select(
        [
            series <= -HEAVY_CURRENT_A,
            (series > -HEAVY_CURRENT_A) & (series < -IDLE_CURRENT_A),
            series.between(-IDLE_CURRENT_A, IDLE_CURRENT_A, inclusive="both"),
            (series > IDLE_CURRENT_A) & (series < HEAVY_CURRENT_A),
            series >= HEAVY_CURRENT_A,
        ],
        [
            "heavy_regen",
            "light_regen",
            "idle",
            "light_discharge",
            "heavy_discharge",
        ],
        default="idle",
    )
    return pd.Categorical(
        values,
        categories=[
            "heavy_regen",
            "light_regen",
            "idle",
            "light_discharge",
            "heavy_discharge",
        ],
        ordered=True,
    )


def soc_band(series: pd.Series) -> pd.Categorical:
    return pd.cut(
        series,
        bins=[0, 20, 40, 60, 80, 100],
        labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
        include_lowest=True,
        ordered=True,
    )


def find_partition_dates(base: Path, vehicle_id: str) -> set[str]:
    return {
        p.parent.name.replace("dt=", "")
        for p in base.glob(f"dt=*/vehicle_id={vehicle_id}.parquet")
    }


def load_partition(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    df = pd.read_parquet(path, columns=columns)
    df = normalize_ts(df)

    bad_ts = int(df["timestamp"].isna().sum())
    if bad_ts:
        raise ValueError(f"{label} has {bad_ts} invalid timestamps")

    return df


def deduplicate_on_keys(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """
    Drop duplicate timestamp+vehicle_id rows, keeping the first.
    Returns deduplicated dataframe and number of removed rows.
    """
    before = len(df)
    df = df.sort_values(["timestamp", "vehicle_id"]).drop_duplicates(
        subset=["timestamp", "vehicle_id"],
        keep="first",
    ).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  [WARN] {label}: removed {removed:,} duplicate key rows")
    return df, removed


# ── Find shared partitions ──────────────────────────────────────────────
vcu_dates = find_partition_dates(VCU_SILVER, VEHICLE_ID)
pack_dates = find_partition_dates(CELL_PACK, VEHICLE_ID)
module_dates = find_partition_dates(CELL_MODULE, VEHICLE_ID)

dates = sorted(vcu_dates & pack_dates & module_dates)

print(f"Shared partitions found: {len(dates)}")
for d in dates:
    print(f"  {d}")

if not dates:
    raise SystemExit("No shared partitions found. Nothing to build.")


# ── Build master partitions ─────────────────────────────────────────────
results: list[dict] = []

for partition_dt in dates:
    print(f"\nBuilding partition {partition_dt} ...")

    try:
        vcu_path = VCU_SILVER / f"dt={partition_dt}" / f"vehicle_id={VEHICLE_ID}.parquet"
        pack_path = CELL_PACK / f"dt={partition_dt}" / f"vehicle_id={VEHICLE_ID}.parquet"
        module_path = CELL_MODULE / f"dt={partition_dt}" / f"vehicle_id={VEHICLE_ID}.parquet"

        vcu = load_partition(vcu_path, VCU_COLS, f"VCU silver [{partition_dt}]")
        pack = load_partition(pack_path, PACK_COLS, f"cell pack_features [{partition_dt}]")
        module = load_partition(module_path, MODULE_COLS, f"cell module_features [{partition_dt}]")

        raw_vcu_rows = len(vcu)
        raw_pack_rows = len(pack)
        raw_module_rows = len(module)

        # Deduplicate keys before merge
        vcu, vcu_dupes_removed = deduplicate_on_keys(vcu, f"VCU silver [{partition_dt}]")
        pack, pack_dupes_removed = deduplicate_on_keys(pack, f"cell pack_features [{partition_dt}]")
        module, module_dupes_removed = deduplicate_on_keys(module, f"cell module_features [{partition_dt}]")

        vcu_rows = len(vcu)
        pack_rows = len(pack)
        module_rows = len(module)

        if vcu_rows == 0 or pack_rows == 0 or module_rows == 0:
            raise ValueError(
                f"Empty source after dedup "
                f"(vcu={vcu_rows}, pack={pack_rows}, module={module_rows})"
            )

        # Rename ambiguous quality columns
        vcu = vcu.rename(columns={"quality_flag": "vcu_quality_flag"})
        pack = pack.rename(columns={"cell_quality_flag": "cell_quality_flag_pack"})
        module = module.rename(columns={"cell_quality_flag": "cell_quality_flag_module"})

        # Merge on timestamp + vehicle_id
        merged = vcu.merge(pack, on=["timestamp", "vehicle_id"], how="inner")
        merged = merged.merge(module, on=["timestamp", "vehicle_id"], how="inner")
        merged = merged.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)

        merged_rows = len(merged)
        if merged_rows == 0:
            raise ValueError("Merged result has 0 rows")

        # Canonical cell quality flag from pack layer
        merged["cell_quality_flag"] = merged["cell_quality_flag_pack"]

        # Normalize battery_status blanks
        merged["battery_status"] = (
            merged["battery_status"]
            .replace("", pd.NA)
            .fillna("unknown")
        )

        # Derived columns
        merged["dt"] = partition_dt
        merged["cell_spread_v"] = merged["max_cell_voltage_v"] - merged["min_cell_voltage_v"]
        merged["abs_current_a"] = merged["battery_current_a"].abs()
        merged["current_direction"] = current_direction(merged["battery_current_a"])
        merged["soc_band"] = soc_band(merged["soc_pct"])
        merged["operating_regime"] = operating_regime(merged["battery_current_a"])
        merged["is_clean"] = (
            merged["cell_quality_flag"].fillna(0).astype("int64") & SAFE_MASK
        ) == 0

        # Drop duplicate quality columns now that canonical flag is created
        merged = merged.drop(
            columns=["cell_quality_flag_pack", "cell_quality_flag_module"],
            errors="ignore",
        )

        # Write output
        out_path = OUT_DIR / f"dt={partition_dt}" / f"vehicle_id={VEHICLE_ID}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)

        clean_rows = int(merged["is_clean"].sum())
        clean_pct = round((clean_rows / merged_rows) * 100, 2)

        result = {
            "dt": partition_dt,
            "status": "OK",
            "raw_vcu_rows": raw_vcu_rows,
            "raw_pack_rows": raw_pack_rows,
            "raw_module_rows": raw_module_rows,
            "vcu_dupes_removed": vcu_dupes_removed,
            "pack_dupes_removed": pack_dupes_removed,
            "module_dupes_removed": module_dupes_removed,
            "vcu_rows": vcu_rows,
            "pack_rows": pack_rows,
            "module_rows": module_rows,
            "merged_rows": merged_rows,
            "clean_rows": clean_rows,
            "clean_pct": clean_pct,
            "retention_vs_vcu_pct": round((merged_rows / vcu_rows) * 100, 2) if vcu_rows else 0.0,
            "retention_vs_pack_pct": round((merged_rows / pack_rows) * 100, 2) if pack_rows else 0.0,
            "retention_vs_module_pct": round((merged_rows / module_rows) * 100, 2) if module_rows else 0.0,
            "min_timestamp": merged["timestamp"].min().isoformat(),
            "max_timestamp": merged["timestamp"].max().isoformat(),
            "output_path": str(out_path),
        }
        results.append(result)

        print(
            f"  [OK] rows={merged_rows:,} | clean={clean_pct:.2f}% | "
            f"dupes_removed(vcu/pack/module)="
            f"{vcu_dupes_removed:,}/{pack_dupes_removed:,}/{module_dupes_removed:,} | "
            f"retention(vcu/pack/module)="
            f"{result['retention_vs_vcu_pct']:.2f}%/"
            f"{result['retention_vs_pack_pct']:.2f}%/"
            f"{result['retention_vs_module_pct']:.2f}%"
        )

    except Exception as e:
        results.append(
            {
                "dt": partition_dt,
                "status": f"FAIL: {e}",
                "raw_vcu_rows": 0,
                "raw_pack_rows": 0,
                "raw_module_rows": 0,
                "vcu_dupes_removed": 0,
                "pack_dupes_removed": 0,
                "module_dupes_removed": 0,
                "vcu_rows": 0,
                "pack_rows": 0,
                "module_rows": 0,
                "merged_rows": 0,
                "clean_rows": 0,
                "clean_pct": 0.0,
                "retention_vs_vcu_pct": 0.0,
                "retention_vs_pack_pct": 0.0,
                "retention_vs_module_pct": 0.0,
                "min_timestamp": None,
                "max_timestamp": None,
                "output_path": None,
            }
        )
        print(f"  [FAIL] {e}")


# ── Summary + manifest ──────────────────────────────────────────────────
summary = pd.DataFrame(results)

ok_mask = summary["status"].eq("OK")
built_count = int(ok_mask.sum())
total_count = len(summary)

print("\n" + "=" * 90)
print(f"Built partitions: {built_count} / {total_count}")

if built_count:
    print(f"Total merged rows: {int(summary.loc[ok_mask, 'merged_rows'].sum()):,}")
    print(f"Total clean rows:  {int(summary.loc[ok_mask, 'clean_rows'].sum()):,}")

print("\nBuild summary:")
print(summary.to_string(index=False))

manifest = {
    "built_at": dt.datetime.now().isoformat(),
    "vehicle_id": VEHICLE_ID,
    "build_unit": "shared_partition_folder",
    "source_paths": {
        "vcu_silver": str(VCU_SILVER),
        "cell_pack_features": str(CELL_PACK),
        "cell_module_features": str(CELL_MODULE),
    },
    "output_path": str(OUT_DIR),
    "safe_mask": {
        "PACK_GAP": PACK_GAP,
        "CELL_HARD_BREACH": CELL_HARD_BREACH,
        "combined_safe_mask": SAFE_MASK,
    },
    "current_thresholds": {
        "idle_current_a": IDLE_CURRENT_A,
        "heavy_current_a": HEAVY_CURRENT_A,
    },
    "selected_columns": {
        "vcu_cols": VCU_COLS,
        "pack_cols": PACK_COLS,
        "module_cols": MODULE_COLS,
    },
    "partitions_considered": dates,
    "results": results,
}

manifest_path = OUT_DIR / "manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=2))

print(f"\nManifest saved -> {manifest_path}")