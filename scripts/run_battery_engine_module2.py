"""
Battery Intelligence Engine — Module 2 Runner (EV01 Loader)
============================================================

Weak Cell / Weak Module / Hot Sensor Intelligence.

Loads the master dataset and gold cell_features partitions, runs four analyses,
and writes all outputs to outputs/battery_engine/.

Usage:
    python scripts/run_battery_engine_module2.py

Outputs:
    outputs/battery_engine/weak_cell_profile.csv
    outputs/battery_engine/hot_sensor_profile.csv
    outputs/battery_engine/module_chronic_risk.csv
    outputs/battery_engine/cell_delta_profile.csv
    outputs/battery_engine/battery_engine_module2_summary.json
    reports/battery_engine_module2_report.md
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import DATA_DIR, PROJECT_ROOT as CFG_ROOT
from battery_intelligence.io import (
    load_master_dataset,
    write_csv,
    write_json,
    write_markdown,
)
from battery_intelligence.cell_intelligence import (
    compute_weak_cell_profile,
    compute_hot_sensor_profile,
    compute_module_chronic_risk,
    compute_cell_delta_profile,
    generate_module2_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_DIR = DATA_DIR / "analysis" / "master"
CELL_FEATURES_DIR = DATA_DIR / "gold_cells" / "cell_features"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "battery_engine"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "battery_engine_module2_report.md"
VEHICLE_ID   = "EV01"
MACHINE_TYPE = "Loader"


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    t_start = time.perf_counter()
    run_ts = datetime.datetime.utcnow().isoformat()

    _print_section("Battery Intelligence Engine — Module 2")
    print(f"  Run time         : {run_ts}")
    print(f"  Master           : {MASTER_DIR}")
    print(f"  Cell features    : {CELL_FEATURES_DIR}")
    print(f"  Outputs          : {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Load master dataset
    # ------------------------------------------------------------------
    _print_section("Step 1/5 — Loading master dataset")
    df = load_master_dataset(MASTER_DIR)
    n_rows = len(df)
    n_partitions = df["dt"].nunique() if "dt" in df.columns else "unknown"
    print(f"  Loaded {n_rows:,} rows across {n_partitions} date partitions")

    date_start = str(df["timestamp"].min())[:10] if "timestamp" in df.columns else "?"
    date_end   = str(df["timestamp"].max())[:10] if "timestamp" in df.columns else "?"
    print(f"  Date range       : {date_start} → {date_end}")

    # ------------------------------------------------------------------
    # Step 2: Weak cell + hot sensor profiles (from master)
    # ------------------------------------------------------------------
    _print_section("Step 2/5 — Weak cell & hot sensor profiles")

    weak_cell_df = compute_weak_cell_profile(df)
    chronic_weak = int(weak_cell_df["chronic_weak"].sum()) if not weak_cell_df.empty else 0
    print(f"  Total cells seen as module-lowest : {len(weak_cell_df)}")
    print(f"  Chronically weak cells (>{15}%)   : {chronic_weak}")
    if not weak_cell_df.empty:
        top = weak_cell_df.head(5)
        print("  Top 5 weakest cells:")
        for _, r in top.iterrows():
            print(f"    {r['cell_id']:<10}  M{r['module_id']}  {r['pct_lowest_in_module']:.1f}%  [{r['weakness_status']}]")

    hot_sensor_df = compute_hot_sensor_profile(df)
    chronic_hot = int(hot_sensor_df["chronic_hot"].sum()) if not hot_sensor_df.empty else 0
    print(f"\n  Total sensors seen as module-hottest : {len(hot_sensor_df)}")
    print(f"  Chronically hot sensors (>{15}%)     : {chronic_hot}")
    if not hot_sensor_df.empty:
        top = hot_sensor_df.head(5)
        print("  Top 5 hottest sensors:")
        for _, r in top.iterrows():
            print(f"    {r['sensor_id']:<10}  M{r['module_id']}  {r['pct_hottest_in_module']:.1f}%  [{r['heat_status']}]")

    # ------------------------------------------------------------------
    # Step 3: Module chronic risk (from master)
    # ------------------------------------------------------------------
    _print_section("Step 3/5 — Module chronic risk ranking")

    module_risk_df = compute_module_chronic_risk(df)
    print(f"  Modules analysed : {len(module_risk_df)}")
    print("  Module risk ranking:")
    for _, r in module_risk_df.iterrows():
        print(
            f"    Module {int(r['module_id'])}  "
            f"score={r['module_chronic_risk_score']:>3}  "
            f"avg_v_range={r['avg_voltage_range_mv']:.1f}mV  "
            f"avg_t_range={r['avg_temp_range_c']:.1f}°C  "
            f"[{r['module_chronic_risk_status']}]"
        )

    # ------------------------------------------------------------------
    # Step 4: Cell delta profile (from gold cell_features, incremental)
    # ------------------------------------------------------------------
    _print_section("Step 4/5 — Cell delta profile (gold cell_features)")

    cell_features_partitions = sorted(CELL_FEATURES_DIR.glob(f"dt=*/vehicle_id={VEHICLE_ID}.parquet"))
    n_cell_partitions = len(cell_features_partitions)
    print(f"  Cell feature partitions found : {n_cell_partitions}")

    if n_cell_partitions == 0:
        print(f"  [WARN] No cell feature partitions found at {CELL_FEATURES_DIR}")
        print(f"         Skipping cell delta analysis.")
        cell_delta_df = __import__("pandas").DataFrame()
    else:
        print(f"  Loading incrementally (one partition at a time)...")
        cell_delta_df = compute_cell_delta_profile(CELL_FEATURES_DIR, VEHICLE_ID)
        print(f"  Cells profiled : {len(cell_delta_df)}")

        if not cell_delta_df.empty:
            status_counts = cell_delta_df["weakness_status"].value_counts().to_dict()
            print("  Weakness status distribution:")
            for status in ("Critical", "Monitor", "Watch", "Healthy"):
                cnt = status_counts.get(status, 0)
                print(f"    {status:<10} {cnt:>4} cells")

            print("  Top 5 weakest cells (most negative mean_delta_mv):")
            for _, r in cell_delta_df.head(5).iterrows():
                print(
                    f"    {r['cell_id']:<10}  M{r['module_id']}  "
                    f"mean_delta={r['mean_delta_mv']:.2f}mV  "
                    f"outlier={r['outlier_frequency_pct']:.1f}%  "
                    f"score={r['weakness_score']:.1f}  [{r['weakness_status']}]"
                )

    # ------------------------------------------------------------------
    # Step 5: Write outputs
    # ------------------------------------------------------------------
    _print_section("Step 5/5 — Writing outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "weak_cell_profile":             OUTPUT_DIR / "weak_cell_profile.csv",
        "hot_sensor_profile":            OUTPUT_DIR / "hot_sensor_profile.csv",
        "module_chronic_risk":           OUTPUT_DIR / "module_chronic_risk.csv",
        "cell_delta_profile":            OUTPUT_DIR / "cell_delta_profile.csv",
        "battery_engine_module2_summary": OUTPUT_DIR / "battery_engine_module2_summary.json",
    }

    write_csv(weak_cell_df,    paths["weak_cell_profile"])
    print(f"  ✓ {paths['weak_cell_profile']}")

    write_csv(hot_sensor_df,   paths["hot_sensor_profile"])
    print(f"  ✓ {paths['hot_sensor_profile']}")

    write_csv(module_risk_df,  paths["module_chronic_risk"])
    print(f"  ✓ {paths['module_chronic_risk']}")

    write_csv(cell_delta_df,   paths["cell_delta_profile"])
    print(f"  ✓ {paths['cell_delta_profile']}")

    elapsed = round(time.perf_counter() - t_start, 1)

    # Build summary JSON
    summary_meta: dict = {
        "run_id": f"module2__{run_ts[:10]}",
        "run_timestamp": run_ts,
        "elapsed_seconds": elapsed,
        "vehicle_id": VEHICLE_ID,
        "master_rows_processed": n_rows,
        "master_partitions": int(n_partitions) if isinstance(n_partitions, int) else str(n_partitions),
        "date_range": {"start": date_start, "end": date_end},
        "cell_feature_partitions": n_cell_partitions,
        "weak_cell_summary": {
            "total_cells_seen": len(weak_cell_df),
            "chronic_weak_count": chronic_weak,
            "top_weak_cell": str(weak_cell_df.iloc[0]["cell_id"]) if not weak_cell_df.empty else None,
            "top_weak_cell_pct": float(weak_cell_df.iloc[0]["pct_lowest_in_module"]) if not weak_cell_df.empty else None,
        },
        "hot_sensor_summary": {
            "total_sensors_seen": len(hot_sensor_df),
            "chronic_hot_count": chronic_hot,
            "top_hot_sensor": str(hot_sensor_df.iloc[0]["sensor_id"]) if not hot_sensor_df.empty else None,
            "top_hot_sensor_pct": float(hot_sensor_df.iloc[0]["pct_hottest_in_module"]) if not hot_sensor_df.empty else None,
        },
        "module_risk_summary": (
            module_risk_df[["module_id", "module_chronic_risk_score", "module_chronic_risk_status"]]
            .set_index("module_id")
            .to_dict(orient="index")
            if not module_risk_df.empty else {}
        ),
        "cell_delta_summary": (
            {
                "total_cells": len(cell_delta_df),
                "status_distribution": cell_delta_df["weakness_status"].value_counts().to_dict(),
                "weakest_cell": str(cell_delta_df.iloc[0]["cell_id"]) if not cell_delta_df.empty else None,
                "weakest_cell_mean_delta_mv": float(cell_delta_df.iloc[0]["mean_delta_mv"]) if not cell_delta_df.empty else None,
            }
            if not cell_delta_df.empty else {}
        ),
        "output_paths": {k: str(v) for k, v in paths.items()},
        "report_path": str(REPORT_PATH),
    }

    write_json(summary_meta, paths["battery_engine_module2_summary"])
    print(f"  ✓ {paths['battery_engine_module2_summary']}")

    # Generate and write report
    report_md = generate_module2_report(
        weak_cell_df=weak_cell_df,
        hot_sensor_df=hot_sensor_df,
        module_risk_df=module_risk_df,
        cell_delta_df=cell_delta_df,
        n_master_rows=n_rows,
        n_cell_feature_partitions=n_cell_partitions,
        output_dir=OUTPUT_DIR,
    )
    write_markdown(report_md, REPORT_PATH)
    print(f"  ✓ {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    _print_section("Run Complete")
    print(f"  Master rows processed      : {n_rows:,}")
    print(f"  Date range                 : {date_start} → {date_end}")
    print(f"  Cell feature partitions    : {n_cell_partitions}")
    print(f"  Chronically weak cells     : {chronic_weak}")
    print(f"  Chronically hot sensors    : {chronic_hot}")
    if not module_risk_df.empty:
        riskiest = module_risk_df.iloc[0]
        print(f"  Riskiest module            : Module {int(riskiest['module_id'])}  (score={riskiest['module_chronic_risk_score']}  [{riskiest['module_chronic_risk_status']}])")
    if not cell_delta_df.empty:
        weakest = cell_delta_df.iloc[0]
        print(f"  Weakest cell by delta      : {weakest['cell_id']}  (mean_delta={weakest['mean_delta_mv']:.2f}mV  [{weakest['weakness_status']}])")
    print(f"\n  Output files:")
    for _, path in paths.items():
        print(f"    {path}")
    print(f"    {REPORT_PATH}")
    print(f"\n  Completed in {elapsed}s")


if __name__ == "__main__":
    main()
