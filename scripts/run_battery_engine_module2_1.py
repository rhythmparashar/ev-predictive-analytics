"""
Battery Intelligence Engine — Module 2.1 Runner
================================================

Cell/Sensor Completeness Investigation.

Checks every expected cell (180) and temperature sensor (90) against all pipeline
layers: schema → silver_cells → gold cell_features → Module 2 outputs.

Usage:
    python scripts/run_battery_engine_module2_1.py

Outputs:
    outputs/battery_engine/cell_sensor_completeness_report.csv
    outputs/battery_engine/missing_cells.csv
    outputs/battery_engine/missing_temp_sensors.csv
    outputs/battery_engine/module2_1_summary.json
    reports/battery_engine_module2_1_completeness_report.md
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from configs.settings import DATA_DIR, PROJECT_ROOT as CFG_ROOT
from battery_intelligence.io import write_csv, write_json, write_markdown
from battery_intelligence.completeness_check import (
    run_cell_completeness,
    run_sensor_completeness,
    build_completeness_summary,
    generate_completeness_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SILVER_DIR         = DATA_DIR / "silver_cells"
CELL_FEATURES_DIR  = DATA_DIR / "gold_cells" / "cell_features"
OUTPUT_DIR         = PROJECT_ROOT / "outputs" / "battery_engine"
REPORT_DIR         = PROJECT_ROOT / "reports"
REPORT_PATH        = REPORT_DIR / "battery_engine_module2_1_completeness_report.md"

WEAK_CELL_CSV      = OUTPUT_DIR / "weak_cell_profile.csv"
HOT_SENSOR_CSV     = OUTPUT_DIR / "hot_sensor_profile.csv"
CELL_DELTA_CSV     = OUTPUT_DIR / "cell_delta_profile.csv"

SCHEMA_VOLTAGE     = PROJECT_ROOT / "schema" / "cell_voltage_schema.yaml"
SCHEMA_TEMP        = PROJECT_ROOT / "schema" / "cell_temp_schema.yaml"


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    t_start = time.perf_counter()
    run_ts = datetime.datetime.utcnow().isoformat()

    _print_section("Battery Intelligence Engine — Module 2.1")
    print(f"  Run time  : {run_ts}")
    print(f"  Silver    : {SILVER_DIR}")
    print(f"  Gold      : {CELL_FEATURES_DIR}")
    print(f"  M2 outputs: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Load schemas
    # ------------------------------------------------------------------
    _print_section("Step 1/5 — Loading schemas")

    with open(SCHEMA_VOLTAGE) as f:
        cell_voltage_schema = yaml.safe_load(f)
    with open(SCHEMA_TEMP) as f:
        cell_temp_schema = yaml.safe_load(f)

    print(f"  Voltage schema: {cell_voltage_schema['modules']} modules × {cell_voltage_schema['cells_per_module']} cells")
    print(f"  Temp schema   : {cell_temp_schema['modules']} modules × {cell_temp_schema['sensors_per_module']} sensors")

    # ------------------------------------------------------------------
    # Step 2: Cell completeness
    # ------------------------------------------------------------------
    _print_section("Step 2/5 — Cell completeness (180 cells)")

    cell_df = run_cell_completeness(
        silver_dir=SILVER_DIR,
        cell_features_dir=CELL_FEATURES_DIR,
        weak_cell_csv=WEAK_CELL_CSV,
        cell_delta_csv=CELL_DELTA_CSV,
        cell_voltage_schema=cell_voltage_schema,
    )

    n_in_weak    = int(cell_df["present_in_weak_cell_profile"].sum())
    n_in_delta   = int(cell_df["present_in_cell_delta_profile"].sum())
    n_in_schema  = int(cell_df["present_in_schema"].sum())
    n_in_silver  = int(cell_df["present_in_silver_cells"].sum())
    n_in_gold    = int(cell_df["present_in_gold_cell_features"].sum())

    print(f"  Total expected        : 180")
    print(f"  Present in schema     : {n_in_schema}/180")
    print(f"  Present in silver     : {n_in_silver}/180")
    print(f"  Present in gold       : {n_in_gold}/180")
    print(f"  Present in weak_cell  : {n_in_weak}/180")
    print(f"  Present in cell_delta : {n_in_delta}/180")

    missing_weak = cell_df[~cell_df["present_in_weak_cell_profile"]]
    if not missing_weak.empty:
        print(f"\n  Missing from weak_cell_profile ({len(missing_weak)}):")
        for _, r in missing_weak.iterrows():
            print(f"    {r['cell_id']:<10}  M{r['module_id']}  null_pct={r['silver_null_pct']:.3f}%")

    # ------------------------------------------------------------------
    # Step 3: Sensor completeness
    # ------------------------------------------------------------------
    _print_section("Step 3/5 — Sensor completeness (90 sensors)")

    sensor_df = run_sensor_completeness(
        silver_dir=SILVER_DIR,
        hot_sensor_csv=HOT_SENSOR_CSV,
        cell_temp_schema=cell_temp_schema,
    )

    n_s_schema = int(sensor_df["present_in_schema"].sum())
    n_s_silver = int(sensor_df["present_in_silver_cells"].sum())
    n_s_hot    = int(sensor_df["present_in_hot_sensor_profile"].sum())

    print(f"  Total expected        : 90")
    print(f"  Present in schema     : {n_s_schema}/90")
    print(f"  Present in silver     : {n_s_silver}/90")
    print(f"  Present in hot_sensor : {n_s_hot}/90")

    missing_hot = sensor_df[~sensor_df["present_in_hot_sensor_profile"]]
    if not missing_hot.empty:
        print(f"\n  Missing from hot_sensor_profile ({len(missing_hot)}):")
        for _, r in missing_hot.iterrows():
            print(f"    {r['sensor_id']:<10}  M{r['module_id']}  null_pct={r['silver_null_pct']:.3f}%")

    # ------------------------------------------------------------------
    # Step 4: Build summary and root cause
    # ------------------------------------------------------------------
    _print_section("Step 4/5 — Root cause analysis")

    summary = build_completeness_summary(cell_df, sensor_df)

    print(f"\n  Root cause:")
    print(f"    {summary['root_cause']}")
    print(f"\n  Module 2 rerun needed : {'No' if not summary['module2_rerun_needed'] else 'YES'}")
    print(f"  Recommended fix       : {summary['recommended_fix']}")

    # ------------------------------------------------------------------
    # Step 5: Write outputs
    # ------------------------------------------------------------------
    _print_section("Step 5/5 — Writing outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Combined completeness report CSV
    import pandas as pd
    cell_out   = cell_df.assign(layer="cell")
    sensor_out = sensor_df.assign(layer="sensor")
    combined = pd.concat(
        [
            cell_out.rename(columns={"cell_id": "id", "present_in_weak_cell_profile": "present_in_m2_profile"}),
            sensor_out.rename(columns={"sensor_id": "id", "present_in_hot_sensor_profile": "present_in_m2_profile"}),
        ],
        ignore_index=True,
    )

    paths = {
        "cell_sensor_completeness_report": OUTPUT_DIR / "cell_sensor_completeness_report.csv",
        "missing_cells":                   OUTPUT_DIR / "missing_cells.csv",
        "missing_temp_sensors":            OUTPUT_DIR / "missing_temp_sensors.csv",
        "module2_1_summary":               OUTPUT_DIR / "module2_1_summary.json",
    }

    write_csv(combined, paths["cell_sensor_completeness_report"])
    print(f"  ✓ {paths['cell_sensor_completeness_report']}")

    write_csv(missing_weak.reset_index(drop=True), paths["missing_cells"])
    print(f"  ✓ {paths['missing_cells']}")

    write_csv(missing_hot.reset_index(drop=True), paths["missing_temp_sensors"])
    print(f"  ✓ {paths['missing_temp_sensors']}")

    elapsed = round(time.perf_counter() - t_start, 1)
    summary["run_id"]          = f"module2_1__{run_ts[:10]}"
    summary["run_timestamp"]   = run_ts
    summary["elapsed_seconds"] = elapsed
    summary["output_paths"]    = {k: str(v) for k, v in paths.items()}
    summary["report_path"]     = str(REPORT_PATH)

    write_json(summary, paths["module2_1_summary"])
    print(f"  ✓ {paths['module2_1_summary']}")

    report_md = generate_completeness_report(cell_df, sensor_df, summary)
    write_markdown(report_md, REPORT_PATH)
    print(f"  ✓ {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    _print_section("Run Complete — Final Verdict")

    print(f"  Expected cells   : 180")
    print(f"  Found in schema  : {n_in_schema}/180  silver={n_in_silver}/180  gold={n_in_gold}/180  delta={n_in_delta}/180")
    print(f"  In weak_profile  : {n_in_weak}/180  (missing: {sorted(missing_weak['cell_id'].tolist())})")

    print(f"\n  Expected sensors : 90")
    print(f"  Found in schema  : {n_s_schema}/90  silver={n_s_silver}/90")
    print(f"  In hot_profile   : {n_s_hot}/90  (missing: {sorted(missing_hot['sensor_id'].tolist())})")

    print(f"\n  Root cause       : {summary['root_cause'][:120]}...")
    print(f"  Module 2 rerun   : {'Not needed' if not summary['module2_rerun_needed'] else 'NEEDED'}")
    print(f"\n  Completed in {elapsed}s")


if __name__ == "__main__":
    main()
