"""
Battery Intelligence Engine — Module 1 Runner
=============================================

Loads all master dataset partitions, computes battery health scores,
evaluates rule-based alerts, generates aggregated summaries, and writes
all outputs to outputs/battery_engine/.

Usage:
    python scripts/run_battery_engine_module1.py

Outputs:
    outputs/battery_engine/battery_health_scores.parquet
    outputs/battery_engine/battery_alerts.parquet
    outputs/battery_engine/battery_daily_summary.csv
    outputs/battery_engine/battery_module_summary.csv
    outputs/battery_engine/battery_engine_module1_summary.json
    outputs/battery_engine/battery_twin_state_sample.json
    reports/battery_engine_module1_report.md
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import DATA_DIR, PROJECT_ROOT as CFG_ROOT
from battery_intelligence.io import (
    load_master_dataset,
    write_parquet_atomic,
    write_csv,
    write_json,
    write_markdown,
)
from battery_intelligence.health_score import compute_health_scores
from battery_intelligence.rule_engine import evaluate_rules
from battery_intelligence.summary import (
    compute_daily_summary,
    compute_module_summary,
    build_battery_twin_state,
    generate_report,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MASTER_DIR = DATA_DIR / "analysis" / "master"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "battery_engine"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "battery_engine_module1_report.md"

# Columns we expect to use — used to build availability/missing dictionaries
EXPECTED_COLUMNS = [
    "timestamp", "vehicle_id", "dt", "trip_id",
    "soc_pct", "battery_current_a", "abs_current_a",
    "stack_voltage_v", "output_power_kw",
    "max_cell_voltage_v", "min_cell_voltage_v", "avg_cell_voltage_v",
    "cell_spread_v",
    "max_battery_temp_c", "min_battery_temp_c", "avg_battery_temp_c",
    "pack_voltage_range", "pack_voltage_std",
    "pack_temp_range",
    "worst_cell_id", "hottest_module_id", "imbalance_flag",
    "module_1_voltage_range", "module_2_voltage_range",
    "module_3_voltage_range", "module_4_voltage_range", "module_5_voltage_range",
    "module_1_temp_range", "module_2_temp_range",
    "module_3_temp_range", "module_4_temp_range", "module_5_temp_range",
    "module_1_lowest_cell", "module_2_lowest_cell",
    "module_3_lowest_cell", "module_4_lowest_cell", "module_5_lowest_cell",
    "module_1_hottest_sensor", "module_2_hottest_sensor",
    "module_3_hottest_sensor", "module_4_hottest_sensor", "module_5_hottest_sensor",
    "fault_any", "vcu_quality_flag", "cell_quality_flag",
    "current_direction", "operating_regime", "is_clean",
]


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    t_start = time.perf_counter()
    run_ts = datetime.datetime.utcnow().isoformat()

    _print_section("Battery Intelligence Engine — Module 1")
    print(f"  Run time : {run_ts}")
    print(f"  Master   : {MASTER_DIR}")
    print(f"  Outputs  : {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Load master dataset
    # ------------------------------------------------------------------
    _print_section("Step 1/6 — Loading master dataset")
    df = load_master_dataset(MASTER_DIR)
    n_rows = len(df)
    n_partitions = df["dt"].nunique() if "dt" in df.columns else "unknown"
    print(f"  Loaded {n_rows:,} rows across {n_partitions} date partitions")

    # Build column availability
    avail_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    print(f"  Columns used: {len(avail_cols)}/{len(EXPECTED_COLUMNS)}")
    if missing_cols:
        print(f"  Missing optional columns: {missing_cols}")

    # ------------------------------------------------------------------
    # Step 2: Compute health scores
    # ------------------------------------------------------------------
    _print_section("Step 2/6 — Computing battery health scores")
    scored_df = compute_health_scores(df)
    score_col = scored_df["battery_health_score"]
    print(f"  Mean health score : {score_col.mean():.2f}")
    print(f"  Min health score  : {score_col.min():.2f}")
    print(f"  Max health score  : {score_col.max():.2f}")

    status_counts = scored_df["battery_health_status"].value_counts().to_dict()
    print("  Status distribution:")
    for status, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / n_rows
        print(f"    {status:<12} {cnt:>8,}  ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Step 3: Evaluate rules
    # ------------------------------------------------------------------
    _print_section("Step 3/6 — Evaluating rule-based alerts")

    # Add dt to alerts by merging from scored_df
    alerts_df = evaluate_rules(scored_df)
    if not alerts_df.empty and "dt" not in alerts_df.columns and "dt" in scored_df.columns:
        ts_to_dt = scored_df.set_index("timestamp")["dt"].to_dict()
        if "timestamp" in alerts_df.columns:
            alerts_df["dt"] = alerts_df["timestamp"].map(ts_to_dt)

    print(f"  Total alert rows  : {len(alerts_df):,}")
    if not alerts_df.empty and "rule_id" in alerts_df.columns:
        top_alerts = alerts_df["rule_id"].value_counts().head(5)
        print("  Top alert types:")
        for rule_id, cnt in top_alerts.items():
            print(f"    {rule_id:<40} {cnt:>8,}")

    # ------------------------------------------------------------------
    # Step 4: Aggregate summaries
    # ------------------------------------------------------------------
    _print_section("Step 4/6 — Building daily and module summaries")
    daily_df = compute_daily_summary(scored_df, alerts_df)
    module_df = compute_module_summary(scored_df)
    print(f"  Daily summary     : {len(daily_df):,} rows")
    print(f"  Module summary    : {len(module_df):,} rows")

    # ------------------------------------------------------------------
    # Step 5: Build twin state and write outputs
    # ------------------------------------------------------------------
    _print_section("Step 5/6 — Writing outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    twin_state = build_battery_twin_state(scored_df)

    # Select output columns for health scores (drop heavy raw columns)
    score_output_cols = [c for c in (
        "timestamp", "vehicle_id", "dt", "trip_id",
        "soc_pct", "battery_current_a", "abs_current_a",
        "max_battery_temp_c", "pack_temp_range", "pack_voltage_range",
        "output_power_kw", "fault_any",
        "battery_health_score", "battery_health_status",
        "thermal_penalty", "voltage_imbalance_penalty",
        "temperature_imbalance_penalty", "current_stress_penalty",
        "weak_module_penalty", "data_quality_penalty", "fault_penalty",
        "voltage_imbalance_mv", "temp_imbalance_c",
        "weakest_module_by_voltage", "weakest_module_by_temp",
        "top_issue", "recommended_action", "business_impact",
        "imbalance_flag", "worst_cell_id", "hottest_module_id",
        "operating_regime", "current_direction",
    ) if c in scored_df.columns]

    paths = {
        "battery_health_scores": OUTPUT_DIR / "battery_health_scores.parquet",
        "battery_alerts": OUTPUT_DIR / "battery_alerts.parquet",
        "battery_daily_summary": OUTPUT_DIR / "battery_daily_summary.csv",
        "battery_module_summary": OUTPUT_DIR / "battery_module_summary.csv",
        "battery_engine_module1_summary": OUTPUT_DIR / "battery_engine_module1_summary.json",
        "battery_twin_state_sample": OUTPUT_DIR / "battery_twin_state_sample.json",
    }

    write_parquet_atomic(scored_df[score_output_cols], paths["battery_health_scores"])
    print(f"  ✓ {paths['battery_health_scores']}")

    write_parquet_atomic(alerts_df, paths["battery_alerts"])
    print(f"  ✓ {paths['battery_alerts']}")

    write_csv(daily_df, paths["battery_daily_summary"])
    print(f"  ✓ {paths['battery_daily_summary']}")

    write_csv(module_df, paths["battery_module_summary"])
    print(f"  ✓ {paths['battery_module_summary']}")

    write_json(twin_state, paths["battery_twin_state_sample"])
    print(f"  ✓ {paths['battery_twin_state_sample']}")

    elapsed = round(time.perf_counter() - t_start, 1)

    summary_meta = {
        "run_id": f"module1__{run_ts[:10]}",
        "run_timestamp": run_ts,
        "elapsed_seconds": elapsed,
        "rows_processed": n_rows,
        "date_range": {
            "start": str(scored_df["timestamp"].min()) if "timestamp" in scored_df.columns else None,
            "end": str(scored_df["timestamp"].max()) if "timestamp" in scored_df.columns else None,
        },
        "partitions": int(n_partitions) if isinstance(n_partitions, int) else str(n_partitions),
        "columns_used": avail_cols,
        "missing_columns": missing_cols,
        "health_score": {
            "mean": round(float(score_col.mean()), 3),
            "min": round(float(score_col.min()), 3),
            "max": round(float(score_col.max()), 3),
            "std": round(float(score_col.std()), 3),
        },
        "status_distribution": {k: int(v) for k, v in status_counts.items()},
        "alert_count": len(alerts_df),
        "top_alert_types": (
            alerts_df["rule_id"].value_counts().head(12).to_dict()
            if not alerts_df.empty and "rule_id" in alerts_df.columns
            else {}
        ),
        "output_paths": {k: str(v) for k, v in paths.items()},
        "report_path": str(REPORT_PATH),
    }

    write_json(summary_meta, paths["battery_engine_module1_summary"])
    print(f"  ✓ {paths['battery_engine_module1_summary']}")

    # ------------------------------------------------------------------
    # Step 6: Generate markdown report
    # ------------------------------------------------------------------
    _print_section("Step 6/6 — Generating report")
    report_md = generate_report(
        score_df=scored_df,
        alerts_df=alerts_df,
        daily_df=daily_df,
        module_df=module_df,
        twin_state=twin_state,
        avail_cols=avail_cols,
        missing_cols=missing_cols,
        output_dir=OUTPUT_DIR,
    )
    write_markdown(report_md, REPORT_PATH)
    print(f"  ✓ {REPORT_PATH}")

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    _print_section("Run Complete")
    print(f"  Rows processed        : {n_rows:,}")
    print(f"  Date range            : {summary_meta['date_range']['start'][:10]} → {summary_meta['date_range']['end'][:10]}")  # type: ignore[index]
    print(f"  Avg health score      : {summary_meta['health_score']['mean']:.2f} / 100")
    print(f"  Worst health score    : {summary_meta['health_score']['min']:.2f} / 100")
    print(f"  Status distribution   :")
    for status, cnt in sorted(status_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / n_rows
        print(f"    {status:<12} {cnt:>8,}  ({pct:.1f}%)")
    print(f"  Total alerts          : {len(alerts_df):,}")
    print(f"\n  Top 5 alert types:")
    if not alerts_df.empty:
        for rule_id, cnt in alerts_df["rule_id"].value_counts().head(5).items():
            print(f"    {rule_id:<40} {cnt:>7,}")
    print(f"\n  Output files:")
    for name, path in paths.items():
        print(f"    {path}")
    print(f"    {REPORT_PATH}")
    print(f"\n  Completed in {elapsed}s")


if __name__ == "__main__":
    main()
