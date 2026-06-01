"""
Daily Automated Pipeline
========================
Processes new dates through the full chain:
  raw_parquet → silver → cells → gold → master → SOC score → battery health → report

The pipeline tracks state in data/state/pipeline_state.json.
On first run, use --mark-existing to stamp all already-processed historical dates
as done so the pipeline only acts on genuinely new dates going forward.

Usage:
    # First-time setup — stamp all existing silver dates as done:
    python scripts/run_daily_pipeline.py --mark-existing

    # Normal daily run — process only new dates:
    python scripts/run_daily_pipeline.py

    # Process one specific date:
    python scripts/run_daily_pipeline.py --date 2026-05-15

    # Reprocess a date even if already done:
    python scripts/run_daily_pipeline.py --date 2026-05-15 --force

    # See what would run without doing it:
    python scripts/run_daily_pipeline.py --dry-run

Outputs:
    reports/daily/YYYY-MM-DD_pipeline_report.md
    data/state/pipeline_state.json
    outputs/soc_scores/dt=*/vehicle_id=*.parquet
    outputs/battery_health/dt=*/scores_*.parquet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    DATA_DIR, SILVER_DIR, GOLD_DIR, RAW_DIR, RAW_PARQUET_DIR,
    RAW_CELL_VOLTAGE_DIR, STATE_DIR, MACHINE_TYPES,
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "daily"
STATE_FILE  = STATE_DIR / "pipeline_state.json"
MODEL_DIR   = PROJECT_ROOT / "models" / "soc_forecast"
PYTHON      = sys.executable

SOC_MODELS = {
    "Loader":    "v3__2026-05-20__7b14ac62",
    "Excavator": "v3__2026-05-20__7b14ac62",   # replace when EV02 model trained
}
DROP_QUALITY_MASK = 52
SEP = "─" * 60


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_dates": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def mark_processed(state: dict, dt: str, vehicle_id: str, steps: dict) -> None:
    state["processed_dates"].setdefault(dt, {})[vehicle_id] = {
        "ts": datetime.utcnow().isoformat(),
        "steps": steps,
    }
    save_state(state)


# ─────────────────────────────────────────────────────────────────────────────
# Date / vehicle discovery
# ─────────────────────────────────────────────────────────────────────────────

def silver_dates() -> set[str]:
    """Dates that already have silver output."""
    if not SILVER_DIR.exists():
        return set()
    return {
        d.name.replace("dt=", "")
        for d in SILVER_DIR.iterdir()
        if d.is_dir() and d.name.startswith("dt=")
    }


def raw_dates() -> list[str]:
    """All dates present in raw CSV or raw_parquet."""
    dates: set[str] = set()
    for base_dir in (RAW_DIR, RAW_PARQUET_DIR):
        if base_dir.exists():
            for d in base_dir.iterdir():
                if d.is_dir() and d.name.startswith("dt="):
                    dates.add(d.name.replace("dt=", ""))
    return sorted(dates)


def vehicles_for_date(dt: str) -> list[str]:
    """Vehicle IDs available for this date — checks raw CSV, raw_parquet, then silver."""
    ids: set[str] = set()

    # raw CSV (written by ingest_incoming.py)
    raw_csv_dir = RAW_DIR / f"dt={dt}"
    if raw_csv_dir.exists():
        for p in raw_csv_dir.glob("vehicle_id=*.csv"):
            ids.add(p.stem.replace("vehicle_id=", ""))

    # raw_parquet (written by run_day.py after silver step)
    raw_pq_dir = RAW_PARQUET_DIR / f"dt={dt}"
    if raw_pq_dir.exists():
        for p in raw_pq_dir.glob("vehicle_id=*.parquet"):
            ids.add(p.stem.replace("vehicle_id=", ""))

    # silver fallback
    if not ids:
        silver_dir = SILVER_DIR / f"dt={dt}"
        if silver_dir.exists():
            for p in silver_dir.glob("vehicle_id=*.parquet"):
                ids.add(p.stem.replace("vehicle_id=", ""))

    return sorted(ids)


def new_dates(state: dict) -> list[str]:
    """Dates in raw_parquet that are NOT yet in state AND not already in silver."""
    done  = set(state.get("processed_dates", {}).keys())
    exist = silver_dates()
    return [d for d in raw_dates() if d not in done and d not in exist]


# ─────────────────────────────────────────────────────────────────────────────
# Steps — each returns (ok, message)
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> tuple[bool, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if r.returncode == 0:
        return True, "ok"
    err = (r.stderr or r.stdout or "").strip().splitlines()
    return False, err[-1] if err else f"exit {r.returncode}"


def step_silver(dt: str, vehicle_id: str) -> tuple[bool, str]:
    ok, msg = _run([PYTHON, "scripts/run_day.py", "--dt", dt, "--vehicle-id", vehicle_id])
    if not ok and "already exists" in msg.lower():
        return True, "skipped (already done)"
    return ok, msg


def step_cells(dt: str, vehicle_id: str) -> tuple[bool, str]:
    if not (RAW_CELL_VOLTAGE_DIR / f"dt={dt}").exists():
        return True, "skipped (no cell data)"
    return _run([PYTHON, "run_cells.py", "cell_ingest", "--backfill",
                 "--start", dt, "--end", dt, "--vehicle-id", vehicle_id])


def step_gold(dt: str, vehicle_id: str) -> tuple[bool, str]:
    try:
        from features.pipeline import build_gold_for_vehicle_day
        m = build_gold_for_vehicle_day(dt=dt, vehicle_id=vehicle_id)
        rows = m.get("counts", {}).get("window_rows", 0)
        return True, f"{rows:,} rows"
    except Exception as e:
        return False, str(e)


def step_master(_dt: str, vehicle_id: str) -> tuple[bool, str]:
    return _run([PYTHON, "build_master_dataset.py", "--vehicle-id", vehicle_id])


def step_soc(dt: str, vehicle_id: str) -> tuple[bool, str]:
    try:
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        machine  = MACHINE_TYPES.get(vehicle_id, "Unknown")
        run_id   = SOC_MODELS.get(machine)
        if not run_id:
            return False, f"no model for {machine}"

        model_path = MODEL_DIR / run_id / "model.json"
        feat_path  = MODEL_DIR / run_id / "feature_set.json"
        if not model_path.exists():
            return False, "model file not found"

        gold_path = GOLD_DIR / "window_features" / f"dt={dt}" / f"vehicle_id={vehicle_id}"
        parts = list(gold_path.glob("*.parquet")) if gold_path.exists() else []
        if not parts:
            return False, "no gold features"

        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        features = json.loads(feat_path.read_text())["features"]
        target   = "y_soc_t_plus_300s"

        chunks = []
        for f in parts:
            df = pd.read_parquet(f)
            if "label_available" in df.columns:
                df = df[df["label_available"] == 1]
            if "quality_flag" in df.columns:
                df = df[(df["quality_flag"].astype(int) & DROP_QUALITY_MASK) == 0]
            if "soc_pct" in df.columns:
                df = df[df["soc_pct"].notna()]
            if not df.empty and target in df.columns:
                chunks.append(df)

        if not chunks:
            return False, "no clean rows after quality filter"

        df = pd.concat(chunks, ignore_index=True)
        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        yhat = model.predict(df[features].fillna(0).astype("float32"), validate_features=False)
        mae  = float(np.mean(np.abs(df[target].astype(float).values - yhat)))

        out = PROJECT_ROOT / "outputs" / "soc_scores" / f"dt={dt}"
        out.mkdir(parents=True, exist_ok=True)
        pred_df = df[["soc_pct", target]].copy() if "soc_pct" in df.columns else df[[target]].copy()
        pred_df["soc_pred_t300s"] = yhat
        pred_df["vehicle_id"]     = vehicle_id
        pred_df["machine_type"]   = machine
        pred_df["model_run"]      = run_id
        pred_df.to_parquet(out / f"vehicle_id={vehicle_id}.parquet", index=False)

        return True, f"MAE={mae:.3f}%  n={len(df):,}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def step_battery(dt: str, vehicle_id: str) -> tuple[bool, str]:
    try:
        import pandas as pd
        from battery_intelligence.loader import load_master
        from battery_intelligence.health_score import compute_health_scores
        from battery_intelligence.rule_engine import evaluate_rules

        from datetime import date, timedelta
        next_day = (date.fromisoformat(dt) + timedelta(days=1)).isoformat()
        master = load_master(DATA_DIR, start_date=dt, end_date=next_day)
        if master is not None and not master.empty and "vehicle_id" in master.columns:
            master = master[master["vehicle_id"] == vehicle_id]
        if master is None or master.empty:
            return False, "no master data"

        scored = compute_health_scores(master)
        alerts = evaluate_rules(scored)

        out = PROJECT_ROOT / "outputs" / "battery_health" / f"dt={dt}"
        out.mkdir(parents=True, exist_ok=True)
        scored.to_parquet(out / f"scores_{vehicle_id}.parquet", index=False)
        if not alerts.empty:
            alerts.to_parquet(out / f"alerts_{vehicle_id}.parquet", index=False)

        mean_score = scored["health_score"].mean() if "health_score" in scored.columns else None
        score_str  = f"health={mean_score:.1f}/100  " if mean_score else ""
        return True, f"{score_str}alerts={len(alerts)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


import json  # noqa: E402 (needed inside step_soc)

STEPS = [
    ("silver",  step_silver),
    ("cells",   step_cells),
    ("gold",    step_gold),
    ("master",  step_master),
    ("soc",     step_soc),
    ("battery", step_battery),
]

CRITICAL_STEPS = {"silver", "gold"}


# ─────────────────────────────────────────────────────────────────────────────
# Per-date run
# ─────────────────────────────────────────────────────────────────────────────

def process_date(dt: str, state: dict, dry_run: bool = False) -> dict:
    vehicles = vehicles_for_date(dt)
    if not vehicles:
        print(f"  [{dt}] no vehicles — skip")
        return {}

    results: dict[str, dict] = {}
    for vid in vehicles:
        machine = MACHINE_TYPES.get(vid, "Unknown")
        print(f"\n  {SEP}")
        print(f"  {dt}  |  {vid} ({machine})")
        print(f"  {SEP}")

        step_results: dict = {}
        for name, fn in STEPS:
            if dry_run:
                print(f"    [{name:<8}]  DRY RUN")
                step_results[name] = "dry_run"
                continue

            t0 = time.time()
            try:
                ok, msg = fn(dt, vid)
            except Exception:
                ok, msg = False, traceback.format_exc().splitlines()[-1]

            elapsed = time.time() - t0
            icon    = "✓" if ok else "✗"
            print(f"    [{name:<8}] {icon}  {msg}  ({elapsed:.1f}s)")
            step_results[name] = {"ok": ok, "msg": msg, "s": round(elapsed, 1)}

            if not ok and name in CRITICAL_STEPS:
                print(f"    ↳ critical step failed — skipping remaining steps for {vid}")
                break

        results[vid] = step_results
        if not dry_run:
            mark_processed(state, dt, vid, step_results)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(run_date: str, all_results: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Daily Pipeline Report — {run_date}",
        f"",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"",
        "---",
        "",
    ]
    for dt in sorted(all_results):
        lines.append(f"## {dt}")
        lines.append("")
        for vid, steps in all_results[dt].items():
            machine = MACHINE_TYPES.get(vid, "Unknown")
            lines += [f"### {vid} — {machine}", "", "| Step | Status | Detail |", "|---|---|---|"]
            for step, r in steps.items():
                if isinstance(r, dict):
                    icon = "✓" if r["ok"] else "✗"
                    lines.append(f"| {step} | {icon} | {r['msg']} ({r['s']}s) |")
                else:
                    lines.append(f"| {step} | — | {r} |")
            lines.append("")

    path = REPORTS_DIR / f"{run_date}_pipeline_report.md"
    path.write_text("\n".join(lines))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Mark-existing helper
# ─────────────────────────────────────────────────────────────────────────────

def mark_existing(state: dict) -> None:
    """Stamp all dates that already have silver output as processed so the
    pipeline doesn't re-run the entire history on first use."""
    existing = silver_dates()
    if not existing:
        print("No existing silver dates found.")
        return

    count = 0
    ts = datetime.utcnow().isoformat()
    for dt in sorted(existing):
        vehicles = vehicles_for_date(dt)
        if not vehicles:
            # Fall back to detecting from silver
            silver_dir = SILVER_DIR / f"dt={dt}"
            vehicles = [p.stem.replace("vehicle_id=", "") for p in silver_dir.glob("vehicle_id=*.parquet")]
        for vid in vehicles:
            state["processed_dates"].setdefault(dt, {})[vid] = {
                "ts": ts,
                "steps": {"note": "pre-existing — marked by --mark-existing"},
            }
            count += 1

    save_state(state)
    print(f"Marked {len(existing)} dates × vehicles ({count} entries) as already processed.")
    print(f"State saved to {STATE_FILE}")
    print("\nFrom now on, `python scripts/run_daily_pipeline.py` will only process NEW dates.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Daily automated pipeline")
    ap.add_argument("--date",          default=None,       help="Process specific date YYYY-MM-DD")
    ap.add_argument("--force",         action="store_true", help="Reprocess even if already done")
    ap.add_argument("--dry-run",       action="store_true", dest="dry_run")
    ap.add_argument("--mark-existing", action="store_true", dest="mark_existing",
                    help="Stamp all existing silver dates as done (run once on first setup)")
    args = ap.parse_args()

    state = load_state()

    # ── First-time setup mode ────────────────────────────────────────────────
    if args.mark_existing:
        mark_existing(state)
        return

    # ── Decide which dates to process ───────────────────────────────────────
    if args.date:
        dates = [args.date]
    elif args.force:
        dates = raw_dates()
    else:
        dates = new_dates(state)

    if not dates:
        print("Nothing to process — no new dates found.")
        print("Use --date YYYY-MM-DD to process a specific date, or --force to reprocess all.")
        return

    run_label = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"  EV PREDICTIVE ANALYTICS — DAILY PIPELINE")
    print(f"  Dates to process: {len(dates)}")
    if args.dry_run:
        print(f"  MODE: DRY RUN")
    print(f"{'='*60}")

    all_results: dict = {}
    t0 = time.time()

    for dt in dates:
        r = process_date(dt, state, dry_run=args.dry_run)
        if r:
            all_results[dt] = r

    if all_results and not args.dry_run:
        path = write_report(run_label, all_results)
        print(f"\n  Report → {path}")

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
