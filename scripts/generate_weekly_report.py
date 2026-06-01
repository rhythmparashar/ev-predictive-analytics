"""
Weekly Battery Intelligence Report
===================================
Aggregates the last 7 days of pipeline outputs into a single weekly report.
Runs automatically every Sunday via cron.

Usage:
    python scripts/generate_weekly_report.py               # last 7 days
    python scripts/generate_weekly_report.py --week 2026-06-01  # week ending this date

Output:
    reports/weekly/week_ending_YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import DATA_DIR, MACHINE_TYPES

REPORTS_DIR = PROJECT_ROOT / "reports" / "weekly"
SOC_DIR     = PROJECT_ROOT / "outputs" / "soc_scores"
BAT_DIR     = PROJECT_ROOT / "outputs" / "battery_health"
MASTER_DIR  = DATA_DIR / "analysis" / "master"


def week_dates(end: date) -> list[str]:
    return [(end - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]


def load_master_week(dates: list[str], vehicle_id: str) -> pd.DataFrame:
    chunks = []
    for dt in dates:
        p = MASTER_DIR / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"
        if p.exists():
            chunks.append(pd.read_parquet(p))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_soc_week(dates: list[str], vehicle_id: str) -> pd.DataFrame:
    chunks = []
    for dt in dates:
        p = SOC_DIR / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"
        if p.exists():
            chunks.append(pd.read_parquet(p))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_alerts_week(dates: list[str], vehicle_id: str) -> pd.DataFrame:
    chunks = []
    for dt in dates:
        p = BAT_DIR / f"dt={dt}" / f"alerts_{vehicle_id}.parquet"
        if p.exists():
            chunks.append(pd.read_parquet(p))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def load_health_week(dates: list[str], vehicle_id: str) -> pd.DataFrame:
    chunks = []
    for dt in dates:
        p = BAT_DIR / f"dt={dt}" / f"scores_{vehicle_id}.parquet"
        if p.exists():
            chunks.append(pd.read_parquet(p))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def vehicle_section(vehicle_id: str, dates: list[str]) -> list[str]:
    machine  = MACHINE_TYPES.get(vehicle_id, "Unknown")
    master   = load_master_week(dates, vehicle_id)
    soc_df   = load_soc_week(dates, vehicle_id)
    alerts   = load_alerts_week(dates, vehicle_id)
    health   = load_health_week(dates, vehicle_id)

    days_available = sum(
        1 for dt in dates
        if (MASTER_DIR / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet").exists()
    )

    lines = [
        f"## {vehicle_id} — {machine}",
        "",
        f"Data available: **{days_available} / 7 days**",
        "",
    ]

    if master.empty:
        lines += ["_No data for this vehicle this week._", ""]
        return lines

    # ── Operating summary ───────────────────────────────────────────────────
    lines += ["### Operating Summary", "", "| Metric | Value |", "|---|---|"]

    if "soc_pct" in master.columns:
        soc = master["soc_pct"].dropna()
        lines.append(f"| SOC range | {soc.min():.1f}% – {soc.max():.1f}%  (avg {soc.mean():.1f}%) |")

    if "output_power_kw" in master.columns:
        pwr = master["output_power_kw"].dropna()
        lines.append(f"| Peak power | {pwr.max():.1f} kW |")
        lines.append(f"| Avg power  | {pwr.mean():.1f} kW |")

    if "avg_battery_temp_c" in master.columns:
        tmp = master["avg_battery_temp_c"].dropna()
        lines.append(f"| Battery temp | {tmp.min():.1f}°C – {tmp.max():.1f}°C  (avg {tmp.mean():.1f}°C) |")

    if "motor_temperature_c" in master.columns:
        mt = master["motor_temperature_c"].dropna()
        mt = mt[(mt > -39) & (mt < 174)]   # exclude sentinels
        if not mt.empty:
            lines.append(f"| Motor (IGBT) temp | {mt.min():.1f}°C – {mt.max():.1f}°C  (avg {mt.mean():.1f}°C) |")

    if "total_kwh_consumed" in master.columns:
        kwh = master["total_kwh_consumed"].dropna()
        if len(kwh) > 1:
            kwh_week = kwh.iloc[-1] - kwh.iloc[0]
            lines.append(f"| Energy consumed this week | {kwh_week:.1f} kWh |")

    if "fault_any" in master.columns:
        faults = int(master["fault_any"].sum())
        lines.append(f"| Fault events | {faults} |")

    lines.append("")

    # ── SOC model ───────────────────────────────────────────────────────────
    if not soc_df.empty and "y_soc_t_plus_300s" in soc_df.columns and "soc_pred_t300s" in soc_df.columns:
        y    = soc_df["y_soc_t_plus_300s"].astype(float)
        yhat = soc_df["soc_pred_t300s"].astype(float)
        mae  = float(np.mean(np.abs(y - yhat)))
        model_run = soc_df["model_run"].iloc[0] if "model_run" in soc_df.columns else "unknown"
        lines += [
            "### SOC Forecast Accuracy",
            "",
            f"| Model | Weekly MAE |",
            "|---|---|",
            f"| `{model_run}` | **{mae:.3f}%** |",
            "",
        ]

    # ── Battery health ───────────────────────────────────────────────────────
    if not health.empty and "health_score" in health.columns:
        hs = health["health_score"].dropna()
        lines += [
            "### Battery Health",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Avg health score | {hs.mean():.1f} / 100 |",
            f"| Min health score | {hs.min():.1f} / 100 |",
        ]
        lines.append("")

    # ── Alerts ───────────────────────────────────────────────────────────────
    if not alerts.empty:
        lines += ["### Alerts This Week", ""]
        if "rule_id" in alerts.columns:
            top = alerts["rule_id"].value_counts().head(5)
            lines += ["| Alert type | Count |", "|---|---|"]
            for rule, count in top.items():
                lines.append(f"| {rule} | {count:,} |")
        else:
            lines.append(f"Total alert rows: {len(alerts):,}")
        lines.append("")
    else:
        lines += ["### Alerts This Week", "", "No alerts triggered.", ""]

    # ── Daily breakdown ──────────────────────────────────────────────────────
    lines += ["### Daily Breakdown", "", "| Date | Data | SOC avg | Peak power | Health avg |", "|---|---|---|---|---|"]
    for dt in dates:
        day_path = MASTER_DIR / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"
        if not day_path.exists():
            lines.append(f"| {dt} | ✗ missing | — | — | — |")
            continue
        day = pd.read_parquet(day_path)
        soc_avg = f"{day['soc_pct'].mean():.1f}%" if "soc_pct" in day.columns else "—"
        pwr_max = f"{day['output_power_kw'].max():.1f} kW" if "output_power_kw" in day.columns else "—"

        h_path = BAT_DIR / f"dt={dt}" / f"scores_{vehicle_id}.parquet"
        if h_path.exists():
            h = pd.read_parquet(h_path)
            h_avg = f"{h['health_score'].mean():.1f}" if "health_score" in h.columns else "—"
        else:
            h_avg = "—"

        lines.append(f"| {dt} | ✓ | {soc_avg} | {pwr_max} | {h_avg} |")

    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default=None, help="Week ending date YYYY-MM-DD (default: today)")
    args = ap.parse_args()

    end  = date.fromisoformat(args.week) if args.week else date.today()
    dates = week_dates(end)

    print(f"\nGenerating weekly report: {dates[0]} → {dates[-1]}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    vehicles = sorted(MACHINE_TYPES.keys())

    lines = [
        f"# Weekly Battery Intelligence Report",
        f"",
        f"**Period:** {dates[0]} → {dates[-1]}",
        f"**Generated:** {date.today().isoformat()}",
        f"**Fleet:** {', '.join(f'{v} ({MACHINE_TYPES[v]})' for v in vehicles)}",
        f"",
        "---",
        "",
    ]

    for vehicle_id in vehicles:
        lines += vehicle_section(vehicle_id, dates)
        lines += ["---", ""]

    out = REPORTS_DIR / f"week_ending_{end.isoformat()}.md"
    out.write_text("\n".join(lines))
    print(f"Report → {out}")


if __name__ == "__main__":
    main()
