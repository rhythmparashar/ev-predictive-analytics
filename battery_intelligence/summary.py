"""
Aggregation and reporting functions for Battery Intelligence Module 1.

Produces:
  - daily_summary   : one row per dt + vehicle_id
  - module_summary  : one row per dt + vehicle_id + module_id
  - battery_twin_state : JSON snapshot of latest timestamp
  - markdown report
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from battery_intelligence.config import HEALTH_STATUS_PRIORITY, MODULES


def _df_to_md(df: pd.DataFrame) -> str:
    """Convert DataFrame to a markdown table string without requiring tabulate."""
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worst_status(statuses: pd.Series) -> str:
    """Return the worst (lowest priority) status from a series of status strings."""
    present = statuses.dropna().unique()
    if len(present) == 0:
        return "NO_DATA"
    return min(present, key=lambda s: HEALTH_STATUS_PRIORITY.get(s, 99))


def _mode_or_na(series: pd.Series) -> str:
    """Return most frequent value or 'N/A'."""
    m = series.dropna()
    if m.empty:
        return "N/A"
    return str(m.mode().iloc[0])


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

def compute_daily_summary(
    score_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per dt + vehicle_id.
    score_df must have battery_health_score, battery_health_status, and penalty columns.
    alerts_df is the long-format alert DataFrame from evaluate_rules().
    """
    rows: list[dict] = []

    group_cols = [c for c in ("dt", "vehicle_id") if c in score_df.columns]
    if not group_cols:
        raise ValueError("score_df must contain 'dt' and/or 'vehicle_id'")

    has_alerts = not alerts_df.empty

    for keys, grp in score_df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        n = len(grp)
        row = {**key_dict}

        # Health score stats
        row["avg_health_score"] = round(float(grp["battery_health_score"].mean()), 2) if "battery_health_score" in grp else None
        row["min_health_score"] = round(float(grp["battery_health_score"].min()), 2) if "battery_health_score" in grp else None

        if "battery_health_status" in grp.columns:
            row["worst_health_status"] = _worst_status(grp["battery_health_status"])
            status_counts = grp["battery_health_status"].value_counts()
            row["warning_minutes"] = round(float(status_counts.get("WARNING", 0)) / 60, 1)
            row["critical_minutes"] = round(float(status_counts.get("CRITICAL", 0)) / 60, 1)
            row["severe_minutes"] = round(float(status_counts.get("SEVERE", 0)) / 60, 1)
        else:
            row["worst_health_status"] = "NO_DATA"
            row["warning_minutes"] = 0.0
            row["critical_minutes"] = 0.0
            row["severe_minutes"] = 0.0

        # Alert counts from alerts_df
        if has_alerts and "vehicle_id" in alerts_df.columns:
            dt_val = key_dict.get("dt")
            vid = key_dict.get("vehicle_id")
            filt = alerts_df
            if dt_val and "dt" in alerts_df.columns:
                filt = filt[filt["dt"] == dt_val]
            if vid:
                filt = filt[filt["vehicle_id"] == vid]
            row["alert_count"] = len(filt)
            row["critical_alert_count"] = int((filt["severity"] == "CRITICAL").sum()) if "severity" in filt.columns else 0
        else:
            row["alert_count"] = 0
            row["critical_alert_count"] = 0

        # Top issue (mode of top_issue column)
        if "top_issue" in grp.columns:
            row["top_issue"] = _mode_or_na(grp["top_issue"])
        else:
            row["top_issue"] = "N/A"

        # Sensor extremes
        row["max_battery_temp_c"] = round(float(grp["max_battery_temp_c"].max()), 1) if "max_battery_temp_c" in grp else None
        row["max_voltage_imbalance_mv"] = round(float(grp["voltage_imbalance_mv"].max()), 1) if "voltage_imbalance_mv" in grp else None
        row["max_temp_imbalance_c"] = round(float(grp["temp_imbalance_c"].max()), 1) if "temp_imbalance_c" in grp else None
        row["max_current_a"] = round(float(grp["abs_current_a"].max()), 1) if "abs_current_a" in grp else None

        # Fault minutes
        if "fault_any" in grp.columns:
            row["fault_minutes"] = round(float(grp["fault_any"].sum()) / 60, 1)
        else:
            row["fault_minutes"] = 0.0

        # Weakest modules
        row["weakest_module_by_voltage"] = _mode_or_na(grp.get("weakest_module_by_voltage", pd.Series(dtype=str)))
        row["weakest_module_by_temp"] = _mode_or_na(grp.get("weakest_module_by_temp", pd.Series(dtype=str)))

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Module summary
# ---------------------------------------------------------------------------

def compute_module_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per dt + vehicle_id + module_id.
    Uses module_* columns from the master dataset.
    """
    group_cols = [c for c in ("dt", "vehicle_id") if c in score_df.columns]

    rows: list[dict] = []

    for keys, grp in score_df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        for m in MODULES:
            v_col = f"module_{m}_voltage_range"
            t_col = f"module_{m}_temp_range"
            lc_col = f"module_{m}_lowest_cell"
            hs_col = f"module_{m}_hottest_sensor"

            row = {**key_dict, "module_id": m}

            if v_col in grp.columns:
                v_mv = grp[v_col] * 1000
                row["max_voltage_range_mv"] = round(float(v_mv.max()), 2)
                row["avg_voltage_range_mv"] = round(float(v_mv.mean()), 2)
            else:
                row["max_voltage_range_mv"] = None
                row["avg_voltage_range_mv"] = None

            if t_col in grp.columns:
                row["max_temp_range_c"] = round(float(grp[t_col].max()), 2)
                row["avg_temp_range_c"] = round(float(grp[t_col].mean()), 2)
            else:
                row["max_temp_range_c"] = None
                row["avg_temp_range_c"] = None

            row["lowest_cell_frequency"] = _mode_or_na(grp[lc_col]) if lc_col in grp.columns else "N/A"
            row["hottest_sensor_frequency"] = _mode_or_na(grp[hs_col]) if hs_col in grp.columns else "N/A"

            # Risk scoring
            v_max = row["max_voltage_range_mv"] or 0.0
            t_max = row["max_temp_range_c"] or 0.0

            v_score = 40 if v_max > 130 else (20 if v_max > 90 else (10 if v_max > 60 else 0))
            t_score = 40 if t_max > 18 else (20 if t_max > 12 else (10 if t_max > 6 else 0))
            module_risk_score = min(v_score + t_score, 100)

            if v_max > 130 or t_max > 18:
                module_risk_status = "Critical"
            elif v_max > 90 or t_max > 15:
                module_risk_status = "Monitor"
            else:
                module_risk_status = "Healthy"

            row["module_risk_score"] = module_risk_score
            row["module_risk_status"] = module_risk_status

            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Battery twin state
# ---------------------------------------------------------------------------

def build_battery_twin_state(score_df: pd.DataFrame) -> dict:
    """
    Return a JSON-serialisable snapshot of the latest row in score_df
    formatted for a digital twin dashboard or 3D battery model.
    """
    if score_df.empty:
        return {}

    if "timestamp" in score_df.columns:
        row = score_df.sort_values("timestamp").iloc[-1]
    else:
        row = score_df.iloc[-1]

    def _v(col: str, default=None):
        val = row.get(col, default)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            # Convert numpy scalars to native Python types
            return val.item() if hasattr(val, "item") else val
        except Exception:
            return str(val)

    state = {
        "vehicle_id": _v("vehicle_id", "unknown"),
        "timestamp": str(_v("timestamp", "")),
        "battery": {
            "health_score": _v("battery_health_score"),
            "status": _v("battery_health_status"),
            "soc_pct": _v("soc_pct"),
            "max_temp_c": _v("max_battery_temp_c"),
            "voltage_imbalance_mv": _v("voltage_imbalance_mv"),
            "temp_imbalance_c": _v("temp_imbalance_c"),
            "abs_current_a": _v("abs_current_a"),
            "output_power_kw": _v("output_power_kw"),
            "imbalance_flag": bool(_v("imbalance_flag", 0)),
            "fault_active": bool(_v("fault_any", 0)),
            "weakest_module_by_voltage": _v("weakest_module_by_voltage"),
            "weakest_module_by_temp": _v("weakest_module_by_temp"),
            "top_issue": _v("top_issue"),
            "recommended_action": _v("recommended_action"),
            "business_impact": _v("business_impact"),
        },
        "penalties": {
            "thermal": _v("thermal_penalty"),
            "voltage_imbalance": _v("voltage_imbalance_penalty"),
            "temperature_imbalance": _v("temperature_imbalance_penalty"),
            "current_stress": _v("current_stress_penalty"),
            "weak_module": _v("weak_module_penalty"),
            "data_quality": _v("data_quality_penalty"),
            "fault": _v("fault_penalty"),
        },
    }

    # Per-module snapshot
    modules: list[dict] = []
    for m in MODULES:
        modules.append({
            "module_id": m,
            "voltage_range_mv": round(_v(f"module_{m}_voltage_range", 0.0) * 1000, 2),
            "temp_range_c": round(_v(f"module_{m}_temp_range", 0.0), 2),
            "lowest_cell": _v(f"module_{m}_lowest_cell"),
            "hottest_sensor": _v(f"module_{m}_hottest_sensor"),
        })
    state["modules"] = modules

    return state


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(
    score_df: pd.DataFrame,
    alerts_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    module_df: pd.DataFrame,
    twin_state: dict,
    avail_cols: list[str],
    missing_cols: list[str],
    output_dir: Path,
) -> str:
    """Generate the full Battery Engine Module 1 markdown report."""

    n_rows = len(score_df)
    ts_col = score_df["timestamp"] if "timestamp" in score_df.columns else None
    date_range = (
        f"{ts_col.min().date()} → {ts_col.max().date()}" if ts_col is not None else "N/A"
    )

    # Health score distribution
    score_col = score_df["battery_health_score"] if "battery_health_score" in score_df.columns else pd.Series(dtype=float)
    score_desc = score_col.describe()

    # Status distribution
    status_counts: dict = {}
    if "battery_health_status" in score_df.columns:
        status_counts = score_df["battery_health_status"].value_counts().to_dict()

    # Top alerts
    alert_types: dict = {}
    if not alerts_df.empty and "rule_id" in alerts_df.columns:
        alert_types = alerts_df["rule_id"].value_counts().head(10).to_dict()

    # Worst health days
    worst_days = pd.DataFrame()
    if not daily_df.empty and "avg_health_score" in daily_df.columns:
        worst_days = daily_df.nsmallest(5, "avg_health_score")[
            [c for c in ("dt", "vehicle_id", "avg_health_score", "min_health_score",
                         "worst_health_status", "top_issue") if c in daily_df.columns]
        ]

    # Weakest modules
    weak_modules = pd.DataFrame()
    if not module_df.empty and "module_risk_score" in module_df.columns:
        weak_modules = module_df[module_df["module_risk_status"] != "Healthy"].nlargest(
            10, "module_risk_score"
        )[
            [c for c in ("dt", "vehicle_id", "module_id",
                         "max_voltage_range_mv", "max_temp_range_c",
                         "module_risk_status", "module_risk_score") if c in module_df.columns]
        ]

    import json as _json

    lines: list[str] = [
        "# Battery Engine Module 1 Report",
        f"> Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"> Vehicle: EV01  ",
        "",
        "---",
        "",
        "## 1. Rows Processed",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total rows | {n_rows:,} |",
        f"| Date range | {date_range} |",
        f"| Total alert rows | {len(alerts_df):,} |",
        f"| Days in daily summary | {len(daily_df):,} |",
        "",
        "---",
        "",
        "## 2. Date Range",
        "",
        f"**{date_range}**",
        "",
        "---",
        "",
        "## 3. Columns Used",
        "",
        f"{len(avail_cols)} columns available from master dataset used in scoring:",
        "",
        "```",
        ", ".join(sorted(avail_cols)),
        "```",
        "",
        "---",
        "",
        "## 4. Missing Optional Columns",
        "",
    ]

    if missing_cols:
        lines += [f"- `{c}`" for c in sorted(missing_cols)]
    else:
        lines.append("All expected columns present.")

    lines += [
        "",
        "---",
        "",
        "## 5. Health Score Distribution",
        "",
        "| Stat | Value |",
        "|------|-------|",
        f"| Mean | {score_desc.get('mean', 0):.2f} |",
        f"| Std  | {score_desc.get('std', 0):.2f} |",
        f"| Min  | {score_desc.get('min', 0):.2f} |",
        f"| 25%  | {score_desc.get('25%', 0):.2f} |",
        f"| 50%  | {score_desc.get('50%', 0):.2f} |",
        f"| 75%  | {score_desc.get('75%', 0):.2f} |",
        f"| Max  | {score_desc.get('max', 0):.2f} |",
        "",
        "---",
        "",
        "## 6. Status Distribution",
        "",
        "| Status | Row Count | % of Total |",
        "|--------|-----------|------------|",
    ]

    for status, count in sorted(status_counts.items(), key=lambda x: HEALTH_STATUS_PRIORITY.get(x[0], 99)):
        pct = 100 * count / n_rows if n_rows > 0 else 0
        lines.append(f"| {status} | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 7. Top Alert Types",
        "",
        "| Rule ID | Alert Count |",
        "|---------|-------------|",
    ]

    for rule_id, count in alert_types.items():
        lines.append(f"| {rule_id} | {count:,} |")

    if not alert_types:
        lines.append("| No alerts triggered | — |")

    lines += [
        "",
        "---",
        "",
        "## 8. Worst Battery Health Days",
        "",
    ]

    if not worst_days.empty:
        lines.append(_df_to_md(worst_days))
    else:
        lines.append("No daily data available.")

    lines += [
        "",
        "---",
        "",
        "## 9. Weakest Modules Summary",
        "",
    ]

    if not weak_modules.empty:
        lines.append(_df_to_md(weak_modules))
    else:
        lines.append("All modules rated Healthy across all dates.")

    lines += [
        "",
        "---",
        "",
        "## 10. Example Battery Twin State",
        "",
        "```json",
        _json.dumps(twin_state, indent=2, default=str),
        "```",
        "",
        "---",
        "",
        "## 11. Output File Paths",
        "",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `{output_dir}/battery_health_scores.parquet` | Full health score + penalties per timestamp |",
        f"| `{output_dir}/battery_alerts.parquet` | Long-format rule alerts (triggered rows only) |",
        f"| `{output_dir}/battery_daily_summary.csv` | Daily aggregated health metrics per vehicle |",
        f"| `{output_dir}/battery_module_summary.csv` | Per-module daily risk scores |",
        f"| `{output_dir}/battery_engine_module1_summary.json` | Run metadata and key stats |",
        f"| `{output_dir}/battery_twin_state_sample.json` | Latest-timestamp twin state snapshot |",
        "",
    ]

    return "\n".join(lines)
