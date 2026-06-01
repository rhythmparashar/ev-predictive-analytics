# reports/cell_health_report.py
"""
Cell health report — one row per module per day.

Reads gold_cells (module_features + pack_features) for a given date
and produces daily_health_reports with module status labels.

Output per module per day:
  dt, vehicle_id, module_id,
  status,                    (Healthy / Monitor / Imbalance Rising /
                              Thermal Hotspot / Degraded Cell / Critical)
  avg_voltage_std,
  max_voltage_range,
  avg_temp_range,
  max_temp_spike,
  worst_cell_id,
  pct_time_imbalanced,
  trend_direction            (stable / worsening / improving)

Dashboard output example:
  Battery Pack Health — 2026-03-10
    Module 1 → Healthy
    Module 2 → Healthy
    Module 3 → Imbalance Rising
    Module 4 → Thermal Hotspot
    Module 5 → Healthy

Called by run_cells.py report command.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ingestion.io import write_parquet_atomic


# -------------------------------------------------
# YAML loader
# -------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Load gold_cells outputs
# -------------------------------------------------

def _parquet_path(base: Path, dt: str, vehicle_id: str) -> Path:
    return base / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"


def load_module_features(
    gold_module_features_dir: Path,
    dt: str,
    vehicle_id: str,
) -> pd.DataFrame:
    path = _parquet_path(gold_module_features_dir, dt, vehicle_id)
    if not path.exists():
        raise FileNotFoundError(
            f"module_features not found: {path}\n"
            f"Run: python run_cells.py gold --dt {dt} first."
        )
    return pd.read_parquet(path)


def load_pack_features(
    gold_pack_features_dir: Path,
    dt: str,
    vehicle_id: str,
) -> pd.DataFrame:
    path = _parquet_path(gold_pack_features_dir, dt, vehicle_id)
    if not path.exists():
        raise FileNotFoundError(
            f"pack_features not found: {path}\n"
            f"Run: python run_cells.py gold --dt {dt} first."
        )
    return pd.read_parquet(path)


def list_vehicles(
    gold_module_features_dir: Path,
    dt: str,
) -> list[str]:
    d = gold_module_features_dir / f"dt={dt}"
    if not d.exists():
        return []
    return [
        p.stem.replace("vehicle_id=", "")
        for p in sorted(d.glob("vehicle_id=*.parquet"))
    ]


# -------------------------------------------------
# Per-module daily stats
# -------------------------------------------------

def compute_module_daily_stats(
    module_feat: pd.DataFrame,
    module: int,
    cfg: dict,
) -> dict:
    """
    Compute daily summary stats for one module.

    Returns dict of stats — status assigned separately.
    """

    v_std_col = f"module_{module}_voltage_std"
    v_range_col = f"module_{module}_voltage_range"
    t_mean_col = f"module_{module}_temp_mean"
    t_range_col = f"module_{module}_temp_range"
    lowest_col = f"module_{module}_lowest_cell"
    hottest_col = f"module_{module}_hottest_sensor"

    stats: dict = {"module_id": module}

    # -------------------------------------------------
    # Voltage stats
    # -------------------------------------------------

    if v_std_col in module_feat.columns:
        stats["avg_voltage_std"] = float(module_feat[v_std_col].mean())
    else:
        stats["avg_voltage_std"] = None

    if v_range_col in module_feat.columns:
        stats["max_voltage_range"] = float(module_feat[v_range_col].max())
        warn_v = float(cfg["voltage"]["imbalance_warn_mv"]) / 1000.0
        stats["pct_time_imbalanced"] = float((module_feat[v_range_col] > warn_v).mean())
    else:
        stats["max_voltage_range"] = None
        stats["pct_time_imbalanced"] = None

    if lowest_col in module_feat.columns:
        non_null = module_feat[lowest_col].dropna()
        stats["worst_cell_id"] = non_null.mode().iloc[0] if not non_null.empty else None
    else:
        stats["worst_cell_id"] = None

    # -------------------------------------------------
    # Temperature stats
    # -------------------------------------------------

    if t_mean_col in module_feat.columns:
        stats["avg_temp_mean"] = float(module_feat[t_mean_col].mean())
        stats["max_temp_mean"] = float(module_feat[t_mean_col].max())
    else:
        stats["avg_temp_mean"] = None
        stats["max_temp_mean"] = None

    if t_range_col in module_feat.columns:
        stats["avg_temp_range"] = float(module_feat[t_range_col].mean())
        stats["max_temp_spike"] = float(module_feat[t_range_col].max())
    else:
        stats["avg_temp_range"] = None
        stats["max_temp_spike"] = None

    if hottest_col in module_feat.columns:
        non_null = module_feat[hottest_col].dropna()
        stats["hottest_sensor_id"] = non_null.mode().iloc[0] if not non_null.empty else None
    else:
        stats["hottest_sensor_id"] = None

    return stats


# -------------------------------------------------
# Status assignment
# -------------------------------------------------

def assign_module_status(stats: dict, cfg: dict) -> str:
    """
    Assign module health status label from daily stats.

    Checks in order — first match wins:
      Critical → Degraded Cell → Imbalance Rising →
      Thermal Hotspot → Monitor → Healthy
    """

    v_cfg = cfg["voltage"]
    t_cfg = cfg["temperature"]
    s_cfg = cfg["status"]
    hr_cfg = cfg["health_report"]

    signals_degraded = 0

    alert_v = float(v_cfg["imbalance_alert_mv"]) / 1000.0
    if (stats.get("max_voltage_range") or 0) > alert_v:
        signals_degraded += 1

    if (stats.get("max_temp_spike") or 0) > float(t_cfg["hotspot_delta_c"]):
        signals_degraded += 1

    if (stats.get("pct_time_imbalanced") or 0) > float(hr_cfg["pct_time_imbalanced_alert"]):
        signals_degraded += 1

    # -------------------------------------------------
    # Critical
    # -------------------------------------------------
    if signals_degraded >= int(s_cfg["critical"]["min_signals_degraded"]):
        return "Critical"

    # -------------------------------------------------
    # Degraded Cell
    # Heuristic for now: severe imbalance with a persistent dominant worst cell
    # -------------------------------------------------
    degraded_v = (stats.get("max_voltage_range") or 0) > alert_v
    has_named_worst_cell = stats.get("worst_cell_id") is not None
    if degraded_v and has_named_worst_cell:
        return "Degraded Cell"

    # -------------------------------------------------
    # Imbalance Rising
    # -------------------------------------------------
    alert_pct = float(hr_cfg["pct_time_imbalanced_alert"])
    if (stats.get("pct_time_imbalanced") or 0) > alert_pct:
        return "Imbalance Rising"

    # -------------------------------------------------
    # Thermal Hotspot
    # -------------------------------------------------
    if (
        (stats.get("max_temp_spike") or 0) > float(t_cfg["hotspot_delta_c"])
        or (stats.get("max_temp_mean") or 0) > float(t_cfg["alert_c"])
    ):
        return "Thermal Hotspot"

    # -------------------------------------------------
    # Monitor
    # -------------------------------------------------
    warn_v = float(v_cfg["imbalance_warn_mv"]) / 1000.0
    warn_pct = float(hr_cfg["pct_time_imbalanced_warn"])

    monitor = any([
        (stats.get("max_voltage_range") or 0) > warn_v,
        (stats.get("pct_time_imbalanced") or 0) > warn_pct,
        (stats.get("avg_temp_range") or 0) > float(t_cfg["hotspot_delta_c"]),
        (stats.get("avg_temp_mean") or 0) > float(t_cfg["warn_c"]),
    ])

    if monitor:
        return "Monitor"

    # -------------------------------------------------
    # Healthy
    # -------------------------------------------------
    return "Healthy"


# -------------------------------------------------
# Trend direction
# -------------------------------------------------

def compute_trend_direction(
    gold_daily_health_dir: Path,
    vehicle_id: str,
    module: int,
    current_dt: str,
    trend_window_days: int,
    min_std_increase_pct: float,
) -> str:
    """
    Compare current day avg_voltage_std against
    the rolling mean of the previous trend_window_days.

    Returns: 'worsening' | 'improving' | 'stable'

    If insufficient history → 'stable'
    """

    available = sorted(gold_daily_health_dir.glob(f"dt=*/vehicle_id={vehicle_id}.parquet"))

    history = []
    for p in available:
        file_dt = p.parent.name.replace("dt=", "")
        if file_dt < current_dt:
            history.append(p)

    if len(history) < 2:
        return "stable"

    recent = history[-trend_window_days:]

    std_values = []
    for p in recent:
        df = pd.read_parquet(p)
        row = df[df["module_id"] == module]
        if not row.empty and "avg_voltage_std" in row.columns:
            val = row["avg_voltage_std"].iloc[0]
            if pd.notna(val):
                std_values.append(float(val))

    if len(std_values) < 2:
        return "stable"

    baseline = std_values[0]
    latest = std_values[-1]

    if baseline == 0:
        return "stable"

    change_pct = ((latest - baseline) / baseline) * 100

    if change_pct > min_std_increase_pct:
        return "worsening"
    elif change_pct < -min_std_increase_pct:
        return "improving"
    else:
        return "stable"


# -------------------------------------------------
# Print dashboard
# -------------------------------------------------

def print_dashboard(
    dt: str,
    vehicle_id: str,
    report_rows: list[dict],
) -> None:
    """Print module health status to terminal."""

    print(f"\n── Battery Pack Health  {vehicle_id}  {dt} ───────────────")
    for row in sorted(report_rows, key=lambda x: x["module_id"]):
        status = row["status"]
        module = row["module_id"]
        padding = " " * max(1, 20 - len(status))
        pct_imbal = row.get("pct_time_imbalanced") or 0.0
        trend = row.get("trend_direction", "stable")
        print(
            f"  Module {module} → {status}{padding}"
            f"imbalanced={pct_imbal:.1%}  trend={trend}"
        )
    print()


# -------------------------------------------------
# Main entry point
# -------------------------------------------------

def run_cell_health_report(
    dt: str,
    gold_module_features_dir: Path,
    gold_pack_features_dir: Path,
    gold_daily_health_dir: Path,
    voltage_schema_path: Path,
    cell_cfg_path: Path,
    reports_dir: Path,
    vehicle_id_filter: str | None = None,
) -> dict:
    """
    Build daily health report for all vehicles on a given date.

    Returns manifest dict.
    """

    v_schema = load_yaml(voltage_schema_path)
    cfg = load_yaml(cell_cfg_path)

    modules = v_schema["modules"]
    trend_window_days = int(cfg["health_report"]["trend_window_days"])
    min_std_increase_pct = float(cfg["status"]["imbalance_rising"]["min_std_increase_pct"])
    min_rows = int(cfg["health_report"]["min_rows_for_report"])

    vehicles = list_vehicles(gold_module_features_dir, dt)
    if vehicle_id_filter:
        vehicles = [v for v in vehicles if v == vehicle_id_filter]

    if not vehicles:
        raise FileNotFoundError(
            f"No module_features found for dt={dt}.\n"
            f"Run: python run_cells.py gold --dt {dt} first."
        )

    manifest = {
        "dt": dt,
        "pipeline": "cell_report",
        "vehicles_processed": [],
        "per_vehicle": {},
        "status": "success",
    }

    for vehicle_id in vehicles:
        # -------------------------------------------------
        # Load gold features
        # -------------------------------------------------

        module_feat = load_module_features(
            gold_module_features_dir, dt, vehicle_id
        )
        _ = load_pack_features(
            gold_pack_features_dir, dt, vehicle_id
        )

        if len(module_feat) < min_rows:
            manifest["per_vehicle"][vehicle_id] = {
                "status": "skipped",
                "reason": f"Only {len(module_feat)} rows (min={min_rows})",
            }
            continue

        # -------------------------------------------------
        # Build per-module report rows
        # -------------------------------------------------

        report_rows = []

        for m in modules:
            stats = compute_module_daily_stats(
                module_feat=module_feat,
                module=m,
                cfg=cfg,
            )

            status = assign_module_status(stats, cfg)

            trend = compute_trend_direction(
                gold_daily_health_dir=gold_daily_health_dir,
                vehicle_id=vehicle_id,
                module=m,
                current_dt=dt,
                trend_window_days=trend_window_days,
                min_std_increase_pct=min_std_increase_pct,
            )

            report_rows.append({
                "dt": dt,
                "vehicle_id": vehicle_id,
                "module_id": m,
                "status": status,
                "avg_voltage_std": stats.get("avg_voltage_std"),
                "max_voltage_range": stats.get("max_voltage_range"),
                "avg_temp_range": stats.get("avg_temp_range"),
                "max_temp_spike": stats.get("max_temp_spike"),
                "worst_cell_id": stats.get("worst_cell_id"),
                "hottest_sensor_id": stats.get("hottest_sensor_id"),
                "pct_time_imbalanced": stats.get("pct_time_imbalanced"),
                "trend_direction": trend,
            })

        # -------------------------------------------------
        # Write daily_health_reports parquet
        # -------------------------------------------------

        report_df = pd.DataFrame(report_rows)

        out_path = (
            gold_daily_health_dir
            / f"dt={dt}"
            / f"vehicle_id={vehicle_id}.parquet"
        )
        write_parquet_atomic(report_df, out_path)

        # -------------------------------------------------
        # Print dashboard to terminal
        # -------------------------------------------------

        print_dashboard(dt, vehicle_id, report_rows)

        # -------------------------------------------------
        # Manifest
        # -------------------------------------------------

        manifest["vehicles_processed"].append(vehicle_id)

        manifest["per_vehicle"][vehicle_id] = {
            "status": "ok",
            "report_path": str(out_path),
            "modules": len(report_rows),
            "status_counts": report_df["status"].value_counts().to_dict(),
        }

    # -------------------------------------------------
    # Write manifest JSON
    # -------------------------------------------------

    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"dt={dt}.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(
        f"[cell_report] dt={dt} "
        f"vehicles={len(manifest['vehicles_processed'])} "
        f"report={report_path}"
    )

    return manifest