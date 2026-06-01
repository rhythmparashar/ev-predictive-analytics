"""
Battery Intelligence Engine — Module 2: Weak Cell / Weak Module / Hot Sensor Intelligence.

Four analyses, all vectorized:

1. weak_cell_profile      — which cells are chronically the lowest voltage in their module
2. hot_sensor_profile     — which sensors are chronically the hottest in their module
3. module_chronic_risk    — per-module chronic voltage and thermal risk ranking
4. cell_delta_profile     — per-cell mean deviation and outlier frequency from gold cell_features

Data sources:
  - master dataset  → analyses 1, 2, 3  (module_*_lowest_cell, module_*_hottest_sensor, module_*_voltage_range, etc.)
  - gold cell_features → analysis 4    (M*_C*_delta, M*_C*_is_outlier per timestamp)

Voltage deltas in gold cell_features are in VOLTS — multiplied ×1000 for mV display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from battery_intelligence.config import MODULE2_THRESHOLDS as TH, MODULES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pct(count: int, total: int) -> float:
    return round(100.0 * count / total, 2) if total > 0 else 0.0


def _module_from_cell_id(cell_id: str) -> int | None:
    """Extract module number from cell id like 'M3_C15' → 3."""
    try:
        return int(str(cell_id).split("_")[0].replace("M", ""))
    except (ValueError, IndexError):
        return None


def _module_from_sensor_id(sensor_id: str) -> int | None:
    """Extract module number from sensor id like 'M2_T7' → 2."""
    try:
        return int(str(sensor_id).split("_")[0].replace("M", ""))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Analysis 1 — Weak Cell Profile
# ---------------------------------------------------------------------------

def compute_weak_cell_profile(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify cells that are chronically the lowest voltage in their module.

    Uses module_*_lowest_cell columns from the master dataset.
    A cell is 'chronic' if it appears as the module's lowest voltage cell
    more than MODULE2_THRESHOLDS['chronic_weak_cell_pct']% of all timestamps.

    Also cross-references worst_cell_id (pack-level lowest) for each cell.

    Returns
    -------
    pd.DataFrame with columns:
        cell_id, module_id, times_lowest_in_module, pct_lowest_in_module,
        times_worst_in_pack, pct_worst_in_pack, chronic_weak, weakness_status
    """
    total = len(master_df)
    rows: list[dict] = []

    for m in MODULES:
        col = f"module_{m}_lowest_cell"
        if col not in master_df.columns:
            continue
        counts = master_df[col].dropna().value_counts()
        for cell_id, cnt in counts.items():
            rows.append({
                "cell_id": str(cell_id),
                "module_id": m,
                "times_lowest_in_module": int(cnt),
                "pct_lowest_in_module": _pct(cnt, total),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "cell_id", "module_id", "times_lowest_in_module",
            "pct_lowest_in_module", "times_worst_in_pack",
            "pct_worst_in_pack", "chronic_weak", "weakness_status",
        ])

    df = pd.DataFrame(rows)

    # Pack-level worst cross-reference
    if "worst_cell_id" in master_df.columns:
        pack_counts = master_df["worst_cell_id"].dropna().value_counts().to_dict()
    else:
        pack_counts = {}

    df["times_worst_in_pack"] = df["cell_id"].map(pack_counts).fillna(0).astype(int)
    df["pct_worst_in_pack"] = df["times_worst_in_pack"].apply(lambda x: _pct(x, total))

    threshold = TH["chronic_weak_cell_pct"]
    df["chronic_weak"] = df["pct_lowest_in_module"] > threshold

    df["weakness_status"] = np.select(
        [
            df["pct_lowest_in_module"] > threshold * 3,
            df["pct_lowest_in_module"] > threshold * 2,
            df["pct_lowest_in_module"] > threshold,
        ],
        ["Critical", "Monitor", "Watch"],
        default="Normal",
    )

    return df.sort_values("pct_lowest_in_module", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 2 — Hot Sensor Profile
# ---------------------------------------------------------------------------

def compute_hot_sensor_profile(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify temperature sensors that are chronically the hottest in their module.

    Uses module_*_hottest_sensor columns from the master dataset.
    A sensor is 'chronic' if it appears as the module's hottest sensor
    more than MODULE2_THRESHOLDS['chronic_hot_sensor_pct']% of all timestamps.

    Returns
    -------
    pd.DataFrame with columns:
        sensor_id, module_id, times_hottest_in_module, pct_hottest_in_module,
        chronic_hot, heat_status
    """
    total = len(master_df)
    rows: list[dict] = []

    for m in MODULES:
        col = f"module_{m}_hottest_sensor"
        if col not in master_df.columns:
            continue
        counts = master_df[col].dropna().value_counts()
        for sensor_id, cnt in counts.items():
            rows.append({
                "sensor_id": str(sensor_id),
                "module_id": m,
                "times_hottest_in_module": int(cnt),
                "pct_hottest_in_module": _pct(cnt, total),
            })

    if not rows:
        return pd.DataFrame(columns=[
            "sensor_id", "module_id", "times_hottest_in_module",
            "pct_hottest_in_module", "chronic_hot", "heat_status",
        ])

    df = pd.DataFrame(rows)

    threshold = TH["chronic_hot_sensor_pct"]
    df["chronic_hot"] = df["pct_hottest_in_module"] > threshold

    df["heat_status"] = np.select(
        [
            df["pct_hottest_in_module"] > threshold * 3,
            df["pct_hottest_in_module"] > threshold * 2,
            df["pct_hottest_in_module"] > threshold,
        ],
        ["Critical", "Monitor", "Watch"],
        default="Normal",
    )

    return df.sort_values("pct_hottest_in_module", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis 3 — Module Chronic Risk
# ---------------------------------------------------------------------------

def compute_module_chronic_risk(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank all 5 modules by chronic voltage and thermal risk across the full dataset.

    For each module computes:
      - avg and max voltage_range (mV) across all timestamps
      - avg and max temp_range (°C) across all timestamps
      - pct of timestamps where this module had the highest voltage range (weakest)
      - pct of timestamps where this module was hottest_module_id
      - chronic_voltage_risk, chronic_thermal_risk flags
      - overall module_chronic_risk_score (0–100) and module_chronic_risk_status

    Returns
    -------
    pd.DataFrame with one row per module, sorted by module_chronic_risk_score desc.
    """
    total = len(master_df)

    # Find weakest module by voltage at each timestamp (highest voltage range)
    v_range_cols = [
        f"module_{m}_voltage_range"
        for m in MODULES
        if f"module_{m}_voltage_range" in master_df.columns
    ]
    if v_range_cols:
        v_df = master_df[v_range_cols].fillna(0.0)
        weakest_v_per_row = (
            v_df.idxmax(axis=1)
            .str.extract(r"module_(\d+)_voltage_range", expand=False)
            .astype(float)
        )
    else:
        weakest_v_per_row = pd.Series(dtype=float)

    # Find hottest module at each timestamp from hottest_module_id column
    if "hottest_module_id" in master_df.columns:
        hottest_module_per_row = pd.to_numeric(
            master_df["hottest_module_id"], errors="coerce"
        )
    else:
        hottest_module_per_row = pd.Series(dtype=float)

    rows: list[dict] = []

    for m in MODULES:
        row: dict = {"module_id": m}

        # Voltage range stats
        v_col = f"module_{m}_voltage_range"
        if v_col in master_df.columns:
            v_mv = master_df[v_col].dropna() * 1000.0
            row["avg_voltage_range_mv"] = round(float(v_mv.mean()), 2)
            row["max_voltage_range_mv"] = round(float(v_mv.max()), 2)
            row["p95_voltage_range_mv"] = round(float(v_mv.quantile(0.95)), 2)
        else:
            row["avg_voltage_range_mv"] = None
            row["max_voltage_range_mv"] = None
            row["p95_voltage_range_mv"] = None

        # Temp range stats
        t_col = f"module_{m}_temp_range"
        if t_col in master_df.columns:
            t = master_df[t_col].dropna()
            row["avg_temp_range_c"] = round(float(t.mean()), 2)
            row["max_temp_range_c"] = round(float(t.max()), 2)
            row["p95_temp_range_c"] = round(float(t.quantile(0.95)), 2)
        else:
            row["avg_temp_range_c"] = None
            row["max_temp_range_c"] = None
            row["p95_temp_range_c"] = None

        # Mean voltage stats
        vm_col = f"module_{m}_voltage_mean"
        if vm_col in master_df.columns:
            vm = master_df[vm_col].dropna()
            row["avg_cell_voltage_v"] = round(float(vm.mean()), 4)
        else:
            row["avg_cell_voltage_v"] = None

        # Frequency as weakest module (highest voltage range)
        if not weakest_v_per_row.empty:
            pct_weakest = _pct(int((weakest_v_per_row == m).sum()), total)
        else:
            pct_weakest = 0.0
        row["pct_time_weakest_voltage"] = pct_weakest

        # Frequency as hottest module
        if not hottest_module_per_row.empty:
            pct_hottest = _pct(int((hottest_module_per_row == m).sum()), total)
        else:
            pct_hottest = 0.0
        row["pct_time_hottest"] = pct_hottest

        # Chronic risk flags
        row["chronic_voltage_risk"] = (
            (row["avg_voltage_range_mv"] or 0) > TH["chronic_module_voltage_mv"]
            or pct_weakest > TH["chronic_weak_module_pct"]
        )
        row["chronic_thermal_risk"] = (
            (row["avg_temp_range_c"] or 0) > TH["chronic_module_temp_c"]
            or pct_hottest > TH["chronic_hot_module_pct"]
        )

        # Risk score: voltage component (0–50) + thermal component (0–50)
        avg_v = row["avg_voltage_range_mv"] or 0.0
        avg_t = row["avg_temp_range_c"] or 0.0

        v_score = (
            50 if avg_v > 150 else
            40 if avg_v > 120 else
            30 if avg_v > 80 else
            15 if avg_v > 40 else
            0
        )
        t_score = (
            50 if avg_t > 20 else
            40 if avg_t > 15 else
            30 if avg_t > 10 else
            15 if avg_t > 6 else
            0
        )

        # Frequency bonus
        freq_bonus = min(int(pct_weakest / 10) * 5 + int(pct_hottest / 10) * 5, 20)
        risk_score = min(v_score + t_score + freq_bonus, 100)

        row["module_chronic_risk_score"] = risk_score
        row["module_chronic_risk_status"] = (
            "Critical" if risk_score >= TH["module_risk_critical"] else
            "Monitor"  if risk_score >= TH["module_risk_monitor"] else
            "Healthy"
        )

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("module_chronic_risk_score", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Analysis 4 — Cell Delta Profile (from gold cell_features)
# ---------------------------------------------------------------------------

def compute_cell_delta_profile(
    cell_features_dir: Path,
    vehicle_id: str = "EV01",
) -> pd.DataFrame:
    """
    Load gold cell_features partitions and compute per-cell statistics.

    For each of the 180 cells (M1_C1 ... M5_C36):
      - mean_delta_mv     : average deviation from module mean (V × 1000)
      - std_delta_mv      : std of deviation (V × 1000)
      - outlier_frequency : % of timestamps where cell was flagged as outlier
      - weakness_score    : composite score for ranking cells within module
      - weakness_status   : Healthy / Watch / Monitor / Critical

    Processes partitions one at a time to avoid loading 540 columns × 1.4M rows at once.

    Returns
    -------
    pd.DataFrame with one row per cell.
    """
    partitions = sorted(cell_features_dir.glob(f"dt=*/vehicle_id={vehicle_id}.parquet"))

    if not partitions:
        return pd.DataFrame()

    # Identify delta and is_outlier columns from first partition
    sample = pd.read_parquet(partitions[0])
    delta_cols = [c for c in sample.columns if c.endswith("_delta")]
    outlier_cols = [c for c in sample.columns if c.endswith("_is_outlier")]

    if not delta_cols:
        return pd.DataFrame()

    # Running accumulators: sum_delta, sum_sq_delta, sum_outlier, count per cell
    sum_delta: dict[str, float] = {c: 0.0 for c in delta_cols}
    sum_sq_delta: dict[str, float] = {c: 0.0 for c in delta_cols}
    sum_outlier: dict[str, int] = {c: 0 for c in outlier_cols}
    n_total: int = 0

    for path in partitions:
        cols_to_read = ["timestamp"] + delta_cols + outlier_cols
        cols_to_read = [c for c in cols_to_read if c in sample.columns]
        part = pd.read_parquet(path, columns=cols_to_read)
        n = len(part)
        if n == 0:
            continue
        n_total += n
        for c in delta_cols:
            if c in part.columns:
                vals = part[c].fillna(0.0).values
                sum_delta[c] += float(vals.sum())
                sum_sq_delta[c] += float((vals ** 2).sum())
        for c in outlier_cols:
            if c in part.columns:
                sum_outlier[c] += int(part[c].fillna(0).astype(int).sum())

    if n_total == 0:
        return pd.DataFrame()

    # Build per-cell summary
    rows: list[dict] = []

    for delta_col in delta_cols:
        # delta_col format: M1_C1_delta → cell_id = M1_C1
        cell_id = delta_col.replace("_delta", "")
        outlier_col = f"{cell_id}_is_outlier"

        parts = cell_id.split("_")
        module_id = int(parts[0].replace("M", "")) if parts else None
        cell_num = int(parts[1].replace("C", "")) if len(parts) > 1 else None

        mean_delta_mv = round((sum_delta[delta_col] / n_total) * 1000.0, 4)
        variance = (sum_sq_delta[delta_col] / n_total) - (sum_delta[delta_col] / n_total) ** 2
        std_delta_mv = round(float(np.sqrt(max(variance, 0.0))) * 1000.0, 4)

        outlier_freq = (
            round(100.0 * sum_outlier[outlier_col] / n_total, 2)
            if outlier_col in sum_outlier else 0.0
        )

        # Weakness score: lower mean_delta (more negative) = weaker cell
        # Combine normalised negativity + outlier frequency
        negativity = max(-mean_delta_mv, 0.0)   # 0 if above mean, positive if below
        weakness_score = round(min(negativity * 2 + outlier_freq * 3, 100.0), 2)

        status = (
            "Critical" if weakness_score > 50 else
            "Monitor"  if weakness_score > 20 else
            "Watch"    if weakness_score > 8  else
            "Healthy"
        )

        rows.append({
            "cell_id": cell_id,
            "module_id": module_id,
            "cell_num": cell_num,
            "mean_delta_mv": mean_delta_mv,
            "std_delta_mv": std_delta_mv,
            "outlier_frequency_pct": outlier_freq,
            "weakness_score": weakness_score,
            "weakness_status": status,
        })

    df = pd.DataFrame(rows)

    # Rank within module (1 = weakest)
    df["rank_in_module"] = (
        df.groupby("module_id")["mean_delta_mv"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    return df.sort_values(["module_id", "mean_delta_mv"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |"]
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def generate_module2_report(
    weak_cell_df: pd.DataFrame,
    hot_sensor_df: pd.DataFrame,
    module_risk_df: pd.DataFrame,
    cell_delta_df: pd.DataFrame,
    n_master_rows: int,
    n_cell_feature_partitions: int,
    output_dir: Path,
) -> str:
    import datetime

    chronic_cells = int(weak_cell_df["chronic_weak"].sum()) if not weak_cell_df.empty else 0
    chronic_sensors = int(hot_sensor_df["chronic_hot"].sum()) if not hot_sensor_df.empty else 0
    critical_modules = (
        int((module_risk_df["module_chronic_risk_status"] == "Critical").sum())
        if not module_risk_df.empty else 0
    )
    critical_cells_delta = (
        int((cell_delta_df["weakness_status"] == "Critical").sum())
        if not cell_delta_df.empty else 0
    )

    worst_cell = (
        weak_cell_df.iloc[0]["cell_id"]
        if not weak_cell_df.empty else "N/A"
    )
    hottest_sensor = (
        hot_sensor_df.iloc[0]["sensor_id"]
        if not hot_sensor_df.empty else "N/A"
    )
    riskiest_module = (
        int(module_risk_df.iloc[0]["module_id"])
        if not module_risk_df.empty else "N/A"
    )

    top_weak_cells = weak_cell_df[weak_cell_df["chronic_weak"]].head(10) if not weak_cell_df.empty else weak_cell_df
    top_hot_sensors = hot_sensor_df[hot_sensor_df["chronic_hot"]].head(10) if not hot_sensor_df.empty else hot_sensor_df
    worst_delta_cells = cell_delta_df.head(10) if not cell_delta_df.empty else cell_delta_df

    lines = [
        "# Battery Intelligence Engine — Module 2 Report",
        f"> Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"> Vehicle: EV01",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Master dataset rows analysed | {n_master_rows:,} |",
        f"| Cell feature partitions loaded | {n_cell_feature_partitions} |",
        f"| Chronically weak cells | {chronic_cells} |",
        f"| Chronically hot sensors | {chronic_sensors} |",
        f"| Modules at Critical chronic risk | {critical_modules} |",
        f"| Cells with Critical delta weakness | {critical_cells_delta} |",
        f"| Most frequently weakest cell | {worst_cell} |",
        f"| Most frequently hottest sensor | {hottest_sensor} |",
        f"| Riskiest module | Module {riskiest_module} |",
        "",
        "---",
        "",
        "## 1. Module Chronic Risk Ranking",
        "",
        "Modules ranked by chronic voltage and thermal risk across the full date range.",
        "",
        _md_table(module_risk_df[[
            "module_id", "avg_voltage_range_mv", "max_voltage_range_mv",
            "avg_temp_range_c", "max_temp_range_c",
            "pct_time_weakest_voltage", "pct_time_hottest",
            "chronic_voltage_risk", "chronic_thermal_risk",
            "module_chronic_risk_score", "module_chronic_risk_status",
        ]]),
        "",
        "---",
        "",
        "## 2. Chronically Weak Cells (Lowest Voltage Frequency)",
        "",
        f"Cells appearing as the lowest voltage in their module > {TH['chronic_weak_cell_pct']}% of all timestamps.",
        f"Total chronic weak cells: **{chronic_cells}**",
        "",
        _md_table(top_weak_cells[[
            "cell_id", "module_id", "times_lowest_in_module",
            "pct_lowest_in_module", "times_worst_in_pack",
            "pct_worst_in_pack", "weakness_status",
        ]]),
        "",
        "---",
        "",
        "## 3. Chronically Hot Sensors (Hottest Sensor Frequency)",
        "",
        f"Sensors appearing as the hottest in their module > {TH['chronic_hot_sensor_pct']}% of all timestamps.",
        f"Total chronic hot sensors: **{chronic_sensors}**",
        "",
        _md_table(top_hot_sensors[[
            "sensor_id", "module_id", "times_hottest_in_module",
            "pct_hottest_in_module", "heat_status",
        ]]),
        "",
        "---",
        "",
        "## 4. Cell Voltage Delta Profile (Weakest Cells by Deviation)",
        "",
        "Per-cell mean deviation from module mean voltage across all timestamps.",
        "Negative mean_delta_mv = cell consistently below its module mean (weaker cell).",
        "",
        _md_table(worst_delta_cells[[
            "cell_id", "module_id", "cell_num",
            "mean_delta_mv", "std_delta_mv",
            "outlier_frequency_pct", "weakness_score",
            "weakness_status", "rank_in_module",
        ]]),
        "",
        "---",
        "",
        "## 5. Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `{output_dir}/weak_cell_profile.csv` | Per-cell frequency as module lowest — all cells |",
        f"| `{output_dir}/hot_sensor_profile.csv` | Per-sensor frequency as module hottest — all sensors |",
        f"| `{output_dir}/module_chronic_risk.csv` | Per-module chronic risk scores and flags |",
        f"| `{output_dir}/cell_delta_profile.csv` | Per-cell mean delta, std, outlier frequency from gold features |",
        f"| `{output_dir}/battery_engine_module2_summary.json` | Run metadata |",
        f"| `reports/battery_engine_module2_report.md` | This report |",
        "",
    ]

    return "\n".join(lines)
