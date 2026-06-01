# EV Telemetry ML — Cell Health Pipeline
> **Status: Locked Architecture — Ready to Build**
> **Phase:** Cell Health Analytics (parallel to SOC forecast, does not modify existing pipeline)

---

## Overview

```
Phase 1 ✅   raw → silver                (VCU ingestion, fault alignment, trip labeling)
Phase 2 ✅   silver → gold               (window features, trip features, daily stats)
Phase 3 ✅   gold → SOC model            (XGBoost SOC forecast, MAE 0.538%)
Phase 4 ⏳   SOC model improvement       (SHAP, tuning, multi-vehicle)
Phase 5 🔒   Cell health pipeline        (THIS DOCUMENT)
Phase 6 ⏳   Cell health scoring/model   (when fault/replacement labels exist)
```

**Core principle:** Existing SOC pipeline is completely untouched. Cell pipeline runs in parallel.

---

## New Pipeline Commands

```bash
# Existing — unchanged
python run.py ingest --dt YYYY-MM-DD          # raw → silver (VCU)
python run.py gold   --dt YYYY-MM-DD          # silver → gold (VCU)
python run.py eval   --task soc_forecast --dt YYYY-MM-DD

# New — cell pipeline
python run.py cell_ingest --dt YYYY-MM-DD     # raw_cell_* → silver_cells
python run.py cell_gold   --dt YYYY-MM-DD     # silver_cells → gold_cells
python run.py cell_report --dt YYYY-MM-DD     # gold_cells → daily_health_reports
```

Each command is independent. Cell pipeline does not depend on VCU silver.

---

## Full Folder Structure

```
ev-predictive-analytics/
│
├── run.py                              ← add cell_ingest, cell_gold, cell_report
│
├── configs/
│   ├── settings.py                     ← add new dir constants
│   ├── resample.yaml                   ← UNCHANGED
│   ├── trip.yaml                       ← UNCHANGED
│   ├── gold.yaml                       ← UNCHANGED
│   └── cell.yaml                       ← NEW
│
├── schema/
│   ├── telemetry_schema.yaml           ← UNCHANGED
│   ├── ranges.yaml                     ← UNCHANGED
│   ├── signal_classes.yaml             ← UNCHANGED
│   ├── quality_flags.yaml              ← UNCHANGED
│   ├── units.yaml                      ← UNCHANGED
│   ├── cell_voltage_schema.yaml        ← NEW
│   ├── cell_temp_schema.yaml           ← NEW
│   └── cell_quality_flags.yaml         ← NEW
│
├── ingestion/
│   ├── ingest.py                       ← UNCHANGED
│   ├── io.py                           ← UNCHANGED
│   ├── validators.py                   ← UNCHANGED
│   ├── resampler.py                    ← UNCHANGED
│   ├── faults.py                       ← UNCHANGED
│   ├── trip_segmentor.py               ← UNCHANGED
│   ├── cell_ingest.py                  ← NEW
│   └── cell_validators.py              ← NEW
│
├── features/
│   ├── rolling.py                      ← UNCHANGED
│   ├── lags.py                         ← UNCHANGED
│   ├── physics.py                      ← UNCHANGED
│   ├── trip_agg.py                     ← UNCHANGED
│   ├── pipeline.py                     ← UNCHANGED
│   ├── utils.py                        ← UNCHANGED
│   ├── cell_health.py                  ← NEW
│   ├── cell_agg.py                     ← NEW
│   └── cell_pipeline.py                ← NEW
│
├── reports/
│   └── cell_health_report.py           ← NEW
│
├── tasks/
│   ├── soc_forecast/                   ← UNCHANGED
│   └── cell_health/                    ← LATER (Phase 6, when labels exist)
│
├── data/
│   ├── raw/                            ← UNCHANGED
│   ├── raw_faults/                     ← UNCHANGED
│   ├── raw_cell_voltage/               ← READY
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.csv
│   ├── raw_cell_temp/                  ← READY
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.csv
│   ├── raw_parquet/                    ← UNCHANGED
│   ├── silver/                         ← UNCHANGED
│   ├── silver_cells/                   ← NEW
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.parquet
│   ├── gold/                           ← UNCHANGED
│   ├── gold_cells/                     ← NEW
│   │   ├── cell_features/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── vehicle_id=EV01.parquet
│   │   ├── module_features/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── vehicle_id=EV01.parquet
│   │   ├── pack_features/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── vehicle_id=EV01.parquet
│   │   └── daily_health_reports/
│   │       └── dt=YYYY-MM-DD/
│   │           └── vehicle_id=EV01.parquet
│   ├── state/                          ← UNCHANGED
│   ├── reports/                        ← UNCHANGED
│   └── drift_reports/                  ← UNCHANGED
```

---

## Data Layer Contracts

### `silver_cells` — what it contains

```
silver_cells/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
  ├── timestamp                        datetime64[ns] — joined key with VCU silver
  ├── vehicle_id                       string
  ├── M1_C1 ... M5_C36                 float64 — 180 cell voltage columns
  ├── M1_T1 ... M5_T18                 float64 — 90 cell temp columns
  └── cell_quality_flag                int64 — bitmask (see below)
```

### `gold_cells/cell_features` — one row per timestamp

```
  ├── timestamp, vehicle_id
  ├── M1_C1_delta ... M5_C36_delta     deviation from module mean (180 cols)
  ├── M1_C1_zscore ... M5_C36_zscore   z-score within module (180 cols)
  └── M1_C1_is_outlier ... M5_C36_is_outlier   binary flag (180 cols)
```

### `gold_cells/module_features` — one row per timestamp

```
  ├── timestamp, vehicle_id
  ├── module_1_voltage_mean            mean cell voltage in module
  ├── module_1_voltage_std             spread within module
  ├── module_1_voltage_range           max - min within module
  ├── module_1_lowest_cell             column name of lowest voltage cell
  ├── module_1_temp_mean
  ├── module_1_temp_range
  ├── module_1_hottest_sensor          column name of hottest temp sensor
  └── ... (same for modules 2-5)
```

### `gold_cells/pack_features` — one row per timestamp

```
  ├── timestamp, vehicle_id
  ├── pack_voltage_range               max - min across all 180 cells
  ├── pack_voltage_std                 std across all 180 cells
  ├── pack_temp_range                  max - min across all 90 sensors
  ├── worst_cell_id                    column name of lowest voltage cell
  ├── hottest_module_id                module number with highest avg temp
  └── imbalance_flag                   1 if pack_voltage_range > threshold
```

### `gold_cells/daily_health_reports` — one row per module per day

```
  ├── dt, vehicle_id, module_id
  ├── status                           Healthy / Monitor / Imbalance Rising /
  │                                    Thermal Hotspot / Degraded / Critical
  ├── avg_voltage_std                  average intra-module spread across day
  ├── max_voltage_range                worst spread seen during day
  ├── avg_temp_range                   average thermal spread across day
  ├── max_temp_spike                   worst temp delta from module mean
  ├── worst_cell_id                    cell with lowest voltage during day
  ├── pct_time_imbalanced              fraction of day above imbalance threshold
  └── trend_direction                  stable / worsening / improving
```

---

## Quality Flags — `cell_quality_flags.yaml`

| Flag | Value | Meaning |
|---|---|---|
| `CELL_FFILLED` | 1 | 1-8 nulls, forward filled (single cell group dropout) |
| `MODULE_GAP` | 2 | 9-169 nulls, partial module dropout, filled where possible |
| `PACK_GAP` | 4 | 170-180 nulls, complete pack reporting gap, no fill |
| `CELL_SOFT_BREACH` | 8 | Cell voltage or temp outside soft range |
| `CELL_HARD_BREACH` | 16 | Cell voltage or temp outside hard range |
| `CELL_FLATLINE` | 32 | Cell stuck at constant value across window |
| `CELL_OUTLIER` | 64 | Cell is statistical outlier from module peers |

---

## Null Handling Rules — from data analysis

```
Observed null patterns across 518,494 rows:
  4-8 nulls    →  1,451 rows  →  single cell group dropout
  9-169 nulls  →  3,121 rows  →  partial to full module dropout
  180 nulls    →  6,133 rows  →  complete pack reporting gap (57% of null rows)
```

| Null count | Classification | Action | Flag |
|---|---|---|---|
| 0 | Clean | No action | None |
| 1-8 | Cell group dropout | Forward fill | `CELL_FFILLED` |
| 9-169 | Partial module gap | Forward fill where possible | `MODULE_GAP` |
| 170-180 | Pack gap | No fill | `PACK_GAP` |

---

## Module Status Taxonomy

```
Healthy           → all signals nominal, no worsening trend
Monitor           → deviation present but stable, within thresholds
Imbalance Rising  → module voltage_std trending up across recent days
Thermal Hotspot   → one or more temp sensors consistently above module mean
Degraded Cell     → one or more cells persistently below module mean voltage
Critical          → multiple signals degraded, immediate attention required
```

Status is determined by `cell_health_report.py` from `daily_health_reports`.

---

## `configs/cell.yaml` — key thresholds

```yaml
null_handling:
  cell_group_dropout_max: 8
  pack_gap_threshold: 170

voltage:
  nominal_v: 3.2
  imbalance_warn_mv: 50
  imbalance_alert_mv: 100
  outlier_zscore: 2.5

temperature:
  hotspot_delta_c: 5
  warn_c: 45
  alert_c: 55

health_report:
  min_rows_for_report: 60
  trend_window_days: 7
```

---

## New Files — Purpose Summary

| File | Purpose |
|---|---|
| `schema/cell_voltage_schema.yaml` | 180 column definitions, dtypes, required flags |
| `schema/cell_temp_schema.yaml` | 90 column definitions, dtypes, required flags |
| `schema/cell_quality_flags.yaml` | Cell-specific bitmask definitions |
| `configs/cell.yaml` | All cell thresholds and health scoring rules |
| `ingestion/cell_ingest.py` | Pipeline entry point — loads, validates, writes silver_cells |
| `ingestion/cell_validators.py` | Null classification, range checks, flatline detection |
| `features/cell_health.py` | Per-cell deviation, z-score, outlier flag per timestamp |
| `features/cell_agg.py` | Module and pack aggregations per timestamp |
| `features/cell_pipeline.py` | silver_cells → gold_cells entry point |
| `reports/cell_health_report.py` | gold_cells → daily_health_reports + module status labels |

---

## Build Order

```
Step 1   schema/cell_quality_flags.yaml
Step 2   schema/cell_voltage_schema.yaml
Step 3   schema/cell_temp_schema.yaml
Step 4   configs/cell.yaml
Step 5   configs/settings.py  (add new dir constants)
Step 6   ingestion/cell_validators.py
Step 7   ingestion/cell_ingest.py
Step 8   features/cell_health.py
Step 9   features/cell_agg.py
Step 10  features/cell_pipeline.py
Step 11  reports/cell_health_report.py
Step 12  run.py  (wire cell_ingest, cell_gold, cell_report commands)
```

---

## What Does Not Change

```
raw/                    ← untouched
raw_faults/             ← untouched
silver/                 ← untouched
gold/                   ← untouched
ingestion/ingest.py     ← untouched
ingestion/validators.py ← untouched
ingestion/resampler.py  ← untouched
features/pipeline.py    ← untouched
tasks/soc_forecast/     ← untouched
All SOC model artifacts ← untouched
```

---

## Phase Roadmap

```
Phase 5a (now)     cell_ingest → silver_cells
Phase 5b (now)     cell_gold → gold_cells (cell + module + pack features)
Phase 5c (now)     cell_report → daily_health_reports + module status dashboard
Phase 6 (later)    tasks/cell_health/ — predictive model when replacement labels exist
```

**First business output:**
```
Battery Pack Health — 2026-03-10
  Module 1 → Healthy
  Module 2 → Healthy
  Module 3 → Imbalance Rising
  Module 4 → Thermal Hotspot
  Module 5 → Healthy
```