# Battery Analytics Engine — Start Here
> Vehicle: **EV01** | Last updated: 2026-05-18
> Active model: `v2__2026-03-13__fc0cd5ed` | Val MAE: **0.441% SOC** — EXCELLENT

---

## What This Project Is

A full-stack EV battery analytics and SOC prediction engine, part of a larger **AI-powered EV Digital Twin and Fleet Intelligence Platform**.

It ingests raw telemetry from the Vehicle Control Unit (VCU) and from battery cell-level sensors, processes it through a multi-phase data pipeline, engineers features, trains ML models to predict State of Charge (SOC) 5 minutes ahead, monitors production drift, and produces daily cell health reports per module.

---

## Current Phase Status

```
Phase 1 ✅   Raw CSV → Silver          ingestion, 1-Hz resampling, validation, fault flags, trip IDs
Phase 2 ✅   Silver → Gold             rolling, lag, physics features + trip/daily aggregations
Phase 2b ✅  Cell pipeline             180 cells + 90 temp sensors → cell/module/pack features + health reports
Phase 3 ✅   Gold → Model              XGBoost + LightGBM SOC forecaster, versioned artifacts, drift monitoring
Phase 4 ⏳   Next                      SHAP analysis, hyperparameter tuning, multi-vehicle expansion
```

---

## Folder Structure

```
ev-predictive-analytics/
│
├── run.py                        VCU pipeline runner (Phase 1 + 2)
├── run_cells.py                  Cell pipeline CLI  (Phase 2b)
├── build_master_dataset.py       Joins VCU silver + cell features → master dataset
├── Makefile                      All pipeline commands
├── requirements.txt
│
├── configs/
│   ├── settings.py               All data directory paths (import from here, never hardcode)
│   ├── gold.yaml                 Rolling windows (30/60/300/600s), lag offsets, target definition
│   ├── train_soc.yaml            LightGBM SOC model config
│   ├── cell.yaml                 Cell health thresholds (calibrated from real data)
│   ├── resample.yaml             1-Hz resampler config
│   └── trip.yaml                 Trip detection thresholds
│
├── schema/
│   ├── telemetry_schema.yaml     VCU column names and dtypes
│   ├── ranges.yaml               Signal soft/hard range bounds
│   ├── quality_flags.yaml        VCU quality bitmask definitions
│   ├── cell_voltage_schema.yaml  5 modules × 36 cells = 180 voltage columns
│   ├── cell_temp_schema.yaml     5 modules × 18 sensors = 90 temp columns
│   ├── cell_quality_flags.yaml   Cell quality bitmask (CELL_FFILLED, PACK_GAP, etc.)
│   ├── signal_classes.yaml       fast / slow / status → controls resample fill strategy
│   └── units.yaml                Physical units for all signals
│
├── ingestion/
│   ├── ingest.py                 VCU: raw CSV → silver (validate, resample, fault flags, trip IDs)
│   ├── validators.py             Schema + range checks, quality_flag bitmask
│   ├── faults.py                 Fault CSV → binary fault_* columns on silver rows
│   ├── resampler.py              1-Hz resampling by signal class
│   ├── trip_segmentor.py         Trip start/end detection, cross-day state file
│   ├── io.py                     All file I/O, atomic parquet writes
│   ├── cell_ingest.py            Cell: raw voltage + temp CSVs → silver_cells
│   └── cell_validators.py        Cell null/flatline/range validation
│
├── features/
│   ├── pipeline.py               VCU gold pipeline: rolling + physics + lag + trip agg
│   ├── rolling.py                Rolling mean/std/min/max over 30/60/300/600s windows
│   ├── lags.py                   Lag features at 60s, 300s, 600s
│   ├── physics.py                Physics-derived features (power, efficiency, spread, gradients)
│   ├── trip_agg.py               Per-trip aggregations
│   ├── utils.py                  Shared helpers
│   ├── cell_health.py            Per-cell delta, z-score, outlier flag for all 180 cells
│   ├── cell_agg.py               Module-level and pack-level voltage/temp aggregations
│   └── cell_pipeline.py          Cell gold pipeline entry point
│
├── training/
│   ├── train_soc_xgb.py          XGBoost SOC trainer (59 features, target = y_soc_t_plus_300s)
│   ├── dataset.py                Dataset builder
│   ├── artifacts.py              Model artifact saving + versioning
│   ├── splitter.py               Train/val split (shuffled trips or time-ordered)
│   ├── eval_per_trip.py          Per-trip evaluation
│   └── plot_phase3_debug.py      Debug training plots
│
├── scripts/
│   ├── run_day.py                Silver pipeline: one day
│   ├── backfill.py               Silver pipeline: date range
│   ├── run_gold_day.py           Gold pipeline: one day
│   ├── backfill_gold.py          Gold pipeline: date range
│   ├── train_soc.py              LightGBM SOC train script
│   ├── eval_soc.py               SOC evaluation
│   ├── model_report.py           Model report generation
│   └── split_csv_by_day.py       Splits large raw CSVs into dt= partitions
│
├── monitoring/
│   └── drift.py                  Drift detection: PSI, KS test, mean shift %, null rate change
│
├── evaluation/
│   └── metrics.py                MAE, RMSE, MAPE
│
├── reports/
│   └── cell_health_report.py     Daily module health labels (Healthy → Critical)
│
├── analysis/
│   ├── cell_health_baseline.py   180-cell degradation scan across all dates
│   └── ev01_cell_vcu_analysis_stage1_to_5.ipynb
│
├── dashboard/
│   ├── app.py
│   └── soc_charts.py
│
├── notebooks/                    All analysis/diagnostic notebooks
│   ├── 03_model_analysis.ipynb
│   ├── EV_Phase3_Dataset_Exploration.ipynb
│   ├── ev01_analysis_final.ipynb
│   ├── ev01_battery_diagnostic_report.ipynb
│   ├── ev01_founder_review_battery_diagnostic.ipynb
│   ├── fault_month_report.ipynb
│   ├── telemetry_merge_trip_graphs_high_rpm_only.ipynb
│   └── cell_threshold_tuning_notebook.ipynb
│
├── data/
│   ├── raw/                      Day-partitioned raw VCU CSVs (dt=YYYY-MM-DD/)
│   ├── raw_parquet/              Typed parquet of raw data
│   ├── raw_cell_voltage/         Day-partitioned cell voltage CSVs
│   ├── raw_cell_temp/            Day-partitioned cell temp CSVs
│   ├── raw_faults/               Fault event CSVs
│   ├── silver/                   Clean 1-Hz VCU data with trip_id + fault_* columns
│   ├── silver_cells/             Clean cell voltage + temp data with quality flags
│   ├── gold/
│   │   ├── window_features/      1-Hz feature rows per trip (rolling + lag + physics)
│   │   ├── trip_features/        One row per trip
│   │   └── daily_stats/          One row per vehicle per day
│   ├── gold_cells/
│   │   ├── cell_features/        Per-cell delta/zscore/outlier (180 features/row)
│   │   ├── module_features/      Per-module voltage + temp aggregations
│   │   ├── pack_features/        Pack-level aggregations + imbalance flag
│   │   └── daily_health_reports/ Per-module daily health status
│   ├── analysis/
│   │   └── master/               Master joined dataset (39 partitions, ~685k rows)
│   ├── drift_reports/            Production drift JSON reports
│   ├── intermediate/             Analysis artifact CSVs
│   ├── source_dump/              Original unprocessed raw CSV files (pre-split)
│   ├── reports/                  Pipeline manifests
│   └── state/                    open_trips.parquet (cross-day trip continuity)
│
├── models/
│   ├── soc_xgb_baseline.json     XGBoost baseline — val MAE 0.599%
│   ├── soc_xgb_v2.json           XGBoost v2 — val MAE 0.516%
│   └── soc_forecast/
│       ├── v1__2026-03-01__*/    LightGBM v1 run (model + eval + drift baseline)
│       └── v2__2026-03-13__*/    LightGBM v2 — val MAE 0.441% EXCELLENT ← current best
│
├── battery_analysis_outputs/     Cell/module voltage + temp summary CSVs, problem rankings
├── fault_report_outputs/         50+ PNG fault event charts
├── telemetry_output/             Merged trip CSV, trip report PDF, high-RPM plots
│
├── tests/                        pytest suite
├── ml/                           Placeholder module stubs (not yet implemented)
├── tasks/                        Task runner stubs
├── outputs/                      Miscellaneous output staging
└── docs/                         ← You are here
```

---

## What Has Been Built

### Phase 1 — Ingestion and Silver Pipeline
Raw VCU CSV files are standardised (30+ column renames), validated against schema with soft/hard range checks, resampled to 1 Hz by signal class (fast/slow/status), enriched with binary fault flag columns (`fault_*`, `fault_any`), and trip-segmented (`trip_id = EV01_000001` etc). Output is a clean Parquet silver layer partitioned by `dt=` and `vehicle_id=`.

**Known fault types tracked:** busbar_undervoltage, bus_overvoltage, hardware_overvoltage, total_hardware_failure, ac_hall_failure, module_over_temperature_warning, temperature_difference_failure, low_voltage_undervoltage, software_overcurrent.

### Phase 2 — Gold Feature Engineering
For each trip within each day's silver data, the pipeline computes:
- **Rolling features** — mean/std/min/max at 30s, 60s, 300s, 600s windows on 17 base signals
- **Lag features** — 60s, 300s, 600s lags on the same signals
- **Physics features** — electrical power proxy, power error, cell voltage spread, battery temp spread, mechanical power proxy, efficiency proxy, thermal gradients
- **SOC forecast target** — `y_soc_t_plus_300s` (SOC 5 minutes ahead)

Outputs: `gold/window_features/` (1-Hz feature rows), `gold/trip_features/` (per-trip aggregations), `gold/daily_stats/` (per-day summary).

### Phase 2b — Cell Pipeline
A completely separate pipeline for raw battery cell data:
- **180 cell voltages** (5 modules × 36 cells: M1_C1 … M5_C36)
- **90 temperature sensors** (5 modules × 18 sensors: M1_T1 … M5_T18)

Processes raw CSVs into silver_cells (with null handling, flatline detection, quality flags), then gold_cells:
- `cell_features/` — per-cell delta from module mean, z-score, outlier binary flag
- `module_features/` — per-module voltage mean/std/range/lowest_cell, temp mean/range/hottest_sensor
- `pack_features/` — pack voltage range/std, worst cell, pack temp range, hottest module, imbalance flag
- `daily_health_reports/` — per-module daily status: **Healthy / Monitor / Imbalance Rising / Thermal Hotspot / Degraded Cell / Critical**

All thresholds are calibrated from real data distributions and stored in `configs/cell.yaml`.

### Master Dataset
`build_master_dataset.py` joins VCU silver + cell pack features + cell module features on `timestamp + vehicle_id` into a single analysis-ready dataset.

| Property | Value |
|---|---|
| Location | `data/analysis/master/dt=*/vehicle_id=EV01.parquet` |
| Date range | 2026-01-22 → 2026-03-08 |
| Partitions | 39 |
| Total rows | ~685,000 |
| All partitions | 100% clean |

Key derived columns added: `cell_spread_v`, `abs_current_a`, `current_direction` (charge/idle/discharge), `soc_band` (0–20/20–40/.../80–100), `operating_regime` (heavy_regen → heavy_discharge), `is_clean`.

### Phase 3 — SOC Forecasting and Drift Monitoring

**Model:** XGBoost and LightGBM regressors predicting `y_soc_t_plus_300s` (SOC 5 minutes ahead).

**Feature set:** 59 features spanning SOC, current, voltage, cell voltages, power, motor speed, battery temp, motor temp, energy, and physics-derived signals.

**Results:**

| Model | Val MAE | Val RMSE | Notes |
|---|---|---|---|
| XGBoost baseline | 0.599% SOC | 0.820% | 1 train day |
| XGBoost v2 | 0.516% SOC | 0.697% | 4 train days |
| **LightGBM v2** | **0.441% SOC** | **0.563%** | **Current best — EXCELLENT** |

Persistence (naive carry-forward) baseline MAE = 0.822%. **LightGBM v2 beats it by 46%.**

Top features: `soc_pct` (50%), `soc_pct_roll_mean_60s` (34%), `soc_pct_lag_60s` (11%).

**Drift monitoring** compares each new day's feature distributions against the training baseline stored in `models/soc_forecast/v2__*/drift_baseline.parquet` using PSI, KS test, mean shift %, and null rate change. Reports for 8 production dates exist in `data/drift_reports/`.

---

## How to Run

### VCU Pipeline (Phase 1 + 2)
```bash
# Silver — one day
make run dt=2026-03-15

# Silver — backfill a range
make backfill start=2026-01-15 end=2026-03-17

# Gold — one day
make gold dt=2026-03-15

# Gold — backfill a range
make gold_backfill start=2026-01-15 end=2026-03-17
```

### Cell Pipeline (Phase 2b)
```bash
# Single day — all three stages in order
python run_cells.py cell_ingest --dt 2026-03-15
python run_cells.py cell_gold   --dt 2026-03-15
python run_cells.py cell_report --dt 2026-03-15

# Backfill all stages
python run_cells.py cell_ingest --backfill --start 2026-01-15 --end 2026-03-17
python run_cells.py cell_gold   --backfill --start 2026-01-15 --end 2026-03-17
python run_cells.py cell_report --backfill --start 2026-01-15 --end 2026-03-17
```

### Master Dataset
```bash
python build_master_dataset.py
```

### SOC Model
```bash
# Train (LightGBM)
make train dt=2026-03-13 vehicle_id=EV01

# Evaluate
make eval dt=2026-03-13 vehicle_id=EV01

# XGBoost trainer directly
python training/train_soc_xgb.py
```

### Tests
```bash
make test
```

---

## Current Model Artifacts

| File | Description |
|---|---|
| `models/soc_forecast/v2__2026-03-13__fc0cd5ed/model.json` | Production LightGBM model (1.4 MB) |
| `models/soc_forecast/v2__2026-03-13__fc0cd5ed/eval_report.json` | Full evaluation: per-trip, SOC buckets, error distribution, worst predictions |
| `models/soc_forecast/v2__2026-03-13__fc0cd5ed/drift_baseline.parquet` | Reference feature distributions for production drift checks |
| `models/soc_forecast/v2__2026-03-13__fc0cd5ed/feature_set.json` | Exact 59 features used |
| `models/soc_xgb_v2.json` | XGBoost v2 (1.3 MB) — second-best |

---

## Key Outputs Generated

| Output | Location |
|---|---|
| Master dataset | `data/analysis/master/` |
| Gold VCU features | `data/gold/window_features/`, `trip_features/`, `daily_stats/` |
| Cell gold features | `data/gold_cells/cell_features/`, `module_features/`, `pack_features/` |
| Daily module health | `data/gold_cells/daily_health_reports/` |
| Production drift reports | `data/drift_reports/dt=*.json` (8 dates) |
| Fault charts | `fault_report_outputs/` (50+ PNGs) |
| Battery analysis CSVs | `battery_analysis_outputs/` (6 files) |
| Trip report PDF | `telemetry_output/telemetry_trip_report.pdf` |

---

## Docs Index

| Document | What it covers |
|---|---|
| `Battery_Engine_start.md` | **← This file.** Start here. |
| `SOC_300_v2.md` | Phase-by-phase progress tracker with current model status |
| `status_till_silver.md` | Deep-dive on Phase 1 and Phase 2 pipeline design |
| `data_foundation.md` | Early data architecture design and rationale |
| `cell_health.md` | Cell health pipeline design and threshold calibration |
| `phase3_feature_selection_and_baseline.md` | Phase 3 feature selection process and baseline comparisons |
| `cell/master_dataset.md` | Master dataset column reference |
| `model_cards/soc_forecast_v1.md` | Model card for LightGBM v1 run |
| `ACHIEVEMENT_SUMMARY.md` | Full technical handoff — every file, every output, every metric |
| `CHATGPT_ACHIEVEMENT_SUMMARY.md` | Condensed paste-ready context for external AI tools |
