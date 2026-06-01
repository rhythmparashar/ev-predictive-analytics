# Project Structure — ev-predictive-analytics

Fleet:
- **EV01 — Loader** | Data range: 2026-01-15 → 2026-05-17 | Battery: 5 modules × 36 cells = 180 cells
- **EV02 — Excavator** | Data range: 2026-01-15 → 2026-05-14 | Battery: 5 modules × 36 cells = 180 cells

SOC models are **per machine type** — Loader and Excavator have different duty cycles and require separate models.

---

## Data Pipeline (how data flows)

```
data/raw_parquet/          ← raw CAN/telemetry parquet (one folder per day)
data/raw_cell_voltage/     ← raw per-cell voltage parquet
data/raw_cell_temp/        ← raw per-cell temperature parquet
data/raw_faults/           ← raw motor-controller fault events
        │
        ▼  ingestion/
data/silver/               ← cleaned, resampled 1Hz telemetry
data/silver_cells/         ← cleaned cell voltage + temp merged
        │
        ▼  features/
data/gold/window_features/ ← 61-feature ML windows (SOC model input)
data/gold/trip_features/   ← trip-level aggregates
data/gold/daily_stats/     ← per-day summary stats
data/gold_cells/
  cell_features/           ← per-cell gap (Mn_Cm_delta), outlier flag, z-score
  module_features/         ← per-module temp mean/range, voltage mean/range
  daily_health_reports/    ← daily Healthy/Monitor/Degraded/Critical per module
  pack_features/           ← pack-level voltage spread, SOC
        │
        ▼  build_master_dataset.py
data/analysis/master/      ← merged master dataset (telemetry + cell + module)
```

---

## Source Code

### `ingestion/`
Reads raw parquet → produces silver layer.
- `ingest.py` — main telemetry ingestion (resampling, dedup, trip segmentation)
- `resampler.py` — 1 Hz resampling logic
- `trip_segmentor.py` — splits continuous data into trips
- `validators.py` — schema validation for telemetry
- `cell_ingest.py` — cell voltage + temp ingestion
- `cell_validators.py` — schema validation for cell data
- `faults.py` — fault event ingestion (motor controller faults only)
- `io.py` — shared I/O helpers

### `features/`
Silver → Gold feature engineering.
- `pipeline.py` — main feature pipeline orchestrator
- `lags.py` — lag features (60s, 300s, 600s)
- `rolling.py` — rolling mean/std/max/min windows
- `physics.py` — physics-derived features (power proxy, energy consumed)
- `trip_agg.py` — trip-level aggregates
- `cell_pipeline.py` — cell feature pipeline (gap, z-score, outlier)
- `cell_agg.py` — module-level aggregation from cell features
- `cell_health.py` — daily health label logic (Healthy/Monitor/etc.)
- `utils.py` — shared feature utilities

### `battery_intelligence/`
Higher-level battery intelligence layer (built on top of gold data).
- `loader.py` — **DuckDB-based data loaders** (use these instead of raw pd.read_parquet)
- `health_score.py` — battery health score 0–100
- `rule_engine.py` — rule-based diagnostics (cold+high-load alert, critical cell, etc.)
- `cell_intelligence.py` — per-cell intelligence (worst cell, gap trends)
- `recommendations.py` — maintenance recommendations from rule outputs
- `summary.py` — daily summary aggregation
- `completeness_check.py` — data completeness validation
- `config.py`, `io.py` — module config and I/O

### `ml/`
ML utilities shared across models.
- `train.py` — generic XGBoost training wrapper
- `evaluate.py` — evaluation helpers
- `dataset.py` — dataset construction
- `drift.py` — feature drift detection
- `metrics.py` — MAE/RMSE/MAPE
- `registry.py` — model registry
- `score_day.py` — score a single day with a loaded model
- `io_utils.py` — model I/O helpers
- `feature_sets/base_soc_v1.yaml` — feature list for v1 SOC model

### `configs/`
Project-wide configuration.
- `settings.py` — `DATA_DIR`, `PROJECT_ROOT`, paths (import this everywhere)
- `gold.yaml` — gold layer config
- `cell.yaml` — cell pipeline config
- `resample.yaml` — resampling config
- `trip.yaml` — trip segmentation config
- `train_soc.yaml` — SOC training config (older)

### `schema/`
YAML schemas for data validation.
- `telemetry_schema.yaml` — telemetry column types + ranges
- `cell_voltage_schema.yaml` — cell voltage schema
- `cell_temp_schema.yaml` — cell temp schema
- `quality_flags.yaml` — data quality flag definitions
- `cell_quality_flags.yaml` — cell-specific quality flags
- `ranges.yaml` — physical range limits
- `signal_classes.yaml` — signal classification (battery/motor/system)
- `units.yaml` — units for all signals

### `scripts/`
Runnable one-off and pipeline scripts. **Run these directly.**
- `run_day.py` — process one raw day through silver
- `run_gold_day.py` — run gold feature engineering for one day
- `backfill.py` — backfill silver across all dates
- `backfill_gold.py` — backfill gold across all dates
- `train_soc.py` — older SOC training script (v1/v2 era)
- `train_soc_v3.py` — **current** SOC training script (v3, XGBoost, 61 features)
- `eval_soc.py` — evaluate SOC model (basic)
- `eval_soc_post_april.py` — evaluate v2 model pre vs post firmware update
- `split_csv_by_day.py` — split source CSV dumps into daily parquet
- `generate_cell_degradation_pdf.py` — generate cell degradation PDF report
- `generate_telemetry_correlation_pdf.py` — generate 11-signal correlation PDF
- `run_battery_engine_module1.py` — run Battery Intelligence Module 1 (health score + diagnostics)
- `run_battery_engine_module2.py` — run Battery Intelligence Module 2 (cell completeness)
- `run_battery_engine_module2_1.py` — run Module 2.1 (sensor completeness detail)
- `model_report.py` — generate model analysis report

### `tasks/soc_forecast/`
SOC forecast task definition (older config-driven approach).
- `train.py`, `evaluate.py`, `feature_set.py`, `config.yaml`
- Note: superseded by `scripts/train_soc_v3.py` for active work.

### `monitoring/`
- `drift.py` — production drift monitoring (compare live feature distributions vs training baseline)

### `evaluation/`
- `metrics.py` — standalone evaluation metrics module

---

## Models

### `models/soc_forecast/`
Versioned SOC forecast models. Each version folder contains:
`model.json`, `feature_set.json`, `eval_report.json`, `drift_baseline.parquet`, `config.yaml`

| Version | Date | Val MAE | Notes |
|---|---|---|---|
| `v1__2026-03-01__e50f8b21` | Mar 2026 | — | First baseline |
| `v2__2026-03-13__fc0cd5ed` | Mar 2026 | 0.441% (train-time val) | Pre-April data only |
| **`v3__2026-05-20__7b14ac62`** | May 2026 | **0.511%** | **Current production model** — balanced pre+post firmware data, 3× better than v2 in deployment |

### `models/phase3_debug/`
Debug artifacts from Phase 3 training exploration. Not a production model.

### Root-level model files (`soc_xgb_baseline.json`, `soc_xgb_v2.json`, etc.)
Older flat model files from before versioned model directories were introduced. Superseded by `models/soc_forecast/`.

---

## Outputs

### `outputs/battery_engine/`
Outputs from Battery Intelligence Engine runs.
- `battery_health_scores.parquet` — daily health scores per module
- `battery_alerts.parquet` — triggered alert events
- `battery_daily_summary.csv` — per-day summary table
- `battery_module_summary.csv` — per-module summary
- `cell_delta_profile.csv` — cell voltage gap profiles
- `weak_cell_profile.csv` — chronically weak cells
- `module_chronic_risk.csv` — modules with chronic risk patterns
- `hot_sensor_profile.csv` — temperature hotspot sensors
- `battery_twin_state_sample.json` — digital twin state snapshot
- `battery_engine_module1_summary.json`, `module2_summary.json`, `module2_1_summary.json`

### `outputs/soc_eval_post_april.json`
Evaluation results comparing v2 model on pre-April vs post-April data.

### `outputs/ev01_analysis/`
Older analysis outputs from early EV01 exploration.

---

## Reports

### `reports/`
Generated PDF and markdown reports.
- `cell_degradation_report_EV01.pdf` — cell-level degradation analysis PDF
- `telemetry_correlation_report_EV01.pdf` — 11-signal correlation analysis PDF
- `battery_engine_module1_report.md` — Battery Intelligence Module 1 findings
- `battery_engine_module2_report.md` — Module 2 sensor completeness report
- `battery_engine_module2_1_completeness_report.md` — Module 2.1 detailed completeness
- `soc_eval_post_april.md` — SOC model pre/post firmware evaluation report
- `cell_health_report.py` — script that generates cell health report (should be in scripts/)

---

## Notebooks

### `notebooks/`
Exploratory and analysis notebooks.
- `EV_Phase3_Dataset_Exploration.ipynb` — Phase 3 dataset exploration
- `battery_baseline_analysis.ipynb` — battery baseline analysis
- `cell_degradation_trend.ipynb` — M1_C25 and cell degradation trends
- `cell_threshold_tuning_notebook.ipynb` — threshold calibration for health labels
- `ev01_analysis_final.ipynb` — final EV01 analysis
- `ev01_battery_diagnostic_report.ipynb` — battery diagnostic report notebook
- `ev01_founder_review_battery_diagnostic.ipynb` — founder review version
- `fault_month_report.ipynb` — monthly fault analysis
- `telemetry_battery_correlation.ipynb` — telemetry ↔ battery signal correlations
- `telemetry_merge_trip_graphs_high_rpm_only.ipynb` — high-RPM trip analysis
- `03_model_analysis.ipynb` — SOC model analysis

---

## Docs

### `docs/`
Architecture and analytical documentation.
- `ev-dashboard-CLAUDE.md` — **context file for the new 3D dashboard repo** (copy as CLAUDE.md when starting ev-dashboard)
- `Battery_Engine_start.md` — Battery Intelligence Engine design doc
- `cell_health.md` — cell health scoring methodology
- `cell/master_dataset.md` — master dataset schema
- `SOC_300_v2.md` — SOC model v2 design notes
- `Soc_300_v1.md` — SOC model v1 design notes
- `data_foundation.md` — data layer architecture
- `status_till_silver.md` — pipeline status as of silver layer
- `phase3_feature_selection_and_baseline.md` — Phase 3 feature selection notes
- `model_cards/soc_forecast_v1.md` — model card for v1
- `ACHIEVEMENT_SUMMARY.md` — project milestones summary
- `remember.txt` — personal notes (not code)

---

## Stale / Can Be Archived

These are from earlier phases and are no longer the active path:

| Folder/File | Status | Reason |
|---|---|---|
| `analysis/` | Stale | Early exploration notebooks/scripts, superseded by `notebooks/` and `battery_intelligence/` |
| `battery_analysis_outputs/` | Stale | CSV summaries from early analysis, superseded by `outputs/battery_engine/` |
| `fault_report_outputs/` | Stale | PNG fault plots from early fault exploration |
| `telemetry_output/` | Stale | Early telemetry merge outputs (CSV + PDF) |
| `dashboard/` | Stale | Old Plotly/Dash prototype, superseded by the new 3D React dashboard |
| `training/` | Stale | Phase 3 training exploration code, superseded by `scripts/train_soc_v3.py` |
| `models/soc_xgb_baseline.json` etc. | Stale | Flat model files before versioning, superseded by `models/soc_forecast/v3` |
| `data/raw/` | Duplicate? | Check if identical to `data/raw_parquet/` — may be safe to remove one |
| `data/intermediate/` | Stale | Intermediate CSV files from early data QC |
| `data/min_max_per_timestamp*.csv` | Stale | Early QC outputs at wrong location |
| `tasks/soc_forecast/` | Superseded | Config-driven approach replaced by direct scripts |
| `docs/remember.txt` | Personal | Not part of codebase |

---

## Key Entry Points

| Task | Command |
|---|---|
| Process one new raw day | `python scripts/run_day.py --date YYYY-MM-DD` |
| Build gold features for one day | `python scripts/run_gold_day.py --date YYYY-MM-DD` |
| Run battery intelligence engine | `python scripts/run_battery_engine_module1.py` |
| Train SOC model | `venv/bin/python3 scripts/train_soc_v3.py` |
| Evaluate SOC vs firmware eras | `venv/bin/python3 scripts/eval_soc_post_april.py` |
| Generate cell degradation PDF | `venv/bin/python3 scripts/generate_cell_degradation_pdf.py` |
| Generate correlation PDF | `venv/bin/python3 scripts/generate_telemetry_correlation_pdf.py` |

---

## Important Facts

- **`fault_any`** in telemetry = motor controller fault flag only (not battery faults)
- **Firmware update**: March 27–30, 2026. M1_C25 gap dropped from −43 mV → −4 mV overnight.
- **SOC model v3**: trained on 14 days balanced across pre/post firmware. Val MAE 0.511% on May holdout.
- **DuckDB loaders**: always use `battery_intelligence/loader.py` functions to read parquet — avoids OOM and handles duplicate column names that break pyarrow.
- **venv**: project has its own venv at `venv/`. Use `venv/bin/python3` for scripts.
