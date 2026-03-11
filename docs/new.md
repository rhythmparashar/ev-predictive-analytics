# EV Telemetry ML — Phase Progress Tracker
> **Last updated:** 2026-03-05  
> **Active run:** `v1__2026-03-01__e50f8b21`  
> **Val MAE:** 0.553% SOC

---

## Phase Overview

```
Phase 1 ✅   raw → silver          (ingestion, validation, fault alignment, trip labeling)
Phase 2 ✅   silver → gold         (window features, trip features, daily stats)
Phase 3 ✅   gold → model          (XGBoost SOC forecast, versioned artifacts, drift monitoring)
Phase 4 ⏳   model improvement     (SHAP, hyperparameter tuning, multi-vehicle)
```

---

## Phase 1 — Data Foundation
> **Status: ✅ Complete and Stable**

### What it does
Raw CSV → Silver parquet. One command per date. Deterministic and idempotent.

### Pipeline
```
Raw CSV
  → Column standardisation
  → Timestamp parsing (HH:MM:SS → UTC datetime)
  → Validation + quality_flag bitmask
  → Write raw_parquet
  → Resample to 1 Hz (by signal class)
  → Fault alignment (binary fault_* columns)
  → Trip segmentation (trip_id)
  → Write silver
  → Write manifest JSON
```

### Key files
| File | Purpose |
|---|---|
| `ingestion/ingest.py` | Pipeline entry point per vehicle per day |
| `ingestion/validators.py` | quality_flag bitmask — 7 flag types |
| `ingestion/resampler.py` | 1 Hz resampling by signal class (fast/slow/status) |
| `ingestion/faults.py` | Aligns fault log windows → binary fault_* columns in silver |
| `ingestion/trip_segmentor.py` | Trip start/end detection, cross-day state file |
| `ingestion/io.py` | All file I/O, atomic writes (tmp → rename) |
| `scripts/run_day.py` | CLI entry point: `python run.py ingest --dt YYYY-MM-DD` |

### Outputs
```
data/raw_parquet/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
data/silver/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
  ├── telemetry signals
  ├── quality_flag       (bitmask)
  ├── trip_id            (EV01_000001, EV01_000002, ...)
  ├── fault_*            (one binary column per fault code)
  └── fault_any          (1 if any fault active)
data/reports/dt=YYYY-MM-DD.json
data/state/open_trips.parquet   (cross-day trip continuity)
```

### quality_flag bitmask
| Bit | Value | Meaning |
|---|---|---|
| 0 | 1 | Interpolated (fast signal) |
| 1 | 2 | Forward-filled (slow/status signal) |
| 2 | 4 | Gap inserted (missing source row) |
| 3 | 8 | Soft range breach |
| 4 | 16 | Hard range breach |
| 5 | 32 | Time anomaly |
| 6 | 64 | Sensor flatline |

### Properties
| Property | Status |
|---|---|
| Stable | ✅ |
| Deterministic | ✅ |
| Idempotent (rerun = same output) | ✅ |
| Cross-day trips | ✅ |
| New vehicle = zero code changes | ✅ |
| Fault alignment in silver | ✅ |

---

## Phase 2 — Gold Feature Engineering
> **Status: ✅ Complete and Stable**

### What it does
Silver → Gold. Computes ML-ready features per trip, writes partitioned parquet.

### Pipeline
```
Silver parquet
  → Quality filter: (quality_flag & 52) == 0
  → Trip-level streaming (one group at a time)
  → Rolling features (30s, 60s, 300s, 600s windows)
  → Physics features (voltage imbalance, thermal gradient, efficiency proxy)
  → Lag features (60s, 300s, 600s)
  → SOC target label: y_soc_t_plus_300s = soc_pct.shift(-300)
  → label_available flag
  → Write window_features (partitioned by trip, one part-XXXXXX.parquet per trip)
  → Write trip_features (one row per trip)
  → Write daily_stats
  → Write manifest JSON
```

### Key files
| File | Purpose |
|---|---|
| `features/pipeline.py` | Main Gold builder per vehicle per day |
| `features/rolling.py` | Rolling mean/std/min/max |
| `features/physics.py` | Derived physics variables |
| `features/lags.py` | Lag features |
| `features/trip_agg.py` | Per-trip aggregations |
| `scripts/run_gold_day.py` | CLI entry point: `python run.py gold --dt YYYY-MM-DD` |

### Outputs
```
data/gold/window_features/dt=YYYY-MM-DD/vehicle_id=EV01/
  part-000001.parquet ... part-XXXXXX.parquet   (one per trip)
  ├── telemetry signals + rolling + lag + physics features
  ├── quality_flag, trip_id, fault_any
  ├── y_soc_t_plus_300s   (target)
  └── label_available
data/gold/trip_features/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
data/gold/daily_stats/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
data/reports/gold/dt=YYYY-MM-DD/vehicle_id=EV01.json
```

### Properties
| Property | Status |
|---|---|
| Silver never modified | ✅ |
| Streaming per trip (low RAM) | ✅ |
| Atomic directory writes | ✅ |
| Column projection (schema-safe) | ✅ |
| Target label + label_available flag | ✅ |

---

## Phase 3 — Production ML Pipeline
> **Status: ✅ Complete and Verified**  
> **Active model:** `v1__2026-03-01__e50f8b21`  
> **Val MAE:** 0.553% SOC (Acceptable — threshold for Excellent is < 0.5%)

### What it does
Gold → XGBoost model. Full train/eval/versioning/drift pipeline. Two operating modes:
- `train` — full retrain, new versioned run folder
- `eval` — inference only on a new day, no retraining

### New folder structure introduced in Phase 3
```
ev-telemetry-ml/
│
├── tasks/
│   └── soc_forecast/
│       ├── feature_set.py     ← canonical FEATURES list (single source of truth)
│       ├── config.yaml        ← all hyperparameters + train/val dates
│       ├── train.py           ← training entry point
│       └── evaluate.py        ← per-trip + SOC bucket eval
│
├── training/
│   ├── dataset.py             ← Gold loader, quality filter, derived features
│   ├── splitter.py            ← trip-level chronological split
│   └── artifacts.py           ← versioned run folder save/load
│
├── evaluation/
│   └── metrics.py             ← MAE, RMSE, MAPE, per-group, SOC buckets, baselines
│
├── monitoring/
│   └── drift.py               ← PSI + KS test per feature vs training baseline
│
├── models/
│   └── soc_forecast/
│       └── v1__2026-03-01__e50f8b21/
│           ├── model.json
│           ├── feature_set.json
│           ├── data_fingerprint.json
│           ├── eval_report.json
│           ├── drift_baseline.parquet
│           └── config.yaml
│
├── data/
│   └── drift_reports/
│       ├── dt=YYYY-MM-DD.json          ← drift-only report
│       └── dt=YYYY-MM-DD_eval.json     ← eval + drift combined
│
└── run.py                     ← single CLI entry point
```

### Commands
```bash
# Full pipeline for a new date
python run.py ingest --dt 2026-03-05
python run.py gold   --dt 2026-03-05

# Retrain (edit tasks/soc_forecast/config.yaml dates first)
python run.py train  --task soc_forecast

# Inference-only eval on new day (no retraining)
python run.py eval   --task soc_forecast --dt 2026-03-05

# Pin specific run
python run.py eval   --task soc_forecast --dt 2026-03-05 --run-id v1__2026-03-01__e50f8b21

# Drift check only
python run.py drift  --task soc_forecast --dt 2026-03-05

# Backfill
python run.py ingest --backfill --start 2026-01-01 --end 2026-03-04
```

### Model — current run `v1__2026-03-01__e50f8b21`
| Setting | Value |
|---|---|
| Algorithm | XGBoost Regressor |
| Target | `y_soc_t_plus_300s` (SOC 5 min ahead) |
| Features | 61 |
| Train dates | 2026-02-23, 24, 25, 26, 2026-03-01 |
| Val date | 2026-03-04 |
| Train rows | 184,144 across 172 trips |
| Val rows | 7,072 across 18 trips |
| Best iteration | 163 of 800 |

### Results
| Split | MAE | RMSE | MAPE |
|---|---|---|---|
| Train | 0.276% | 0.374% | 0.40% |
| Val | 0.553% | 0.705% | 0.78% |
| Train-val gap | 0.276% | — | — |

| Baseline | Val MAE |
|---|---|
| Persistence (current SOC) | 0.981% |
| Rolling mean 60s | 1.092% |
| **XGBoost model** | **0.553%** |

### Accuracy thresholds
| Metric | Excellent | Acceptable | Investigate |
|---|---|---|---|
| Val MAE | < 0.5% | 0.5–1.5% | > 2% |
| Train-val gap | < 0.3% | < 0.5% | > 1% |

**Current status: ACCEPTABLE** (MAE 0.553%, gap 0.276%)

### Top features
| Rank | Feature | Importance |
|---|---|---|
| 1 | soc_pct | 0.495 |
| 2 | soc_pct_roll_mean_60s | 0.327 |
| 3 | soc_pct_lag_60s | 0.140 |
| 4 | stack_voltage_v | 0.007 |
| 5 | motor_speed_rpm_roll_std_60s | 0.003 |

`soc_pct` + `soc_pct_roll_mean_60s` + `soc_pct_lag_60s` = 96.2% of total importance.

### Versioned run folder — what's saved
| File | Contents |
|---|---|
| `model.json` | XGBoost model (native format) |
| `feature_set.json` | Exact 61 features used at train time |
| `data_fingerprint.json` | SHA-256 hash of every parquet file used |
| `eval_report.json` | Full MAE/RMSE/MAPE + per-trip + SOC buckets |
| `drift_baseline.parquet` | Feature distributions from training data |
| `config.yaml` | Config snapshot at train time |

### Drift monitoring
Runs on every `eval` call. Compares new day's feature distributions against `drift_baseline.parquet`.

| Metric | Flag threshold |
|---|---|
| PSI | > 0.1 MONITOR, > 0.2 DRIFT |
| KS test p-value | < 0.05 DRIFT |
| Mean shift % | > 10% DRIFT |
| Null rate change | > 5% DRIFT |

**Known permanent drifters to remove before next retrain:**
- `total_kwh_consumed` + lags — cumulative odometer, PSI=11 by design
- `motor_temperature_c` + lags/rolling — all zeros on EV01, sensor not fitted

### Phase 3 properties
| Property | Status |
|---|---|
| Canonical feature list (single source of truth) | ✅ `tasks/soc_forecast/feature_set.py` |
| Config-driven hyperparameters | ✅ `tasks/soc_forecast/config.yaml` |
| Versioned run folders | ✅ `models/soc_forecast/v{n}__{date}__{hash}/` |
| Data fingerprinting (SHA-256) | ✅ `data_fingerprint.json` |
| Trip-level chronological split | ✅ no data leakage |
| Drift monitoring (PSI + KS) | ✅ daily after `eval` |
| Inference-only eval (no retrain) | ✅ `python run.py eval` |
| Baseline comparison | ✅ persistence + rolling mean |
| Per-trip MAE breakdown | ✅ |
| SOC bucket breakdown | ✅ |
| Error distribution (p5/p50/p95) | ✅ |
| Single CLI entry point | ✅ `run.py` |
| Scalable to new tasks | ✅ add `tasks/<new_task>/` folder |
| SHAP explainability | ⏳ Phase 4 |
| Hyperparameter tuning (Optuna) | ⏳ Phase 4 |
| Multi-vehicle generalisation | ⏳ Phase 4 |
| Retrain trigger (rolling MAE) | ⏳ Phase 4 |

---

## Phase 4 — Model Improvement + Explainability
> **Status: ⏳ Not started**

### Planned work

**Feature cleanup (do before next retrain)**
- Drop `total_kwh_consumed`, `total_kwh_consumed_lag_60s`, `total_kwh_consumed_lag_300s` — cumulative counter, drifts permanently
- Drop `motor_temperature_c`, `motor_temperature_c_lag_300s`, `motor_temperature_c_roll_mean_300s`, `motor_temperature_c_roll_std_60s` — all zeros on EV01
- Result: 61 → 54 features, cleaner drift reports

**SHAP explainability**
- Per-prediction feature attribution
- SHAP summary plot — global feature drivers
- SHAP waterfall plots — explain individual trips to clients
- Add `shap>=0.44` to `requirements.txt`

**Hyperparameter tuning**
- Optuna Bayesian search over `max_depth`, `learning_rate`, `subsample`, `min_child_weight`
- Walk-forward cross-validation (train month N, val month N+1)
- Target: push val MAE below 0.5% (Excellent)
- Add `optuna>=3.4` to `requirements.txt`

**Fault-aware evaluation**
- Separate MAE for `fault_any == 0` vs `fault_any == 1`
- Quantify whether active faults degrade SOC prediction accuracy
- If MAE degrades significantly: train fault-conditioned model variant

**Multi-vehicle generalisation**
- When EV02 data available: train on EV01, test on EV02
- Measure cross-vehicle MAE degradation
- Determine: one shared model vs per-vehicle models

**Retrain trigger**
- Rolling 7-day MAE tracked in `data/drift_reports/`
- If rolling MAE exceeds 1.0%: flag for retrain
- Implement in `monitoring/health.py`

---

## Known Issues / Watch List

| Issue | Severity | Action |
|---|---|---|
| `total_kwh_consumed` drifts permanently (PSI=11) | Medium | Drop from feature_set.py before next retrain |
| `motor_temperature_c` all zeros on EV01 | Low | Drop from feature_set.py before next retrain |
| 2026-02-23 has 538k rows vs ~60k for other dates | Monitor | Investigate if anomaly or normal busy day |
| Charging detection (EV01_000744) — model misses SOC gain at `battery_current_a=0` | Low | More charging examples needed in training data |
| Drift report flags 56/61 features on 2026-03-04 | Expected | Most is regime shift (temp/load), not model failure — 2026-03-04 was the val date |

---

## Quick Reference — Daily Workflow

```bash
# New day arrives
python run.py ingest --dt YYYY-MM-DD    # raw → silver
python run.py gold   --dt YYYY-MM-DD    # silver → gold

# Score with current model + drift check
python run.py eval   --task soc_forecast --dt YYYY-MM-DD

# Retrain (when needed — update config.yaml dates first)
python run.py train  --task soc_forecast
```

---

## Adding a New Task (e.g. fault detection)

```bash
mkdir tasks/fault_detection
# Create: feature_set.py, config.yaml, train.py, evaluate.py
# Wire into run.py cmd_train / cmd_eval (2 lines each)
python run.py train --task fault_detection
```

Nothing in `ingestion/`, `features/`, `training/`, `evaluation/`, or `monitoring/` changes.



ev-predictive-analytics/
│
├── run.py
│
├── configs/
│   ├── settings.py
│   ├── resample.yaml
│   ├── trip.yaml
│   └── gold.yaml
│
├── schema/
│   ├── telemetry_schema.yaml
│   ├── ranges.yaml
│   ├── signal_classes.yaml
│   ├── quality_flags.yaml
│   └── units.yaml
│
├── ingestion/
│   ├── ingest.py
│   ├── io.py
│   ├── validators.py
│   ├── resampler.py
│   ├── faults.py
│   ├── trip_segmentor.py
│   └── tests/
│       ├── test_validators.py
│       ├── test_resampler.py
│       ├── test_trip_segmentor.py
│       ├── test_faults.py
│       └── test_io.py
│
├── features/
│   ├── rolling.py
│   ├── lags.py
│   ├── physics.py
│   ├── trip_agg.py
│   ├── pipeline.py
│   ├── utils.py
│   └── tests/
│       ├── test_rolling.py
│       ├── test_lags.py
│       ├── test_physics.py
│       ├── test_trip_agg.py
│       └── test_pipeline.py
│
├── tasks/
│   └── soc_forecast/
│       ├── config.yaml
│       ├── feature_set.py
│       ├── train.py
│       └── evaluate.py
│
├── training/
│   ├── dataset.py
│   ├── splitter.py
│   └── artifacts.py
│
├── evaluation/
│   └── metrics.py
│
├── monitoring/
│   └── drift.py
│
├── scripts/
│   ├── run_day.py
│   └── run_gold_day.py
│
├── data/
│   ├── raw/
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.csv
│   │
│   ├── raw_faults/
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.csv
│   │
│   ├── raw_parquet/
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.parquet
│   │
│   ├── silver/
│   │   └── dt=YYYY-MM-DD/
│   │       └── vehicle_id=EV01.parquet
│   │
│   ├── gold/
│   │   ├── window_features/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── vehicle_id=EV01/
│   │   │           ├── part-000001.parquet
│   │   │           └── ...
│   │   │
│   │   ├── trip_features/
│   │   │   └── dt=YYYY-MM-DD/
│   │   │       └── vehicle_id=EV01.parquet
│   │   │
│   │   └── daily_stats/
│   │       └── dt=YYYY-MM-DD/
│   │           └── vehicle_id=EV01.parquet
│   │
│   ├── reports/
│   │   ├── dt=YYYY-MM-DD.json
│   │   └── gold/
│   │       └── dt=YYYY-MM-DD/
│   │           └── vehicle_id=EV01.json
│   │
│   ├── drift_reports/
│   │   ├── dt=YYYY-MM-DD.json
│   │   └── dt=YYYY-MM-DD_eval.json
│   │
│   ├── state/
│   │   └── open_trips.parquet
│   │
│   └── samples/
│       └── ...
│
├── models/
│   └── soc_forecast/
│       └── v1__2026-03-01__e50f8b21/
│           ├── model.json
│           ├── feature_set.json
│           ├── data_fingerprint.json
│           ├── eval_report.json
│           ├── drift_baseline.parquet
│           └── config.yaml
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_check.ipynb
│
├── tests/
│   └── test_end_to_end.py
│
├── requirements.txt
├── README.md
├── Makefile
└── .env.example