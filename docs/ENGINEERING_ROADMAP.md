# Engineering Roadmap — EV Predictive Analytics

**Purpose:** Turn this ML pipeline into a full-fledged production system covering software engineering, MLOps, system design, ML/AI, and algorithms. Each section includes what to build, why it matters for interviews, and specific implementation guidance for this codebase.

---

## Table of Contents

1. [Software Engineering Foundation](#1-software-engineering-foundation)
2. [ML & Algorithm Improvements](#2-ml--algorithm-improvements)
3. [MLOps Infrastructure](#3-mlops-infrastructure)
4. [System Design & API Layer](#4-system-design--api-layer)
5. [Production Monitoring & Alerting](#5-production-monitoring--alerting)
6. [Data Engineering Upgrades](#6-data-engineering-upgrades)
7. [Interview Talking Points by Role](#7-interview-talking-points-by-role)
8. [Execution Priority](#8-execution-priority)

---

## 1. Software Engineering Foundation

### 1.1 Testing

**Current state:** `tests/` directory is empty.

**What to build:**

```
tests/
├── unit/
│   ├── test_ingestion.py          # column standardization, timestamp parsing
│   ├── test_features.py           # lag/rolling/physics feature correctness
│   ├── test_health_score.py       # penalty math, score clipping 0–100
│   ├── test_drift.py              # PSI computation, KS thresholds
│   └── test_validators.py         # schema validation, range checks
├── integration/
│   ├── test_silver_pipeline.py    # raw → silver end-to-end with fixture data
│   └── test_gold_pipeline.py      # silver → gold end-to-end
└── conftest.py                    # shared fixtures (sample dataframes, temp dirs)
```

**Specific tests to write:**

```python
# test_health_score.py — covers the penalty math exactly
def test_thermal_penalty_thresholds():
    df = pd.DataFrame({"max_battery_temp_c": [20, 38, 42, 48]})
    out = compute_health_scores(df)
    assert out["thermal_penalty"].tolist() == [0, 5, 10, 20]

def test_score_clamped_to_zero():
    # stack all penalties — score must never go below 0
    df = make_worst_case_df()
    out = compute_health_scores(df)
    assert (out["battery_health_score"] >= 0).all()

# test_drift.py — PSI is a formula, test it exactly
def test_psi_identical_distributions_is_zero():
    arr = np.random.normal(0, 1, 1000)
    assert _psi(arr, arr) < 0.01

def test_psi_very_different_distributions_is_high():
    ref = np.random.normal(0, 1, 1000)
    new = np.random.normal(10, 1, 1000)
    assert _psi(ref, new) > 0.2
```

**Why it matters:** Every ML engineer interview will ask "how do you ensure correctness?" You can point to tests that validate the penalty math, PSI computation, and schema validation.

---

### 1.2 CI/CD with GitHub Actions

**Create `.github/workflows/ci.yml`:**

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: pytest tests/unit -q --tb=short
      - run: pip install flake8 black
      - run: black --check .
      - run: flake8 ingestion/ features/ battery_intelligence/ ml/ monitoring/ --max-line-length=100
```

**Why it matters:** Shows you treat this as production code, not a notebook experiment.

---

### 1.3 Structured Logging

**Current state:** All modules use `print()`.

**Replace with a single logger setup in `configs/logging_config.py`:**

```python
import logging, sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

Every module then does: `logger = get_logger(__name__)`

For production, add a `FileHandler` that writes to `logs/pipeline_YYYY-MM-DD.log` and rotate daily.

**Why it matters:** Structured logs are how you debug production issues. Interviewers notice whether you use print vs logging.

---

### 1.4 Type Hints & Pydantic Config

Add full type hints to all function signatures across `ingestion/`, `features/`, `battery_intelligence/`, `ml/`.

Replace `configs/settings.py` path constants with a Pydantic BaseSettings model:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: Path = Path("data")
    mlflow_uri: str = "http://localhost:5001"
    soc_mae_alert_threshold: float = 1.5
    drop_quality_mask: int = 52

    class Config:
        env_file = ".env"

settings = Settings()
```

This makes every config value overridable via environment variable — critical for containerized deployment.

---

### 1.5 Pre-commit Hooks

**Create `.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks: [{id: black}]
  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks: [{id: isort}]
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks: [{id: flake8, args: [--max-line-length=100]}]
```

---

## 2. ML & Algorithm Improvements

### 2.1 Time-Series Cross-Validation (Fix the Leakage Risk)

**Current issue:** The existing split shuffles trips randomly. With time-series data, future trips can leak into training.

**Replace with walk-forward validation:**

```python
def walk_forward_splits(dates: list[str], n_folds: int = 5):
    """
    Expanding window — train on [0..k], validate on [k+1..k+n].
    Never uses future data to train.
    """
    fold_size = len(dates) // (n_folds + 1)
    for i in range(1, n_folds + 1):
        train = dates[:i * fold_size]
        val   = dates[i * fold_size:(i + 1) * fold_size]
        yield train, val
```

**Report average + std of MAE across folds.** This is the correct evaluation for time-series and is a standard interview question.

---

### 2.2 Hyperparameter Tuning with Optuna

Add `scripts/tune_soc.py`:

```python
import optuna

def objective(trial):
    params = {
        "n_estimators":   trial.suggest_int("n_estimators", 100, 500),
        "max_depth":      trial.suggest_int("max_depth", 3, 8),
        "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return mean_absolute_error(y_val, model.predict(X_val))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
```

Log best params to MLflow alongside the model.

---

### 2.3 SHAP Explainability (Already Installed)

Add `scripts/explain_model.py`:

```python
import shap, matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Summary plot — save as PNG for README
shap.summary_plot(shap_values, X_val, show=False)
plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")

# Force plot for a single prediction — great for demo
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_val.iloc[0],
    matplotlib=True,
)
```

**Interview value:** "I used SHAP to verify the model learned physically meaningful patterns — SOC 60s lag dominates prediction as expected, current and voltage have secondary importance. This confirmed the model isn't fitting noise."

---

### 2.4 Uncertainty Quantification — Conformal Prediction

XGBoost gives a point estimate. Add prediction intervals:

```python
from sklearn.model_selection import train_test_split

# Calibration set (never seen during training)
y_calib_true = ...
y_calib_pred = model.predict(X_calib)
residuals = np.abs(y_calib_true - y_calib_pred)

# 95% conformal interval
alpha = 0.05
q = np.quantile(residuals, 1 - alpha)

def predict_with_interval(X):
    pred = model.predict(X)
    return pred - q, pred, pred + q   # lower, point, upper
```

This adds `soc_lower_95`, `soc_pred`, `soc_upper_95` to every prediction — operationally useful for range anxiety alerts.

---

### 2.5 Anomaly Detection on Cell Voltages — Isolation Forest

Add `scripts/detect_cell_anomalies.py`:

**Goal:** Automatically detect degraded cells or sensor faults from voltage gap time series.

```python
from sklearn.ensemble import IsolationForest

# Feature vector per cell per day: [mean_gap, std_gap, trend_slope, outlier_rate]
features = build_cell_daily_features(gold_cells_df)

detector = IsolationForest(contamination=0.05, random_state=42)
detector.fit(features[features["dt"] < FIRMWARE_DATE])  # train on clean baseline

scores = detector.decision_function(features)
anomalies = features[detector.predict(features) == -1]
```

**What this catches:** The M1_C25 firmware anomaly (−43 mV → −4 mV drop) should appear as a high-confidence anomaly on March 27. This is a great demo story.

---

### 2.6 Remaining Useful Life (RUL) Prediction

**Goal:** Predict how many days until a cell crosses a degradation threshold.

```python
# Build a daily cell health time series
# Label: days_until_monitor = days until daily_health == "Monitor"
# Feature: rolling stats of Mn_Cm_delta over past 30 days

# Simple regression first:
from xgboost import XGBRegressor
rul_model = XGBRegressor()
rul_model.fit(X_rul_train, y_days_until_threshold)
```

This upgrades the project from "diagnosis" (what is happening now) to "prognosis" (when will something fail) — a significant leap in battery intelligence.

---

### 2.7 Multi-Output SOC Forecasting

**Current:** Predicts SOC at t+300s only.

**Upgrade:** Predict at t+60s, t+300s, t+600s simultaneously using `MultiOutputRegressor` or a single XGBoost with multiple targets:

```python
targets = ["y_soc_t_plus_60s", "y_soc_t_plus_300s", "y_soc_t_plus_600s"]
from sklearn.multioutput import MultiOutputRegressor
model = MultiOutputRegressor(XGBRegressor(**best_params))
model.fit(X_train, y_train[targets])
```

This gives a full SOC trajectory forecast, not a single point.

---

### 2.8 Deep Learning — LSTM for SOC (Optional but High Signal)

For roles that need deep learning exposure:

```python
import torch
import torch.nn as nn

class SOCLSTMForecaster(nn.Module):
    def __init__(self, input_size=61, hidden=128, layers=2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):  # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # last hidden state → prediction
```

Use a 60-second input window (60 timesteps × 61 features). Compare LSTM vs XGBoost on MAE — XGBoost will likely win on this dataset size, which is also a valid and interesting finding to discuss.

---

### 2.9 Change Point Detection

**Goal:** Automatically detect the firmware update date without manual labeling.

```python
import ruptures as rpt

# Run on daily M1_C25 gap mean time series
signal = daily_cell_stats["M1_C25_gap_mean"].values
algo = rpt.Pelt(model="rbf").fit(signal)
change_points = algo.predict(pen=3)
# Should recover ~March 27 automatically
```

This is a strong algorithmic talking point: "The firmware update on March 27 was detected automatically using PELT change point detection on the M1_C25 voltage gap series."

---

## 3. MLOps Infrastructure

### 3.1 Model Serving API — FastAPI

**Create `api/main.py`:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb, numpy as np, json
from pathlib import Path

app = FastAPI(title="EV SOC Forecast API", version="1.0")

# Load model at startup
MODEL_DIR = Path("models/soc_forecast/v4__2026-06-02__fdd89133")
model = xgb.XGBRegressor()
model.load_model(str(MODEL_DIR / "model.json"))
features = json.loads((MODEL_DIR / "feature_set.json").read_text())["features"]

class TelemetryInput(BaseModel):
    soc_pct: float
    battery_current_a: float
    stack_voltage_v: float
    output_power_kw: float
    motor_speed_rpm: float
    # ... all 61 features with defaults=0

class SOCPrediction(BaseModel):
    soc_t_plus_300s: float
    soc_lower_95: float
    soc_upper_95: float
    model_version: str
    confidence: str   # HIGH / MEDIUM / LOW based on interval width

@app.post("/predict", response_model=SOCPrediction)
async def predict_soc(data: TelemetryInput):
    X = np.array([[getattr(data, f, 0.0) for f in features]], dtype="float32")
    pred = float(np.clip(model.predict(X)[0], 0, 100))
    interval = 0.54 * 1.96  # approximate from calibration set
    return SOCPrediction(
        soc_t_plus_300s=pred,
        soc_lower_95=max(0, pred - interval),
        soc_upper_95=min(100, pred + interval),
        model_version="v4__2026-06-02__fdd89133",
        confidence="HIGH" if interval < 1.0 else "MEDIUM",
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model": "v4__2026-06-02__fdd89133"}
```

**Run with:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`

**Add a `/batch_predict` endpoint** that accepts a list of timesteps (a full trip) and returns the SOC trajectory.

---

### 3.2 Docker + docker-compose

**`Dockerfile`:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`:**

```yaml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    env_file: .env

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports: ["5001:5000"]
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000

  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./reports:/app/reports
    env_file: .env
    command: python scripts/run_daily_pipeline.py
```

Run the full system: `docker-compose up`

---

### 3.3 MLflow Model Registry (Staging → Production Promotion)

Upgrade the current MLflow tracking to use the full registry:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5001")

# After training — register model
with mlflow.start_run() as run:
    mlflow.xgboost.log_model(model, "model", registered_model_name="soc_forecast_loader")
    mlflow.log_params(best_params)
    mlflow.log_metrics({"val_mae": val_mae, "val_rmse": val_rmse})

# Promote to staging after review
client.transition_model_version_stage(
    name="soc_forecast_loader",
    version="4",
    stage="Staging",
)

# Promote to production after drift check passes
client.transition_model_version_stage(
    name="soc_forecast_loader",
    version="4",
    stage="Production",
)
```

The daily pipeline then loads the Production-stage model instead of a hardcoded version string.

---

### 3.4 Automated Retraining Pipeline

**Create `scripts/retrain_if_needed.py`:**

```python
"""
Checks last 7 days of MLflow MAE metrics.
If rolling MAE > threshold OR PSI drift detected → triggers retrain.
"""

def should_retrain(vehicle_id: str, window_days: int = 7) -> tuple[bool, str]:
    recent_mae = fetch_recent_mae_from_mlflow(vehicle_id, window_days)
    drift_status = load_latest_drift_report(vehicle_id)

    if recent_mae > MAE_RETRAIN_THRESHOLD:  # e.g. 1.0%
        return True, f"rolling MAE {recent_mae:.3f}% > threshold"

    if drift_status["overall_status"] == "DRIFT":
        n_drifted = drift_status["n_drift"]
        return True, f"{n_drifted} features in DRIFT state"

    return False, "model healthy"

if __name__ == "__main__":
    for vid in ["EV01", "EV02"]:
        needs_retrain, reason = should_retrain(vid)
        if needs_retrain:
            print(f"{vid}: retraining — {reason}")
            subprocess.run(["python", "scripts/train_soc_v3.py", "--vehicle-id", vid])
```

Add this as a weekly cron job or as the final step in `run_daily_pipeline.py`.

---

### 3.5 Data Versioning with DVC

```bash
pip install dvc dvc-s3

dvc init
dvc add data/silver/ data/gold/ models/soc_forecast/
dvc remote add -d s3remote s3://your-bucket/ev-analytics-dvc

# Each training run
dvc commit
git add data/silver.dvc models/soc_forecast.dvc
git commit -m "data: update silver for 2026-05-01"
dvc push
```

Now every model version is linked to the exact data version that produced it.

---

## 4. System Design & API Layer

### 4.1 Full API Design

```
POST /api/v1/predict/soc          → single-timestep SOC forecast
POST /api/v1/predict/soc/batch    → full trip SOC trajectory
GET  /api/v1/vehicles/{id}/health → latest health score + alerts
GET  /api/v1/vehicles/{id}/cells  → cell-level gap profile
GET  /api/v1/models/active        → active model version per machine type
POST /api/v1/pipeline/trigger     → manually trigger a pipeline run for a date
GET  /api/v1/drift/latest         → latest drift report per vehicle
GET  /api/v1/metrics/daily        → daily MAE from MLflow, last 30 days
```

Add request validation, API versioning (`/api/v1/`), and OpenAPI docs (FastAPI provides them automatically at `/docs`).

---

### 4.2 PostgreSQL for Pipeline State & Alerts

**Current state:** `data/state/pipeline_state.json` is a flat file.

**Replace with PostgreSQL:**

```sql
CREATE TABLE pipeline_runs (
    id          SERIAL PRIMARY KEY,
    dt          DATE NOT NULL,
    vehicle_id  VARCHAR(10) NOT NULL,
    step        VARCHAR(30) NOT NULL,
    status      VARCHAR(10) NOT NULL,  -- ok / failed / skipped
    message     TEXT,
    duration_s  FLOAT,
    created_at  TIMESTAMP DEFAULT NOW(),
    UNIQUE (dt, vehicle_id, step)
);

CREATE TABLE alerts (
    id           SERIAL PRIMARY KEY,
    dt           DATE NOT NULL,
    vehicle_id   VARCHAR(10) NOT NULL,
    rule_id      VARCHAR(50) NOT NULL,
    severity     VARCHAR(10) NOT NULL,
    evidence     TEXT,
    count        INTEGER,
    created_at   TIMESTAMP DEFAULT NOW()
);

CREATE TABLE daily_metrics (
    id           SERIAL PRIMARY KEY,
    dt           DATE NOT NULL,
    vehicle_id   VARCHAR(10) NOT NULL,
    model_run    VARCHAR(50),
    mae_pct      FLOAT,
    n_samples    INTEGER,
    drift_status VARCHAR(10),
    created_at   TIMESTAMP DEFAULT NOW()
);
```

This gives you queryable history instead of a JSON file, powers the API endpoints, and enables trend analysis.

---

### 4.3 Celery + Redis for Async Pipeline

For heavy processing jobs (backfills, retraining), move to a task queue instead of blocking the API:

```python
# tasks/celery_tasks.py
from celery import Celery

app = Celery("ev_analytics", broker="redis://localhost:6379/0")

@app.task
def run_pipeline_for_date(dt: str):
    from scripts.run_daily_pipeline import process_date, load_state
    state = load_state()
    return process_date(dt, state)

@app.task
def retrain_soc_model(vehicle_id: str):
    import subprocess
    result = subprocess.run(["python", "scripts/train_soc_v3.py", "--vehicle-id", vehicle_id])
    return result.returncode
```

The API becomes: `POST /api/v1/pipeline/trigger` → `{"task_id": "abc123"}` → `GET /api/v1/tasks/abc123` → status.

Add to `docker-compose.yml`:
```yaml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]

celery_worker:
  build: .
  command: celery -A tasks.celery_tasks worker --loglevel=info
  depends_on: [redis]
```

---

### 4.4 Feature Store (Lightweight)

The current architecture recomputes features from silver on every training run. Instead, cache daily feature statistics:

```python
# After building gold for a date, save a feature stats row
feature_stats = {
    "dt": dt,
    "vehicle_id": vehicle_id,
    "feature_means": X.mean().to_dict(),
    "feature_stds": X.std().to_dict(),
    "n_rows": len(X),
}
# Write to data/feature_store/dt=YYYY-MM-DD/vehicle_id=EV01.parquet
```

Training then reads from the feature store instead of re-running the full feature pipeline, cutting training time significantly. This is what enterprise feature stores (Feast, Tecton) do at scale.

---

### 4.5 Architecture Diagram (System Design Interview)

```
                                          ┌─────────────────────────────────────┐
                                          │           EV Vehicles               │
                                          │   EV01 (Loader) | EV02 (Excavator)  │
                                          └──────────────┬──────────────────────┘
                                                         │ CAN/BMS telemetry (CSV)
                                                         ▼
                                          ┌─────────────────────────────────────┐
                                          │           Ingestion Layer            │
                                          │  validate → 1Hz resample → trip seg │
                                          │  quality flags → fault merge         │
                                          └──────────────┬──────────────────────┘
                                                         │ silver parquet
                                                         ▼
                                          ┌─────────────────────────────────────┐
                                          │         Feature Engineering          │
                                          │  rolling · lag · physics · cell agg  │
                                          │  61 features → gold parquet          │
                                          └──────────┬──────────────────────────┘
                           ┌─────────────────────────┤
                           │                         │ gold features
                           ▼                         ▼
            ┌──────────────────────┐   ┌────────────────────────────────┐
            │  XGBoost SOC Model   │   │   Battery Intelligence Engine  │
            │  0.54% val MAE       │   │   Health score 0–100           │
            │  61 features         │   │   12 rule-based alerts         │
            │  v4 production       │   │   Cell-level gap tracking      │
            └──────────┬───────────┘   └──────────────┬─────────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │      Monitoring Layer         │
                       │  PSI drift · KS test          │
                       │  MAE tracking in MLflow       │
                       │  Auto-retrain trigger         │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │        FastAPI Layer          │
                       │  /predict/soc                 │
                       │  /vehicles/{id}/health        │
                       │  /drift/latest                │
                       └──────────────┬───────────────┘
                                      │
                         ┌────────────┼────────────────┐
                         ▼            ▼                 ▼
               ┌──────────────┐  ┌─────────┐  ┌───────────────┐
               │  PostgreSQL  │  │  Redis  │  │  MLflow        │
               │  (state, DB) │  │  (queue)│  │  (registry)    │
               └──────────────┘  └─────────┘  └───────────────┘
```

---

## 5. Production Monitoring & Alerting

### 5.1 Alerting — Slack / Email on Drift

Extend `run_daily_pipeline.py` to send targeted alerts:

```python
def send_drift_alert(vehicle_id: str, drift_report: dict, mae: float):
    if drift_report["overall_status"] != "DRIFT" and mae < MAE_ALERT_THRESHOLD:
        return

    subject = f"[{vehicle_id}] SOC Model Alert — {drift_report['dt']}"
    body = (
        f"Vehicle: {vehicle_id}\n"
        f"Date: {drift_report['dt']}\n"
        f"Daily MAE: {mae:.3f}%\n"
        f"Drift status: {drift_report['overall_status']}\n"
        f"Features in DRIFT: {drift_report['n_drift']}\n\n"
        f"Drifted features:\n"
        + "\n".join(
            f"  {r['feature']}: PSI={r['psi']:.3f}, shift={r['mean_shift_pct']:.1f}%"
            for r in drift_report["features"]
            if r["status"] == "DRIFT"
        )
    )
    # Send via scripts/send_report_email.py or Slack webhook
```

---

### 5.2 Grafana Dashboard (if running locally)

Point Grafana at the PostgreSQL `daily_metrics` table:

| Panel | Query |
|---|---|
| Daily MAE over time (line) | `SELECT dt, mae_pct FROM daily_metrics WHERE vehicle_id = 'EV01'` |
| Health score distribution (hist) | From `battery_health` outputs |
| Active drift features (table) | From `drift_reports` JSON |
| Alert counts by type (bar) | From `alerts` table |
| Model version timeline | From `pipeline_runs` |

---

### 5.3 Model Performance SLOs

Define and track these thresholds explicitly:

| SLO | Target | Alert threshold |
|---|---|---|
| Daily SOC MAE | < 0.8% | > 1.0% (warning), > 1.5% (critical) |
| Drift features | 0 in DRIFT | > 3 features drifting |
| Pipeline success rate | > 99% | any critical step failure |
| Battery health data freshness | < 24h lag | > 48h without update |

---

## 6. Data Engineering Upgrades

### 6.1 Great Expectations for Data Contracts

Validate raw data quality before it enters the pipeline:

```python
import great_expectations as gx

context = gx.get_context()
suite = context.add_expectation_suite("raw_telemetry")

suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
    column="soc_pct", min_value=0, max_value=100
))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column="timestamp"))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(
    column="stack_voltage_v", mostly=0.95
))
```

Gate the pipeline: if validation fails, halt ingestion and alert.

---

### 6.2 Apache Airflow DAG (Replace the manual daily script)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

with DAG(
    "ev_daily_pipeline",
    schedule_interval="0 6 * * *",   # 6am daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    ingest = PythonOperator(task_id="silver", python_callable=step_silver, ...)
    cells  = PythonOperator(task_id="cells",  python_callable=step_cells, ...)
    gold   = PythonOperator(task_id="gold",   python_callable=step_gold, ...)
    soc    = PythonOperator(task_id="soc",    python_callable=step_soc, ...)
    health = PythonOperator(task_id="battery",python_callable=step_battery, ...)

    ingest >> cells >> gold >> soc >> health
```

This replaces the custom state-machine in `run_daily_pipeline.py` with a proper DAG with retry logic, backfill support, and a web UI.

---

### 6.3 Pandera Schema Validation

Replace YAML-based validators with Pandera type-safe schemas:

```python
import pandera as pa
from pandera.typing import DataFrame, Series

class SilverSchema(pa.DataFrameModel):
    timestamp: Series[pa.DateTime]
    vehicle_id: Series[str]
    soc_pct: Series[float] = pa.Field(ge=0, le=100, nullable=False)
    battery_current_a: Series[float] = pa.Field(nullable=True)
    stack_voltage_v: Series[float] = pa.Field(ge=200, le=750, nullable=True)
    quality_flag: Series[int] = pa.Field(ge=0)
    trip_id: Series[str] = pa.Field(nullable=True)

@pa.check_types
def validate_silver(df: DataFrame[SilverSchema]) -> DataFrame[SilverSchema]:
    return df
```

---

## 7. Interview Talking Points by Role

### Software Engineer

> "I built a production-grade data pipeline that processes 1Hz CAN telemetry through a medallion architecture (raw → silver → gold). It has atomic writes to prevent corrupted parquet files mid-pipeline, state tracking via JSON so daily runs are idempotent, structured logging throughout, and 40+ unit and integration tests covering schema validation, penalty math, and drift computation. The pipeline is containerised with Docker and tested on every push via GitHub Actions."

Key things to show: tests, logging, CI config, Docker, the state-machine in `run_daily_pipeline.py`

---

### Data Scientist

> "I trained an XGBoost model to predict SOC 5 minutes ahead on 1Hz battery telemetry, achieving 0.54% MAE on a 4-day held-out window versus a 1.78% persistence baseline — a 3.3× improvement. I used walk-forward cross-validation to prevent temporal leakage, SHAP to verify the model learned physically meaningful patterns (SOC lag features dominate, as expected from Coulomb counting theory), and conformal prediction to attach calibrated uncertainty intervals to every forecast."

Key things to show: walk-forward CV code, SHAP summary plot, uncertainty intervals, firmware era analysis

---

### ML/AI Engineer

> "The system has a full MLOps loop: MLflow tracks every experiment with params, metrics, and feature sets. The model registry has staging → production promotion. The daily pipeline computes PSI and KS-test drift per feature and auto-triggers retraining when either the rolling MAE exceeds 1.0% or more than 3 features show DRIFT status. The serving layer is a FastAPI endpoint returning point predictions with 95% conformal intervals, versioned by model run ID."

Key things to show: MLflow registry, drift monitoring code, retraining trigger, FastAPI serve endpoint

---

### Systems / Backend Engineer

> "The architecture has a clean separation between the batch pipeline (daily Celery tasks), the ML serving API (FastAPI + uvicorn), and the monitoring layer (MLflow + PostgreSQL). The API is async, versioned at /api/v1/, and returns structured JSON with model version metadata on every response. For large batch jobs like backfills, I queue tasks via Celery/Redis so the API stays responsive. The system is fully containerised via docker-compose with separate services for the API, pipeline worker, MLflow, Redis, and PostgreSQL."

Key things to show: docker-compose, API design, async task queue, PostgreSQL schema

---

## 8. Execution Priority

Build in this order — each phase stands on its own as a resume addition:

| Phase | What | Time | Unlocks |
|---|---|---|---|
| **P0** | Unit tests (15 tests) + GitHub Actions CI | 2 days | "Production-grade code" claim |
| **P1** | FastAPI `/predict` endpoint + Docker | 2 days | "Deployed ML system" claim |
| **P2** | SHAP explainability + walk-forward CV | 1 day | "ML rigor" claim |
| **P3** | Structured logging + Pydantic config | 1 day | SWE depth |
| **P4** | Optuna hyperparameter tuning | 1 day | ML engineering depth |
| **P5** | Isolation Forest anomaly detection | 2 days | Unsupervised ML, new algorithm |
| **P6** | Conformal prediction intervals | 1 day | ML uncertainty, practical value |
| **P7** | Change point detection (ruptures) | 1 day | Signal processing, algorithms |
| **P8** | PostgreSQL for state + MLflow registry | 3 days | Full MLOps story |
| **P9** | Celery + Redis async pipeline | 2 days | Systems/backend depth |
| **P10** | RUL prediction | 3 days | Prognostics, new model track |
| **P11** | Airflow DAG | 2 days | Data engineering depth |
| **P12** | Grafana dashboard | 2 days | Observability, demo |

**P0–P3 alone transform this from "ML experiment" to "production ML system."**
**P0–P7 gives you a strong story for any ML/data science/SWE role.**
**P0–P12 is a full-stack ML platform that competes with enterprise systems.**
