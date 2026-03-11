# SOC Forecast Model Card — v1

## Status
- **Task:** State-of-Charge (SOC) forecast — 5 minutes ahead  
- **Model:** XGBoost Regressor  
- **Run ID:** `v1__2026-03-01__e50f8b21`  
- **Features:** 61  
- **Status:** Production-ready baseline  

This model predicts vehicle battery SOC **300 seconds ahead** using telemetry signals, rolling features, and lagged features derived from the Gold feature store.

---

# Data

## Training Data
Train dates:

2026-02-23  
2026-02-24  
2026-02-25  
2026-02-26  
2026-03-01  

Dataset size:

- **Train rows:** 184,144  
- **Trips:** 172  

## Validation Data

Primary validation:

2026-03-04

- **Validation rows:** 7,072  
- **Trips:** 18  

---

# Key Metrics

## Training / Validation

| Metric | Value |
|------|------|
| Train MAE | **0.2762** |
| Validation MAE | **0.5525** |
| Validation RMSE | 0.7048 |
| Validation MAPE | 0.7798 |
| Train-Val Gap | **0.276** |

Interpretation:

- Healthy generalization gap  
- No evidence of overfitting  

---

# Additional Out-of-Sample Evaluations

The model was evaluated on **two additional unseen days** after training.

| Eval Date | MAE | RMSE | MAPE |
|------|------|------|------|
| 2026-03-08 | **0.5420** | 0.6865 | 0.7097 |
| 2026-03-10 | **0.5380** | 0.6570 | 0.85 |

Baseline comparison example (2026-03-10):

| Model | MAE |
|------|------|
| Persistence | 1.224 |
| Rolling mean | 1.353 |
| **SOC model** | **0.538** |

Improvement vs persistence baseline:

**≈56% error reduction**

This confirms the model **generalizes across unseen operating days**.

---

# Error Distribution

| Percentile | Error |
|------|------|
| Mean | -0.045 |
| p50 | -0.106 |
| p95 | 1.177 |
| Max Absolute Error | 3.883 |

Observations:

- Prediction bias is minimal  
- 95% of predictions within ~1.18 SOC  
- Rare extreme errors occur during abrupt operating changes (e.g., charging transitions)

---

# SOC Bucket Performance

| SOC Range | MAE |
|------|------|
| 40–60% | **0.684** |
| 60–80% | 0.565 |
| 80–100% | 0.509 |

Observations:

- Prediction accuracy improves at **higher SOC**  
- Mid-range SOC (40–60%) is slightly harder to predict due to dynamic load changes.

---

# Per-Trip Behavior

Across evaluation days:

Typical trip MAE range:

0.40 – 0.65 SOC

Occasional difficult trips:

0.80 – 1.05 SOC

These occur during:

- abrupt discharge regimes  
- charging transitions  
- unusual power usage patterns  

No catastrophic failures were observed.

---

# Drift Monitoring

Drift monitoring uses:

- Population Stability Index (PSI)  
- Kolmogorov–Smirnov test  
- Mean shift percentage  
- Null rate change  

Recent evaluation days flagged **distribution drift across many features**, but **model performance remained stable**.

Example drift drivers:

- SOC distribution shifts  
- battery temperature changes  
- power/current regime changes  
- cumulative energy counter (`total_kwh_consumed`)

Interpretation:

These represent **operating regime shifts**, not model failure.

Drift alerts currently function as **environment change indicators**, not automatic retrain triggers.

---

# Strengths

- Strong improvement over naive baselines  
- Consistent performance across multiple unseen days  
- Healthy generalization gap  
- Stable error distribution  
- Fully reproducible model artifacts  
- Integrated drift monitoring pipeline  
- Modular data pipeline (raw → silver → gold)

---

# Known Limitations

- Prediction accuracy slightly worse in **mid SOC range (40–60%)**  
- Charging events may cause larger forecast errors  
- Drift monitor is sensitive to distribution changes even when model accuracy remains stable  
- Model relies heavily on SOC history features  

---

# Features to Remove Before Next Retrain

These features produce **permanent drift** or contain **no signal**.

Remove cumulative counters:

- total_kwh_consumed  
- total_kwh_consumed_lag_60s  
- total_kwh_consumed_lag_300s  

Remove dead sensor features:

- motor_temperature_c  
- motor_temperature_c_lag_300s  
- motor_temperature_c_roll_mean_300s  
- motor_temperature_c_roll_std_60s  

After removal:

Feature count will reduce from **61 → 54**.

---

# Investigations for Next Iteration

### Charging behavior
Improve prediction during charging transitions.

Actions:

- validate `is_charging_current` logic  
- validate `is_parked_charging` logic  
- improve charge state detection  

---

### Fault-aware evaluation
Add evaluation slice:

MAE when `fault_any == 0`  
MAE when `fault_any == 1`

To measure prediction degradation under fault conditions.

---

### Drift monitoring improvements

Improve drift interpretation:

- distinguish **regime shift** from **model failure**  
- reduce false positive drift alerts  

---

# Next Planned Improvements (Phase 4)

- Feature cleanup (remove drift-prone features)  
- SHAP explainability  
- Hyperparameter tuning with Optuna  
- Fault-aware evaluation metrics  
- Multi-vehicle generalization  
- Retrain trigger based on rolling MAE  

---

# Summary

The SOC forecasting model achieves:

**MAE ≈ 0.54 SOC**

across **multiple independent evaluation days**.

Performance significantly outperforms baseline methods and remains stable despite operating regime changes.

The model is therefore considered:

**Production-ready baseline (v1).**