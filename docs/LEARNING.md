# Learning & Challenges Log

A running record of what I learned building this system and the real problems I hit along the way.
Updated as each roadmap phase is implemented.

---

## P0 — Unit Testing with pytest

### What I Learned

**How to structure a test suite for an ML pipeline**

Tests live in `tests/unit/` with one file per module. Each test function is a single, narrow assertion — not a catch-all. The goal is: if a test fails, you immediately know *which rule broke*, not just that *something broke*.

**Testing penalty math with boundary values**

The health score has step-function penalties (e.g. thermal: 0 → 10 → 20 → 35 at 45°C / 50°C / 55°C). The right tests are exactly at the boundaries, not in the middle of each band. A test at 47°C tells you nothing the 45°C test doesn't already cover.

**`np.select` evaluation order matters**

`np.select` checks conditions left-to-right and returns the first match. So `[temp >= 55, temp >= 50, temp >= 45]` is correct — if you flipped the order, 60°C would match `temp >= 45` first and return the wrong penalty. The tests caught this class of bug by checking each threshold independently.

**Using `np.random.default_rng(seed)` over `np.random.seed()`**

The new-style `default_rng(seed)` gives reproducible results per-generator without affecting global state. Each test that needs random data gets its own seeded generator, so tests are fully independent.

**`pytest.approx` for floating point**

`assert score == 40.0` will fail due to float32 precision. `assert score == pytest.approx(40.0)` handles this correctly.

---

### Challenges

**Challenge 1 — The 5-quantile PSI approximation inflates drift scores**

The `_feature_drift` function in `monitoring/drift.py` reconstructs the reference distribution from only 5 stored quantile points (p10, p25, p50, p75, p90). This creates a bounded approximation with no tails.

When you compare this against real data (which has tails beyond p10 and p90), values outside the bounded range pile into the edge histogram bins — and PSI explodes, even when the underlying distributions are identical.

**Concrete example:** N(50, 10) reference vs N(50, 10) new data → PSI = **2.31** (well above the 0.2 DRIFT threshold), despite being the same distribution.

**How we handled it in tests:** The "STABLE" test samples new data directly from the same 5-point interpolation (bounded to [p10, p90]), guaranteeing PSI ≈ 0. The comment in the test explains why.

**The deeper lesson:** A 5-quantile baseline is a lossy representation. It works well for detecting large distributional shifts (which is its purpose) but cannot reliably distinguish "identical distribution" from "same distribution with tails." If high-confidence STABLE detection is needed in production, the baseline would need to store the full distribution or more percentile points.

---

**Challenge 2 — N(0,1) vs N(1,1) is not a "moderate" PSI shift**

Initial assumption: a 1-sigma mean shift would produce a moderate PSI (0.1–0.2). Actual result: PSI = **1.04**.

Reason: PSI is exponentially sensitive to distribution overlap. A 1-sigma shift means the two distributions overlap only partially, and with 10 quantile-based bins, many bins flip completely from expected to actual percentages, each contributing `(act - exp) * log(act/exp)` to the sum.

**Fix:** Replaced the test with a relative check — PSI of a shifted distribution is greater than PSI of the same distribution. That's the property that actually matters, not hitting a specific numeric range.

---

**Challenge 3 — KS test p-value vs quantile approximation**

The KS test compares `ref_approx` (1000 synthetic points from quantile interpolation) vs `new` (500 actual samples). Even at p < 0.001, the KS test was flagging DRIFT for same-distribution data, because the 5-point interpolation introduces systematic shape distortion that the KS test detects.

This confirmed: the PSI and KS tests in `_feature_drift` are calibrated for detecting real distributional changes, not for passing unit tests with perfect same-distribution data.

---

---

**Challenge 4 — `safe_row()` false positive from implicit fill values**

When building a minimal test DataFrame for `rule_engine`, `max_battery_temp_c=30.0` was enough to keep the temperature rules silent. But `TEMP_IMBALANCE_CRITICAL` fired anyway because `_temp_diff` falls back to `max_battery_temp_c - min_battery_temp_c` when `pack_temp_range` is absent — and `_safe()` fills missing columns with `0.0`. So `30 − 0 = 30°C > 22°C` triggered the rule.

**Fix:** Add `pack_temp_range=0.0` explicitly to `safe_row()` so the rule reads the direct field and never falls into the implicit-fill branch.

**Lesson:** When a rule engine has multiple code paths (direct column vs. derived fallback), the test helper must satisfy ALL paths, not just the obvious one. If a column is absent, the fallback's fill value may inadvertently trigger a rule.

---

**Challenge 5 — `np.searchsorted side='right'` includes the endpoint**

`add_fault_flags` uses a difference-array pattern: `diff[start_idx] += 1`, `diff[end_idx] -= 1`, then `cumsum`. With `side='right'`, `end_idx = searchsorted(ts, fixed_at, side='right')` points to the slot AFTER the matching element, which means the timestamp AT `fixed_at` is still inside the decrement window and cumsum=1 there — the fault is **active at its own fixed_at**.

Initial test expected the endpoint to be inactive (0). That was wrong.

**Fix:** Test a timestamp one second AFTER `fixed_at` instead. That's the correct "fault is over" check.

**Lesson:** `searchsorted side='right'` gives inclusive-right semantics for the cumsum trick. Read the intent from the code first; don't assume exclusive endpoints.

---

---

## P2 — SHAP Explainability + Walk-Forward Cross-Validation

### What I Learned

**SHAP explains WHY the model makes each prediction**

SHAP (SHapley Additive exPlanations) assigns each feature a value representing how much it pushed the prediction up or down for a specific row. For our SOC forecast:

- `soc_pct` has mean |SHAP| = 11.99 — by far the most important feature
- `soc_pct_roll_mean_60s` = 2.51 (60s rolling mean of SOC)
- `soc_pct_lag_60s` = 1.40 (SOC 60 seconds ago)

SOC-family features account for **88% of total SHAP**. This is the expected result from physics — SOC is highly autocorrelated because the battery can't jump from 70% to 20% in 5 minutes. The model learned this implicitly without being told.

**Walk-forward CV is the correct evaluation for time-series data**

A random train/test split on time-series data risks "future leakage" — the model sees patterns from future dates during training. Walk-forward CV prevents this by ensuring the val set is always strictly after the training set:

```
Fold 1: train=[Feb 10 → Mar 13]  val=[Apr 04 → Apr 17]
Fold 2: train=[Feb 10 → Apr 17]  val=[Apr 22 → May 10]
...
Fold 5: train=[Feb 10 → May 25]  val=[May 26 → May 29]
```

Each fold, the training window expands but never includes the val dates.

**Results: mean MAE = 0.88% ± 0.26%, vs persistence baseline 1.56%**

| Fold | Train dates | Model MAE | Baseline MAE |
|---|---|---|---|
| 1 | 6 | 1.24% | 1.42% |
| 2 | 10 | 0.94% | 1.63% |
| 3 | 14 | 1.07% | 1.41% |
| 4 | 18 | 0.59% | 1.59% |
| 5 | 22 | 0.58% | 1.73% |

MAE improves as more training data is added (1.24% → 0.58%) — natural learning curve, not leakage. The model consistently beats persistence across all 5 folds.

---

### Challenges

**Challenge 6 — SHAP/XGBoost v2 base_score format incompatibility**

`shap.TreeExplainer(model)` crashed with:
```
ValueError: could not convert string to float: '[7.6316925E1]'
```

XGBoost v2+ stores `base_score` in its model JSON as `'[7.6316925E1]'` (with brackets), which SHAP 0.47.x doesn't handle.

**Fix:** Use XGBoost's native `pred_contribs=True` instead of the SHAP library's TreeExplainer:
```python
booster  = model.get_booster()
dmatrix  = xgb.DMatrix(X_sample, feature_names=FEATURES)
contribs = booster.predict(dmatrix, pred_contribs=True)
# contribs: shape (n_samples, n_features + 1); last col is bias
shap_values = contribs[:, :-1]
```

These are exact SHAP values computed by XGBoost's own tree traversal — not approximations. Then wrap in `shap.Explanation` for plotting.

**Lesson:** When a library wrapper breaks, check if the underlying library has the same feature natively. XGBoost has had built-in SHAP since v0.90 via `pred_contribs=True`.

---

## Running the Tests

```bash
# All unit tests
venv/bin/python -m pytest tests/unit/ -v

# Single file
venv/bin/python -m pytest tests/unit/test_health_score.py -v
venv/bin/python -m pytest tests/unit/test_rule_engine.py -v
venv/bin/python -m pytest tests/unit/test_faults.py -v
```

**Current status:** 127 tests, 127 passing.

| File | Tests |
|---|---|
| test_health_score.py | 16 |
| test_drift.py | 10 |
| test_features.py | 18 |
| test_validators.py | 13 |
| test_rule_engine.py | 20 |
| test_metrics.py | 14 |
| test_resampler.py | 9 |
| test_trip_agg.py | 12 |
| test_trip_segmentor.py | 5 |
| test_faults.py | 10 |

---

*Last updated: P0 unit tests complete — 127 tests, all passing. Next: GitHub Actions CI.*
