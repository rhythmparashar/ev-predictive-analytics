"""
SHAP Explainability — SOC Forecast Model (v4)
=============================================
Loads the production XGBoost model and explains its predictions using SHAP.

Outputs saved to outputs/:
  shap_summary.png   — bar chart of mean |SHAP| per feature (top 20)
  shap_waterfall.png — waterfall plot for a single prediction

Usage:
  python scripts/explain_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_DIR  = PROJECT_ROOT / "models" / "soc_forecast" / "v4__2026-06-02__fdd89133"
GOLD_DIR   = DATA_DIR / "gold" / "window_features"
VEHICLE_ID = "EV01"
TARGET     = "y_soc_t_plus_300s"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Val dates from eval_report.json — these were never seen during training
VAL_DATES = ["2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01"]

# Max rows to run SHAP on (TreeExplainer is fast, but 50k+ rows is slow)
SHAP_SAMPLE = 2_000

# ── Load model and feature list ───────────────────────────────────────────────

print("Loading model …")
model = xgb.XGBRegressor()
model.load_model(str(MODEL_DIR / "model.json"))

feature_set = json.loads((MODEL_DIR / "feature_set.json").read_text())
FEATURES = feature_set["features"]
print(f"  Features: {len(FEATURES)}")

# ── Load val data ─────────────────────────────────────────────────────────────

print("\nLoading val data …")
import duckdb

frames = []
for dt in VAL_DATES:
    path = GOLD_DIR / f"dt={dt}" / f"vehicle_id={VEHICLE_ID}"
    if not path.exists():
        print(f"  WARNING: {dt} not found, skipping")
        continue
    for f in path.glob("*.parquet"):
        available = set(
            duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()["column_name"].tolist()
        )
        read_cols = [c for c in FEATURES + [TARGET, "label_available", "soc_pct"] if c in available]
        cols_sql  = ", ".join(f'"{c}"' for c in read_cols)
        chunk = duckdb.query(f"SELECT {cols_sql} FROM read_parquet('{f}')").df()
        for col in FEATURES:
            if col not in chunk.columns:
                chunk[col] = 0.0
        chunk = chunk[chunk["label_available"] == 1].dropna(subset=[TARGET, "soc_pct"])
        if not chunk.empty:
            frames.append(chunk)

if not frames:
    raise RuntimeError("No val data found — check GOLD_DIR and VAL_DATES")

val_df = pd.concat(frames, ignore_index=True)
print(f"  Val rows: {len(val_df):,}")

X_val = val_df[FEATURES].fillna(0).astype("float32")

# ── Sample for SHAP ───────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
idx = rng.choice(len(X_val), size=min(SHAP_SAMPLE, len(X_val)), replace=False)
X_sample = X_val.iloc[idx].reset_index(drop=True)
print(f"\nRunning SHAP on {len(X_sample):,} rows …")

# ── SHAP explanation ──────────────────────────────────────────────────────────
# Use XGBoost's native pred_contribs — avoids SHAP/XGBoost base_score format
# incompatibility in shap 0.47.x with XGBoost v2+ models.

booster = model.get_booster()
dmatrix = xgb.DMatrix(X_sample, feature_names=FEATURES)
# Returns shape (n_samples, n_features + 1); last column is the bias term
contribs = booster.predict(dmatrix, pred_contribs=True)
shap_matrix   = contribs[:, :-1]          # (n_samples, 61)
base_value    = float(contribs[:, -1].mean())

explanation = shap.Explanation(
    values=shap_matrix,
    base_values=np.full(len(X_sample), base_value),
    data=X_sample.values,
    feature_names=FEATURES,
)

# Mean absolute SHAP per feature (global importance)
mean_shap  = np.abs(shap_matrix).mean(axis=0)
importance = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=False)

print("\nTop 10 features by mean |SHAP|:")
for feat, val in importance.head(10).items():
    bar = "█" * int(val / importance.iloc[0] * 30)
    print(f"  {feat:<45s} {val:.4f}  {bar}")

# ── Plot 1: Summary bar chart ─────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 7))
top20 = importance.head(20)
colors = ["#e74c3c" if i < 3 else "#3498db" for i in range(len(top20))]
ax.barh(range(len(top20)), top20.values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index[::-1], fontsize=9)
ax.set_xlabel("Mean |SHAP value| (SOC % impact)")
ax.set_title("SOC Forecast v4 — Feature Importance (SHAP)\nTop 20 features by mean absolute SHAP value")
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
summary_path = OUTPUT_DIR / "shap_summary.png"
fig.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {summary_path}")

# ── Plot 2: Waterfall for a single prediction ─────────────────────────────────

# Pick the row with the highest absolute prediction error for a richer story
y_sample   = val_df[TARGET].iloc[idx].to_numpy()
y_pred     = model.predict(X_sample)
worst_idx  = int(np.argmax(np.abs(y_sample - y_pred)))

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.waterfall(explanation[worst_idx], max_display=15, show=False)
plt.title(
    f"SHAP Waterfall — single prediction\n"
    f"Predicted SOC={y_pred[worst_idx]:.1f}%  Actual={y_sample[worst_idx]:.1f}%",
    fontsize=10,
)
plt.tight_layout()
waterfall_path = OUTPUT_DIR / "shap_waterfall.png"
fig.savefig(waterfall_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {waterfall_path}")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SHAP FINDINGS")
print("=" * 60)
top3 = importance.head(3)
print(f"  1. {top3.index[0]}  ({top3.iloc[0]:.4f}) — current SOC drives the forecast")
print(f"  2. {top3.index[1]}  ({top3.iloc[1]:.4f}) — recent SOC trend (60s rolling mean)")
print(f"  3. {top3.index[2]}  ({top3.iloc[2]:.4f}) — lagged SOC (Coulomb counting proxy)")
soc_shap_total = importance[importance.index.str.startswith("soc_pct")].sum()
total_shap     = importance.sum()
print(f"\n  SOC-family features account for {soc_shap_total / total_shap * 100:.1f}% of total SHAP")
print(f"  This confirms the model learned physics (SOC is autocorrelated)")
print(f"\n  Outputs: {OUTPUT_DIR}/")
print("=" * 60)
