import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import (
    regression_metrics,
    _safe_mape,
    error_distribution,
    soc_bucket_metrics,
    persistence_baseline,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAE / RMSE / MAPE
# ═══════════════════════════════════════════════════════════════════════════════

def test_mae_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y, verbose=False)
    assert m["mae"] == pytest.approx(0.0)


def test_mae_constant_error():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = y_true + 2.0
    m = regression_metrics(y_true, y_pred, verbose=False)
    assert m["mae"] == pytest.approx(2.0)


def test_rmse_correct():
    # errors = [0, 0, 3] → RMSE = sqrt((0 + 0 + 9) / 3) = sqrt(3)
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 3.0])
    m = regression_metrics(y_true, y_pred, verbose=False)
    assert m["rmse"] == pytest.approx(np.sqrt(3.0))


def test_mape_correct():
    # y_true=100, y_pred=110 → |error|/max(|100|,1) * 100 = 10%
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    m = regression_metrics(y_true, y_pred, verbose=False)
    assert m["mape"] == pytest.approx(10.0)


def test_regression_metrics_returns_expected_keys():
    y = np.ones(5)
    m = regression_metrics(y, y, verbose=False)
    assert set(m.keys()) == {"mae", "rmse", "mape"}


# ═══════════════════════════════════════════════════════════════════════════════
# _safe_mape
# ═══════════════════════════════════════════════════════════════════════════════

def test_safe_mape_near_zero_does_not_crash():
    # y_true=0 would cause div-by-zero; eps clips denominator to 1.0 → |10-0|/1 * 100 = 1000%
    mape = _safe_mape(np.array([0.0, 0.0]), np.array([10.0, 10.0]), eps=1.0)
    assert np.isfinite(mape)
    assert mape == pytest.approx(1000.0)


def test_safe_mape_normal_case():
    # y_true=50, y_pred=55 → denom = max(50, 1) = 50 → 5/50 * 100 = 10%
    mape = _safe_mape(np.array([50.0]), np.array([55.0]))
    assert mape == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# error_distribution
# ═══════════════════════════════════════════════════════════════════════════════

def test_error_distribution_returns_expected_keys():
    d = error_distribution(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    for key in ("mean", "std", "p5", "p25", "p50", "p75", "p95", "max_abs"):
        assert key in d


def test_error_distribution_p50_correct():
    d = error_distribution(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert d["p50"] == pytest.approx(3.0)


def test_error_distribution_max_abs_uses_absolute_value():
    # Most negative value −10 should dominate max_abs
    d = error_distribution(pd.Series([-10.0, 2.0, 3.0]))
    assert d["max_abs"] == pytest.approx(10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# persistence_baseline
# ═══════════════════════════════════════════════════════════════════════════════

def test_persistence_baseline_correct():
    # y_true=[70, 60], soc_now=[80, 80] → errors=[10, 20] → MAE=15
    y_true  = np.array([70.0, 60.0])
    soc_now = np.array([80.0, 80.0])
    assert persistence_baseline(y_true, soc_now) == pytest.approx(15.0)


def test_persistence_baseline_perfect():
    y = np.array([50.0, 60.0, 70.0])
    assert persistence_baseline(y, y) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# soc_bucket_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_soc_bucket_only_returns_non_empty_buckets():
    df = pd.DataFrame({
        "soc_pct": [5.0, 10.0, 15.0],
        "y_true":  [5.0, 10.0, 15.0],
        "y_pred":  [5.0, 10.0, 15.0],
    })
    rows = soc_bucket_metrics(df, "y_true", "y_pred", "soc_pct")
    assert len(rows) == 1
    assert rows[0]["soc_bucket"] == "0–20%"


def test_soc_bucket_mae_correct():
    # All data in 80–100 bucket, constant error of 1
    df = pd.DataFrame({
        "soc_pct": [85.0, 90.0],
        "y_true":  [85.0, 90.0],
        "y_pred":  [86.0, 91.0],
    })
    rows = soc_bucket_metrics(df, "y_true", "y_pred", "soc_pct")
    assert len(rows) == 1
    assert rows[0]["mae"] == pytest.approx(1.0)


def test_soc_bucket_splits_data_correctly():
    # Data in two buckets: 0–20 and 80–100
    df = pd.DataFrame({
        "soc_pct": [10.0, 15.0, 85.0, 90.0],
        "y_true":  [10.0, 15.0, 85.0, 90.0],
        "y_pred":  [10.0, 15.0, 85.0, 90.0],
    })
    rows = soc_bucket_metrics(df, "y_true", "y_pred", "soc_pct")
    buckets = {r["soc_bucket"] for r in rows}
    assert "0–20%" in buckets
    assert "80–100%" in buckets
    assert len(rows) == 2
