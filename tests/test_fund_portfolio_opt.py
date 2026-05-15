import numpy as np
import pandas as pd
import pytest

from quant.fund.portfolio_opt import (
    current_weights,
    estimate_mu_cov,
    reconcile,
    run_all_methods,
)


def _make_nav(n: int = 200, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n)
    cols = [f"F{i}" for i in range(n_assets)]
    data = {
        c: 1.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n)) for c in cols
    }
    return pd.DataFrame(data, index=idx)


def test_estimate_mu_cov_shapes():
    nav = _make_nav()
    mu, cov, assets = estimate_mu_cov(nav, lookback=120)
    assert mu.shape == (3,)
    assert cov.shape == (3, 3)
    assert assets == ["F0", "F1", "F2"]


def test_estimate_mu_cov_raises_when_too_few_obs():
    nav = _make_nav(n=20)
    with pytest.raises(ValueError, match="可用数据点"):
        estimate_mu_cov(nav, lookback=120)


def test_current_weights_sums_to_one():
    nav = _make_nav()
    holdings = pd.DataFrame(
        [
            {"symbol": "F0", "shares": 100.0, "name": "F0"},
            {"symbol": "F1", "shares": 200.0, "name": "F1"},
            {"symbol": "F2", "shares": 50.0, "name": "F2"},
        ]
    )
    weights, total = current_weights(holdings, nav)
    assert total > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert set(weights.keys()) == {"F0", "F1", "F2"}


def test_current_weights_handles_missing_nav():
    nav = _make_nav()
    holdings = pd.DataFrame(
        [
            {"symbol": "F0", "shares": 100.0, "name": "F0"},
            {"symbol": "MISSING", "shares": 100.0, "name": "X"},
        ]
    )
    weights, _ = current_weights(holdings, nav)
    assert weights["MISSING"] == 0.0
    assert weights["F0"] == 1.0


def test_run_all_methods_returns_three_strategies():
    nav = _make_nav()
    mu, cov, _ = estimate_mu_cov(nav, lookback=120)
    results = run_all_methods(mu, cov, risk_aversion=1.0, w_max=0.6)
    assert set(results.keys()) == {"等权", "MVO", "风险平价"}
    for w, stats in results.values():
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-9).all() and (w <= 0.6 + 1e-9).all()
        assert {"return", "volatility", "sharpe"} <= stats.keys()


def test_reconcile_action_classification():
    assets = ["A", "B", "C"]
    target_w = np.array([0.5, 0.3, 0.2])
    current_w = {"A": 0.2, "B": 0.3, "C": 0.5}  # A 加仓, B 不动, C 减仓
    total_mkt = 10000.0
    df = reconcile(assets, target_w, current_w, total_mkt)
    assert df.loc[df["代码"] == "A", "操作"].iloc[0] == "🟢 建议加仓"
    assert df.loc[df["代码"] == "B", "操作"].iloc[0] == "⚪ 持有不动"
    assert df.loc[df["代码"] == "C", "操作"].iloc[0] == "🔴 建议减仓"


def test_reconcile_uses_name_map():
    assets = ["A"]
    target_w = np.array([1.0])
    df = reconcile(assets, target_w, {"A": 1.0}, 1000.0, name_map={"A": "Alpha基金"})
    assert df.iloc[0]["基金"] == "Alpha基金"
