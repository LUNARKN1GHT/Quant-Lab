"""统计套利相关模块的测试：OU 过程、协整检验、Kalman Filter、市场中性组合"""

import numpy as np
import pandas as pd
import pytest

from quant.strategy.cointegration import (
    EGResult,
    JohansenResult,
    engle_granger,
    johansen,
    residual_diagnostics,
    rolling_hedge_ratio,
)
from quant.strategy.kalman_hedge import fit_kalman, kalman_zscore
from quant.strategy.market_neutral import (
    build_portfolio,
    compute_rolling_beta,
    sector_neutralize,
)
from quant.strategy.ou_process import OUParams, entry_exit_thresholds, fit_ou, ou_zscore
from quant.strategy.pca_basket import (
    basket_signal,
    pca_zscore,
    portfolio_return,
    rolling_pca_residuals,
)

# ── OU 过程 ─────────────────────────────────────────────────────────────────


def make_ar1_series(b: float, n: int = 300, seed: int = 42) -> pd.Series:
    """生成 AR(1) 过程：X_t = b * X_{t-1} + ε，用于验证 OU 参数估计"""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b * x[t - 1] + rng.normal(0, 0.1)
    return pd.Series(x)


def test_fit_ou_half_life():
    # AR(1) b=0.9 → theta=0.1 → half_life = ln2/0.1 ≈ 6.93
    series = make_ar1_series(b=0.9, n=1000)
    params = fit_ou(series)
    assert params.half_life == pytest.approx(np.log(2) / 0.1, rel=0.40)


def test_fit_ou_fast_mean_reversion():
    # b=0.5 → theta=0.5 → half_life ≈ 1.39，比 b=0.9 的 6.93 短很多
    slow = fit_ou(make_ar1_series(b=0.9))
    fast = fit_ou(make_ar1_series(b=0.5))
    assert fast.half_life < slow.half_life


def test_fit_ou_returns_dataclass():
    params = fit_ou(make_ar1_series(b=0.8))
    assert isinstance(params, OUParams)
    assert params.theta > 0
    assert params.sigma > 0
    assert np.isfinite(params.log_likelihood)


def test_ou_zscore_global():
    series = make_ar1_series(b=0.8)
    params = fit_ou(series)
    z = ou_zscore(series, params)
    assert z.mean() == pytest.approx(0.0, abs=0.2)
    assert z.notna().all()


def test_ou_zscore_rolling():
    series = make_ar1_series(b=0.8)
    params = fit_ou(series)
    z = ou_zscore(series, params, window=60)
    # 前 59 个为 NaN，之后有值
    assert z.iloc[:59].isna().all()
    assert z.iloc[59:].notna().any()


def test_entry_exit_thresholds_by_halflife():
    # 半衰期越短，入场阈值越低（更激进）
    fast_params = OUParams(
        theta=1.0, mu=0.0, sigma=0.1, half_life=3.0, log_likelihood=0
    )
    slow_params = OUParams(
        theta=1.0, mu=0.0, sigma=0.1, half_life=25.0, log_likelihood=0
    )
    fast_t = entry_exit_thresholds(fast_params)
    slow_t = entry_exit_thresholds(slow_params)
    assert fast_t["entry"] < slow_t["entry"]
    assert fast_t["exit"] < slow_t["exit"]


# ── 协整检验 ─────────────────────────────────────────────────────────────────


def make_cointegrated_pair(n: int = 200, seed: int = 0):
    """生成真实协整对：共同随机游走 + 小噪声"""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0, 1, n))
    a = pd.Series(trend + rng.normal(0, 0.1, n))
    b = pd.Series(trend * 1.2 + rng.normal(0, 0.1, n))
    return a, b


def make_independent_walks(n: int = 200, seed: int = 0):
    """生成两个独立随机游走（不协整）"""
    rng = np.random.default_rng(seed)
    a = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    b = pd.Series(np.cumsum(rng.normal(0, 1, n)))
    return a, b


def test_engle_granger_detects_cointegration():
    a, b = make_cointegrated_pair()
    result = engle_granger(a, b)
    assert isinstance(result, EGResult)
    assert result.is_cointegrated
    assert result.pvalue < 0.05


def test_engle_granger_returns_residual_series():
    a, b = make_cointegrated_pair()
    result = engle_granger(a, b)
    assert len(result.residual) == len(a)
    assert result.hedge_ratio > 0  # 同向走势，对冲比率为正


def test_johansen_detects_cointegration():
    a, b = make_cointegrated_pair()
    result = johansen(a, b)
    assert isinstance(result, JohansenResult)
    assert result.is_cointegrated
    assert result.n_cointegration >= 1


def test_residual_diagnostics_structure():
    a, b = make_cointegrated_pair()
    eg = engle_granger(a, b)
    diag = residual_diagnostics(eg.residual)
    assert "adf_stat" in diag
    assert "is_stationary" in diag
    assert "durbin_watson" in diag
    # 协整残差应该是平稳的
    assert diag["is_stationary"]


def test_rolling_hedge_ratio_length():
    a, b = make_cointegrated_pair(n=200)
    window = 60
    result = rolling_hedge_ratio(a, b, window=window)
    # 输出长度 = n - window + 1
    assert len(result) == len(a) - window + 1
    assert result.notna().all()


# ── Kalman Filter ─────────────────────────────────────────────────────────────


def make_constant_ratio_pair(ratio: float = 2.0, n: int = 200, seed: int = 0):
    """生成对冲比率恒为 ratio 的价格对"""
    rng = np.random.default_rng(seed)
    b = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 50)
    a = ratio * b + rng.normal(0, 0.2, n)
    return pd.Series(a), b


def test_fit_kalman_output_shape():
    a, b = make_cointegrated_pair()
    result = fit_kalman(a, b)
    assert len(result.hedge_ratio) == len(a)
    assert len(result.spread) == len(a)
    assert len(result.variance) == len(a)


def test_fit_kalman_converges_to_true_ratio():
    # 真实对冲比率为 2.0，KF 后半段估计应接近 2.0
    ratio = 2.0
    a, b = make_constant_ratio_pair(ratio=ratio, n=300)
    result = fit_kalman(a, b)
    # 取后 100 期均值
    estimated = result.hedge_ratio.iloc[-100:].mean()
    assert estimated == pytest.approx(ratio, abs=0.3)


def test_fit_kalman_spread_near_zero():
    # 当对冲比率恒定时，KF 价差应接近 0（均值回归完成）
    a, b = make_constant_ratio_pair(ratio=2.0, n=300)
    result = fit_kalman(a, b)
    # 后半段价差均值应接近 0
    assert result.spread.iloc[-100:].mean() == pytest.approx(0.0, abs=0.5)


def test_kalman_zscore_structure():
    a, b = make_cointegrated_pair()
    result = fit_kalman(a, b)
    z = kalman_zscore(result, window=20)
    # 前 19 个为 NaN，之后有值
    assert z.iloc[:19].isna().all()
    assert z.iloc[19:].notna().any()


# ── 市场中性组合 ───────────────────────────────────────────────────────────────


def make_returns(n_stocks: int = 10, n_days: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n_days, n_stocks)),
        columns=[f"S{i}" for i in range(n_stocks)],
    )


def test_compute_rolling_beta_market_is_one():
    # 股票收益 = 市场收益，Beta 应为 1
    market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100))
    stocks = pd.DataFrame({"A": market, "B": market})
    betas = compute_rolling_beta(stocks, market, window=30)
    # 稳定后 Beta 应接近 1
    tail = betas.iloc[-30:]
    assert tail["A"].mean() == pytest.approx(1.0, abs=0.05)


def test_sector_neutralize_removes_sector_mean():
    # 构造同一行业两只股票，因子值分别为 1 和 3，均值为 2
    # 去均值后应为 -1 和 1
    factor = pd.DataFrame({"A": [1.0], "B": [3.0]})
    sector_map = {"A": "tech", "B": "tech"}
    result = sector_neutralize(factor, sector_map)
    assert result.loc[0, "A"] == pytest.approx(-1.0)
    assert result.loc[0, "B"] == pytest.approx(1.0)


def test_build_portfolio_gross_exposure():
    rng = np.random.default_rng(0)
    n = 50
    scores = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
    betas = pd.Series(rng.uniform(0.5, 1.5, n), index=scores.index)
    weights = build_portfolio(scores, betas, n_long=10, n_short=10)
    # Gross exposure 应归一化为 1
    gross = sum(abs(v) for v in weights.values())
    assert gross == pytest.approx(1.0, rel=1e-6)


def test_build_portfolio_beta_neutral():
    rng = np.random.default_rng(42)
    n = 60
    scores = pd.Series(rng.normal(0, 1, n), index=[f"S{i}" for i in range(n)])
    betas = pd.Series(np.ones(n), index=scores.index)  # 所有 Beta = 1
    weights = build_portfolio(scores, betas, n_long=10, n_short=10, beta_neutral=True)
    # Beta = 1 时，多头和空头各 10 只，Beta 中性 → 净 Beta 应接近 0
    net_beta = sum(w * betas.get(s, 0) for s, w in weights.items())
    assert net_beta == pytest.approx(0.0, abs=0.1)


def test_build_portfolio_empty_when_too_few_stocks():
    scores = pd.Series({"A": 1.0, "B": 2.0})
    betas = pd.Series({"A": 1.0, "B": 1.0})
    result = build_portfolio(scores, betas, n_long=5, n_short=5)
    assert result == {}


# --- PCA 篮子套利 ----------


def make_sector_returns(
    n_stocks: int = 10, n_days: int = 120, seed: int = 0
) -> pd.DataFrame:
    """生成有共同因子结构的板块收益率：r_i = β_i * F + ε_i"""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n_days)  # 板块共同因子
    betas = rng.uniform(0.5, 1.5, n_stocks)
    idio = rng.normal(0, 0.005, (n_days, n_stocks))  # 特质噪声
    data = np.outer(common, betas) + idio
    return pd.DataFrame(data, columns=[f"S{i}" for i in range(n_stocks)])


def test_rolling_pca_residuals_shape():
    returns = make_sector_returns(n_stocks=10, n_days=120)
    window = 60
    resid = rolling_pca_residuals(returns, window=window)
    assert resid.shape == returns.shape
    # 前 window 行应全为 NaN
    assert resid.iloc[:window].isna().all().all()


def test_rolling_pca_residuals_has_values_after_window():
    returns = make_sector_returns(n_stocks=10, n_days=120)
    resid = rolling_pca_residuals(returns, window=60)
    # window 之后应有非 NaN 值
    assert resid.iloc[60:].notna().any().any()


def test_rolling_pca_residuals_smaller_than_raw():
    # 去掉共同因子后，残差的绝对值应小于原始收益率
    returns = make_sector_returns(n_stocks=10, n_days=120)
    resid = rolling_pca_residuals(returns, window=60)
    valid = resid.iloc[60:]
    raw_valid = returns.iloc[60:]
    assert valid.abs().mean().mean() < raw_valid.abs().mean().mean()


def test_pca_zscore_shape():
    returns = make_sector_returns(n_stocks=10, n_days=120)
    resid = rolling_pca_residuals(returns, window=60)
    z = pca_zscore(resid, window=20)
    assert z.shape == resid.shape


def test_pca_zscore_roughly_standardized():
    # rolling zscore 稳定后，各列 std 应接近 1
    returns = make_sector_returns(n_stocks=10, n_days=200)
    resid = rolling_pca_residuals(returns, window=60)
    z = pca_zscore(resid, window=20)
    tail = z.iloc[90:]  # 跳过预热期
    col_stds = tail.std()
    assert (col_stds.dropna() < 1.5).all()


def test_basket_signal_direction():
    # z-score 低于 -entry → 做多(+1)；高于 +entry → 做空(-1)；中间 → 0
    z = pd.DataFrame(
        {
            "A": [-2.0],  # 低估，做多
            "B": [2.0],  # 高估，做空
            "C": [0.1],  # 中性，空仓
        }
    )
    sig = basket_signal(z, entry_threshold=1.5, exit_threshold=0.5)
    assert sig.loc[0, "A"] == 1
    assert sig.loc[0, "B"] == -1
    assert sig.loc[0, "C"] == 0


def test_basket_signal_no_signal_when_moderate():
    z = pd.DataFrame({"A": [0.8], "B": [-0.6]})
    sig = basket_signal(z, entry_threshold=1.5, exit_threshold=0.5)
    assert (sig == 0).all().all()


def test_portfolio_return_long_minus_short():
    # 做多涨的股票，做空跌的股票，收益应为正
    returns = pd.DataFrame(
        {"A": [0.02, 0.01], "B": [-0.02, -0.01]},
        index=pd.date_range("2024-01-01", periods=2),
    )
    signal = pd.DataFrame(
        {"A": [1, 0], "B": [-1, 0]},
        index=returns.index,
    )
    pnl = portfolio_return(signal, returns, holding_period=1)
    # 第一天：做多 A(+2%)，做空 B(-2%) → 收益 = 0.02 - (-0.02) = 0.04
    assert pnl.iloc[0] == pytest.approx(0.04, abs=1e-6)


def test_portfolio_return_no_signal_gives_zero():
    returns = pd.DataFrame(
        {"A": [0.05], "B": [-0.05]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    signal = pd.DataFrame({"A": [0], "B": [0]}, index=returns.index)
    pnl = portfolio_return(signal, returns, holding_period=1)
    assert pnl.iloc[0] == pytest.approx(0.0)


def test_rolling_pca_residuals_too_few_stocks_skipped():
    # 只有 3 只股票（< 5），所有行应全为 NaN
    returns = make_sector_returns(n_stocks=3, n_days=80)
    resid = rolling_pca_residuals(returns, window=60)
    assert resid.isna().all().all()


def test_rolling_pca_residuals_all_nan_today_skipped():
    # 某一天所有股票都是 NaN，该行残差保持 NaN
    returns = make_sector_returns(n_stocks=10, n_days=90)
    returns.iloc[70] = np.nan  # 第 70 天全部置 NaN
    resid = rolling_pca_residuals(returns, window=60)
    assert resid.iloc[70].isna().all()


def test_portfolio_return_last_row_is_nan():
    # 最后一行信号因取不到前向收益，返回 NaN
    returns = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "B": [-0.01, -0.02, -0.03]},
        index=pd.date_range("2024-01-01", periods=3),
    )
    # 最后一天有信号，但 holding_period=1 超出边界
    signal = pd.DataFrame(
        {"A": [0, 0, 1], "B": [0, 0, -1]},
        index=returns.index,
    )
    pnl = portfolio_return(signal, returns, holding_period=1)
    assert np.isnan(pnl.iloc[-1])
