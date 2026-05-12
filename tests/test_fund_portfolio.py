from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quant.fund.portfolio import (
    fund_returns,
    load_nav_matrix,
    portfolio_value,
    risk_metrics,
    style_attribution,
)

# --- 测试数据工厂 ----------


def _make_nav(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, name="date")
    return pd.DataFrame(
        {
            "F1": np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
            "F2": np.cumprod(1 + rng.normal(0.0003, 0.008, n)),
        },
        index=idx,
    )


# --- load_nav_matrix ---------


def test_load_nav_matrix_shape():
    mock_con = MagicMock()
    mock_con.execute.return_value.df.return_value = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]
            ),
            "symbol": ["F1", "F1", "F2", "F2"],
            "nav": [1.0, 1.01, 2.0, 2.02],
        }
    )
    result = load_nav_matrix(mock_con, ["F1", "F2"])
    assert result.shape == (2, 2)
    assert set(result.columns) == {"F1", "F2"}


# --- fund_returns ----------


def test_fund_returns_shape():
    nav = _make_nav()
    ret = fund_returns(nav)
    assert ret.shape[1] == nav.shape[1]
    assert len(ret) < len(nav)


def test_fund_returns_no_nan():
    nav = _make_nav()
    ret = fund_returns(nav)
    assert not ret.isnull().values.any()


# --- risk_metrics ----------


def test_risk_metrics_keys():
    nav = _make_nav()
    r = fund_returns(nav)["F1"]
    m = risk_metrics(r)
    for k in [
        "年化收益",
        "年化波动",
        "Sharpe",
        "Sortino",
        "最大回撤",
        "Calmar",
        "近1月",
        "近3月",
        "近1年",
    ]:
        assert k in m


def test_risk_metrics_max_drawdown_nonpositive():
    nav = _make_nav()
    r = fund_returns(nav)["F1"]
    m = risk_metrics(r)
    assert m["最大回撤"] <= 0


def test_risk_metrics_flat_returns():
    """收益恒为零时 Sharpe/Sortino 应为 0，不抛异常"""
    r = pd.Series([0.0] * 100)
    m = risk_metrics(r)
    assert m["Sharpe"] == 0.0
    assert m["Sortino"] == 0.0


def test_risk_metrics_annual_vol():
    """年化波动 = 日波动 × sqrt(252)"""
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 500))
    m = risk_metrics(r)
    assert m["年化波动"] == pytest.approx(r.std() * np.sqrt(252), rel=1e-6)


# --- portfolio_value -----------


def test_portfolio_value_columns():
    nav = _make_nav()
    holdings = [{"symbol": "F1", "name": "基金A", "shares": 1000, "cost_nav": 1.0}]
    result = portfolio_value(nav, holdings)
    for col in [
        "symbol",
        "shares",
        "cost_nav",
        "latest_nav",
        "cost_value",
        "latest_value",
        "pnl",
        "pnl_pct",
    ]:
        assert col in result.columns


def test_portfolio_value_skips_missing_symbol():
    nav = _make_nav()
    holdings = [{"symbol": "F9", "name": "不存在", "shares": 1000, "cost_nav": 1.0}]
    result = portfolio_value(nav, holdings)
    assert result.empty


def test_portfolio_value_pnl_calculation():
    nav = _make_nav()
    latest_f1 = nav["F1"].iloc[-1]
    cost = 1.0
    shares = 500
    holdings = [{"symbol": "F1", "shares": shares, "cost_nav": cost}]
    result = portfolio_value(nav, holdings)
    assert result.iloc[0]["pnl"] == pytest.approx(shares * (latest_f1 - cost))
    assert result.iloc[0]["pnl_pct"] == pytest.approx(latest_f1 / cost - 1)


# --- style_attribution ----------


def test_style_attribution_returns_dict():
    nav = _make_nav(300)
    r = fund_returns(nav)["F1"]
    bench = fund_returns(nav).rename(columns={"F1": "CSI300", "F2": "Bond"})
    result = style_attribution(r, bench)
    assert isinstance(result, dict)
    assert "Alpha（日）" in result
    assert "R²" in result


def test_style_attribution_r2_in_range():
    nav = _make_nav(300)
    r = fund_returns(nav)["F1"]
    bench = fund_returns(nav).rename(columns={"F1": "CSI300", "F2": "Bond"})
    result = style_attribution(r, bench)
    assert 0 <= result["R²"] <= 1


def test_style_attribution_insufficient_data():
    """少于 60 个公共日期时返回空字典"""
    idx = pd.date_range("2024-01-01", periods=30, name="date")
    r = pd.Series(np.random.normal(0, 0.01, 30), index=idx)
    bench = pd.DataFrame({"B": np.random.normal(0, 0.01, 30)}, index=idx)
    result = style_attribution(r, bench)
    assert result == {}
