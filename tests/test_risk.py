import pandas as pd
import pytest

from quant.risk.metrics import calmar, cvar, max_drawdown, sharpe, sortino, var

# 公共测试数据
RETURNS = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01])
RETURNS_VAR = pd.Series(
    [-0.05, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05]
)


def test_sharpe():
    assert sharpe(RETURNS) == pytest.approx(1.9322, rel=1e-3)


def test_sortino():
    # sortino 只用下行波动率，比 sharpe 更高
    result = sortino(RETURNS)
    assert result > sharpe(RETURNS)
    assert result == pytest.approx(4.490, rel=1e-3)


def test_max_drawdown():
    # 涨 10% 再跌 10%：净值 1 → 1.1 → 0.99，回撤约 -10%
    r = pd.Series([0.1, -0.1])
    assert max_drawdown(r) == pytest.approx(-0.1, rel=1e-3)
    # 回撤必须是负数
    assert max_drawdown(RETURNS) <= 0


def test_calmar():
    # Calmar = 年化收益 / |最大回撤|，必须为正（年化收益为正时）
    r = pd.Series([0.01] * 5 + [-0.02])  # 整体正收益但有回撤
    result = calmar(r)
    assert isinstance(result, float)


def test_var():
    # 5% 分位数：10 个数据点中最小的那个附近
    assert var(RETURNS_VAR) == pytest.approx(-0.0455, rel=1e-3)
    # VaR 必须为负（表示损失）
    assert var(RETURNS_VAR) < 0


def test_cvar():
    # CVaR 是比 VaR 更差的均值，应该小于等于 VaR
    assert cvar(RETURNS_VAR) <= var(RETURNS_VAR)
    assert cvar(RETURNS_VAR) == pytest.approx(-0.05, rel=1e-3)
