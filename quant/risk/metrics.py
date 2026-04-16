from math import sqrt

import pandas as pd


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """年化 Sharpe Ratio"""
    excess_returns = returns - risk_free
    return excess_returns.mean() / excess_returns.std() * sqrt(252)


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """年化 Sortino Ratio"""
    excess_returns = returns - risk_free
    downside = excess_returns[excess_returns < 0]
    return excess_returns.mean() / downside.std() * sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """最大回撤"""
    cum = (1 + returns).cumprod()
    history_highest = cum.cummax()
    drawdown = (cum - history_highest) / history_highest
    return drawdown.min()


def calmar(returns: pd.Series) -> float:
    """Calmar Ratio"""
    annual_return_rate = returns.mean() * 252

    return annual_return_rate / abs(max_drawdown(returns))


def var(returns: pd.Series) -> float:
    """Value at Risk"""
    return returns.quantile(0.05)


def cvar(returns: pd.Series) -> float:
    return returns[returns <= var(returns)].mean()
