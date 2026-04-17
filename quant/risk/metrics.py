from math import sqrt

import numpy as np
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


def drawdown_series(returns: pd.Series) -> pd.Series:
    """计算回撤序列"""
    cum = (1 + returns).cumprod()
    history_highest = cum.cummax()
    return (cum - history_highest) / history_highest


def underwater_stats(returns: pd.Series) -> dict:
    """统计水下时间"""
    dd = drawdown_series(returns)
    is_underwater = dd < 0

    # 计算连续水下天数
    streak = 0
    max_streak = 0
    for underwater in is_underwater:
        if underwater:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    return {"max_underwater_days": max_streak, "avg_drawdown": dd[is_underwater].mean()}


def calmar(returns: pd.Series) -> float:
    """Calmar Ratio"""
    annual_return_rate = returns.mean() * 252

    return annual_return_rate / abs(max_drawdown(returns))


def var(returns: pd.Series) -> float:
    """Value at Risk"""
    return returns.quantile(0.05)


def cvar(returns: pd.Series) -> float:
    return returns[returns <= var(returns)].mean()


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """计算 Beta"""
    return np.polyfit(market_returns, returns, 1)[0]


def alpha(
    returns: pd.Series, market_returns: pd.Series, risk_free: float = 0.0
) -> float:
    """计算年化 alpha"""
    return np.polyfit(market_returns, returns - risk_free, 1)[1]
