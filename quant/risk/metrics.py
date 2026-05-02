"""策略风险指标库

提供常用的绩效与风险度量函数，输入均为日度收益率序列。
年化系数统一使用 252 个交易日。
"""

from math import sqrt

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """计算年化 Sharpe Ratio = 超额收益均值 / 超额收益标准差 * sqrt(252)。

    risk_free 通常传入日度无风险利率（如年化 3% → 0.03/252）。
    """
    excess_returns = returns - risk_free
    return excess_returns.mean() / excess_returns.std() * sqrt(252)


def sortino(returns: pd.Series, risk_free: float = 0.0) -> float:
    """计算年化 Sortino Ratio，下行标准差只统计亏损日的波动。

    与 Sharpe 的区别：分母只用负收益的标准差，对上行波动不惩罚。
    """
    excess_returns = returns - risk_free
    downside = excess_returns[excess_returns < 0]
    return excess_returns.mean() / downside.std() * sqrt(252)


def max_drawdown(returns: pd.Series) -> float:
    """计算最大回撤（负数，绝对值越大越差）。

    最大回撤 = min((累计净值 - 历史最高净值) / 历史最高净值)
    """
    cum = (1 + returns).cumprod()
    history_highest = cum.cummax()
    drawdown = (cum - history_highest) / history_highest
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """计算逐日回撤序列（用于绘制水下曲线）"""
    cum = (1 + returns).cumprod()
    history_highest = cum.cummax()
    return (cum - history_highest) / history_highest


def underwater_stats(returns: pd.Series) -> dict:
    """统计水下（处于回撤中）的连续天数与平均回撤深度"""
    dd = drawdown_series(returns)
    is_underwater = dd < 0

    # 遍历逐日水下状态，统计最长连续水下天数
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
    """计算 Calmar Ratio = 年化收益率 / 最大回撤绝对值。

    值越高说明单位回撤换取的收益越多，适合评估趋势跟踪策略。
    """
    annual_return_rate = returns.mean() * 252
    return annual_return_rate / abs(max_drawdown(returns))


def var(returns: pd.Series) -> float:
    """计算历史 VaR（95% 置信水平）：最差 5% 情形的分位数收益。

    返回负数，绝对值为单日最大亏损的历史估计值。
    """
    return returns.quantile(0.05)


def cvar(returns: pd.Series) -> float:
    """计算 CVaR（条件风险价值，又称 Expected Shortfall）。

    CVaR = VaR 以下所有亏损日的平均收益，比 VaR 更好地刻画尾部风险。
    """
    return returns[returns <= var(returns)].mean()


def beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """用 OLS 线性回归估计市场 Beta（returns 对 market_returns 的斜率）"""
    return np.polyfit(market_returns, returns, 1)[0]


def alpha(
    returns: pd.Series, market_returns: pd.Series, risk_free: float = 0.0
) -> float:
    """用 OLS 估计 Jensen's Alpha（超额收益对市场收益回归的截距，未年化）"""
    return np.polyfit(market_returns, returns - risk_free, 1)[1]
