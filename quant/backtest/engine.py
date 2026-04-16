from typing import Literal

import pandas as pd


def backtest(
    position: pd.DataFrame,
    returns: pd.DataFrame,
    shift: bool = True,
    commission_rate: float = 0.0003,
) -> pd.Series:
    """单个因子因子收益率回测函数"""
    # 计算换手量
    turnover: pd.DataFrame = (position - position.shift(1)).abs()
    commission = turnover.sum(axis=1) * commission_rate

    actual_returns = returns.shift(-1) if shift else returns
    return (position * actual_returns).sum(axis=1) - commission


def rebalance(
    position: pd.DataFrame, freq: Literal["D", "W", "ME"] = "D"
) -> pd.DataFrame:
    """按频率对持仓进行再平衡，非调仓日调仓保持不变"""
    return position.resample(freq).last().reindex(position.index).ffill()
