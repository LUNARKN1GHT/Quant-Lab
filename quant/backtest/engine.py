"""向量化回测引擎

基于持仓矩阵和收益率矩阵做逐日向量化计算，避免逐行循环，适合因子研究阶段的快速验证。
不模拟滑点和涨跌停，适合多股票截面策略，不适合高频或大规模资金的精细模拟。
"""

from typing import Literal

import pandas as pd


def backtest(
    position: pd.DataFrame,
    returns: pd.DataFrame,
    shift: bool = True,
    commission_rate: float = 0.0003,
) -> pd.Series:
    """向量化单因子回测，返回策略日度收益率序列。

    Args:
        position: 持仓权重矩阵，shape=(日期, 股票)，每行权重之和通常为 1
        returns: 日度收益率矩阵，shape 与 position 一致
        shift: 是否将收益率向前移一期（True 表示用今日仓位赚明日收益，消除信号领先偏差）
        commission_rate: 单边手续费率，默认 0.03%（万三）

    Returns:
        策略日度收益率序列
    """
    # 相邻两日持仓变化量的绝对值 = 换手量，乘以费率得到当日摩擦成本
    turnover: pd.DataFrame = (position - position.shift(1)).abs()
    commission = turnover.sum(axis=1) * commission_rate

    # shift=-1 将收益率向前移一期，确保用 t 日仓位对应 t+1 日收益
    actual_returns = returns.shift(-1) if shift else returns
    return (position * actual_returns).sum(axis=1) - commission


def rebalance(
    position: pd.DataFrame, freq: Literal["D", "W", "ME"] = "D"
) -> pd.DataFrame:
    """按指定频率再平衡持仓，非调仓日持仓保持不变（前向填充）。

    Args:
        position: 原始日度持仓矩阵
        freq: 调仓频率，"D"=每日，"W"=每周，"ME"=每月末

    Returns:
        再平衡后的日度持仓矩阵（非调仓日权重不变）
    """
    # resample().last() 取每个周期最后一个调仓信号，ffill 填充中间日期
    return position.resample(freq).last().reindex(position.index).ffill()
