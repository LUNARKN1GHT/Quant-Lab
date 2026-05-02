"""动量因子：衡量过去 N 天的累积收益率

动量效应（Momentum）：过去一段时间内涨势好的股票，未来短期倾向于继续上涨。
公式：momentum = close_t / close_{t-N} - 1，即 N 日区间收益率。
"""

import pandas as pd


def momentum(close: pd.Series, window: int) -> pd.Series:
    """计算 N 日动量因子（区间累积收益率）。

    Args:
        close: 收盘价时间序列
        window: 回看窗口（交易日数），常用值：20（月）、60（季）、120（半年）

    Returns:
        动量因子值序列，前 window 个值为 NaN
    """
    # shift(window) 将收盘价向后移动 N 期，实现向量化的区间收益计算
    return close / close.shift(window) - 1
