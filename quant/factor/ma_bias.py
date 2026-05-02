"""均线偏离率（MA Bias）因子

衡量当前价格偏离 N 日均线的程度。
正值表示价格在均线上方（超买倾向），负值表示在均线下方（超卖倾向）。
常用于均值回归策略：偏离过大时预期向均线靠拢。
"""

import pandas as pd


def ma_bias(close: pd.Series, window: int = 20) -> pd.Series:
    """计算均线偏离率 = (收盘价 - N日均线) / N日均线。

    Args:
        close: 收盘价时间序列
        window: 均线窗口，默认 20 日

    Returns:
        偏离率序列，正值超买、负值超卖
    """
    ma = close.rolling(window=window).mean()
    return (close - ma) / ma
