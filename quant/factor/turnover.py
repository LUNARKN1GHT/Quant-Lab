"""相对换手率因子

用当日成交量除以 N 日均量，衡量当日交投活跃程度相对于历史均值的倍数。
>1 表示今日换手高于均值（放量），<1 表示缩量。
常用于捕捉资金异动或流动性变化信号。
"""

import pandas as pd


def turnover(volume: pd.Series, window: int) -> pd.Series:
    """计算相对换手率 = 当日成交量 / N 日均量。

    Args:
        volume: 成交量时间序列
        window: 均量计算窗口（交易日）

    Returns:
        相对换手率序列
    """
    return volume / volume.rolling(window=window).mean()
