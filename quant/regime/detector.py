"""市场环境（Regime）检测

综合三个信号判断当前市场处于牛市/震荡/熊市：
1. 趋势信号：等权指数相对长期均线的偏离（方向）
2. 市场宽度：N 日内上涨股票占比（多数股上涨才算真正的牛市）
3. 波动率比值：短期 vs 长期波动率（比值过高代表市场恐慌）
"""

from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    """市场环境枚举，同时继承 str 使其可直接用作 DataFrame 列值"""

    BULL = "BULL"   # 牛市：趋势向上且多数股上涨
    RANGE = "RANGE" # 震荡：不满足牛市或熊市条件
    BEAR = "BEAR"   # 熊市：趋势向下且市场宽度收窄或波动率激增


def detect_regime(
    close: pd.DataFrame,
    ma_window: int = 120,
    breadth_window: int = 20,
    vol_short: int = 20,
    vol_long: int = 60,
) -> pd.Series:
    """逐日判断市场环境，返回 Regime 标签序列。

    Args:
        close: 成分股收盘价宽表（行=日期，列=股票代码）
        ma_window: 趋势判断用的长期均线窗口，默认 120 日（半年线）
        breadth_window: 市场宽度计算窗口，默认 20 日
        vol_short: 短期波动率窗口，默认 20 日
        vol_long: 长期波动率窗口，默认 60 日

    Returns:
        Regime 标签序列，index 与 close.index 一致
    """
    # 1. 趋势信号：等权指数（所有成分股收盘价横截面均值）相对长期均线的偏离
    index_close = close.mean(axis=1)
    trend = index_close / index_close.rolling(ma_window).mean() - 1

    # 2. 市场宽度：N 日内上涨（区间涨幅>0）的股票占比
    # 宽度 > 0.55 说明多数股票参与上涨，信号更可信
    breadth = (close.pct_change(breadth_window) > 0).mean(axis=1)

    # 3. 波动率比值：短期/长期 > 1.5 代表近期恐慌情绪明显高于历史水平
    returns = close.pct_change().mean(axis=1)
    vol_short_s = returns.rolling(vol_short).std()
    vol_long_s = returns.rolling(vol_long).std()
    vol_ratio = vol_short_s / vol_long_s

    # 用 np.select 逐条件判断（优先级从上到下，默认 RANGE）
    conditions = [
        (trend > 0) & (breadth > 0.55),
        (trend < 0) & ((breadth < 0.45) | (vol_ratio > 1.5)),
    ]
    choices = [Regime.BULL.value, Regime.BEAR.value]

    regime = pd.Series(
        np.select(conditions, choices, default=Regime.RANGE.value),
        index=close.index,
    )

    return regime
