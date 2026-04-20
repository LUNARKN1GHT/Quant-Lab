from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    """市场环境标记类"""

    BULL = "BULL"
    """牛市"""

    RANGE = "RANGE"
    """普通市场环境"""

    BEAR = "BEAR"
    """熊市"""


def detect_regime(
    close: pd.DataFrame,
    ma_window: int = 120,
    breadth_window: int = 20,
    vol_short: int = 20,
    vol_long: int = 60,
) -> pd.Series:
    """判断当前市场环境

    Args:
        close (pd.DataFrame): 收盘价列表
        ma_window (int, optional): 均线窗口. Defaults to 120.
        breadth_window (int, optional): 市场宽度. Defaults to 20.
        vol_short (int, optional): 短期波动率窗口长度. Defaults to 20.
        vol_long (int, optional): 长期波动率窗口长度. Defaults to 60.

    Returns:
        pd.Series: _description_
    """

    # 1. 趋势信号
    # 等权指数 = 所有成分股收盘价的横截面均值
    index_close = close.mean(axis=1)

    # 趋势 = 当前价格相对 X 日均线的偏离
    trend = index_close / index_close.rolling(ma_window).mean() - 1

    # 2. 市场宽度信号（上涨股票占比）
    # 每日有多少比例的股票 X 日为正
    breadth = (close.pct_change(breadth_window) > 0).mean(axis=1)

    # 3. 波动率信号（恐慌检测）
    returns = close.pct_change().mean(axis=1)
    vol_20 = returns.rolling(vol_short).std()
    vol_60 = returns.rolling(vol_long).std()
    vol_ratio = vol_20 / vol_60

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
