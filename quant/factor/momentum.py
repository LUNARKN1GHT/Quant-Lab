"""动量因子：衡量过去 N 天的累积收益率"""

import pandas as pd


def momentum(close: pd.Series, window: int) -> pd.Series:
    """计算动量因子"""
    # 使用向量化计算
    return close / close.shift(window) - 1
