"""收益率分布形态因子：偏度（Skewness）与峰度（Kurtosis）

偏度衡量收益率分布的不对称性：
- 正偏（右尾长）：小概率大涨，投资者往往高估其价值，未来收益偏低（彩票效应）
- 负偏（左尾长）：小概率大跌，风险更高

峰度衡量分布尾部的厚薄（相对于正态分布）：
- 高峰度（>3）说明极端收益出现频率高于正态分布，尾部风险更大
"""

import pandas as pd


def skewness(close: pd.Series, window: int = 20) -> pd.Series:
    """计算 N 日滚动偏度。

    Args:
        close: 收盘价时间序列
        window: 滚动窗口，默认 20 日

    Returns:
        偏度序列，正值右偏（彩票型），负值左偏（风险型）
    """
    returns = close.pct_change()
    return returns.rolling(window=window).skew()


def kurtosis(close: pd.Series, window: int = 20) -> pd.Series:
    """计算 N 日滚动超额峰度（excess kurtosis，正态分布为 0）。

    Args:
        close: 收盘价时间序列
        window: 滚动窗口，默认 20 日

    Returns:
        超额峰度序列，>0 代表尾部比正态分布更厚
    """
    returns = close.pct_change()
    return returns.rolling(window=window).kurt()
