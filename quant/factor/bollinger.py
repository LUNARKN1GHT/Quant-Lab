"""布林带位置因子（Bollinger Band Position）

将当前价格在布林带内的相对位置映射到 [0, 1]：
- 0 表示价格触及下轨（超卖区域）
- 1 表示价格触及上轨（超买区域）
- 0.5 表示价格位于均线
"""

import pandas as pd


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """计算价格在布林带内的相对位置。

    公式：(close - lower) / (upper - lower)
    上下轨均为均线 ± 2 倍标准差。

    Args:
        close: 收盘价时间序列
        window: 计算窗口，默认 20 日

    Returns:
        位置因子序列，值域理论上在 [0, 1]，极端行情可能超出
    """
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return (close - lower) / (upper - lower)
