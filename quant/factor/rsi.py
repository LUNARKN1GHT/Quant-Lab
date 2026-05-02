"""RSI（相对强弱指数）因子

RSI 衡量一段时间内上涨幅度与总波动的比值，范围 [0, 100]。
RSI > 70 通常视为超买，RSI < 30 视为超卖，可作为均值回归信号。
"""

import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """计算 RSI（相对强弱指数）。

    公式：RSI = 100 - 100 / (1 + RS)，其中 RS = 平均涨幅 / 平均跌幅

    Args:
        close: 收盘价时间序列
        window: 计算窗口，默认 14 日（Wilder 原始参数）

    Returns:
        RSI 序列，值域 [0, 100]
    """
    delta = close.diff()
    # 分离上涨（gain）和下跌（loss）部分，clip 保证两者均为非负数
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    # RS = 平均涨幅 / 平均跌幅
    rs = avg_gain / avg_loss

    return 100 - 100 / (1 + rs)
