"""MACD 柱状图因子（MACD Histogram）

MACD = 快线（短期 EMA）- 慢线（长期 EMA），衡量短期趋势与长期趋势的偏离。
本函数返回 MACD 柱状图 = MACD 线 - 信号线，正值代表短期动量增强，负值代表减弱。
"""

import pandas as pd


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """计算 MACD 柱状图（DIF - DEA）。

    Args:
        close: 收盘价时间序列
        fast: 快线 EMA 窗口，默认 12
        slow: 慢线 EMA 窗口，默认 26
        signal: 信号线 EMA 窗口，默认 9

    Returns:
        MACD 柱状图序列（正值看涨，负值看跌）
    """
    # adjust=False 使用递归方式计算 EMA，与 Wind/同花顺等金融软件结果一致
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line: pd.Series = ema_fast - ema_slow          # DIF
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()  # DEA

    # 柱状图 = DIF - DEA，反映动量加速/减速
    return macd_line - signal_line
