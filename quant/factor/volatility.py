import pandas as pd


def volatility(close: pd.Series, window: int):
    """计算波动率因子"""
    # 计算单日收益率
    daily_return = close / close.shift(1) - 1
    volatility = daily_return.rolling(window=window).std()

    return volatility
