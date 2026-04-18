import pandas as pd


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    upper = ma + 2 * std
    lower = ma - 2 * std

    return (close - lower) / (upper - lower)
