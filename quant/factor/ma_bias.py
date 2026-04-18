import pandas as pd


def ma_bias(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window=window).mean()
    return (close - ma) / ma
