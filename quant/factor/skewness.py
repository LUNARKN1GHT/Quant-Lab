import pandas as pd


def skewness(close: pd.Series, window: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window=window).skew()


def kurtosis(close: pd.Series, window: int = 20) -> pd.Series:
    returns = close.pct_change()
    return returns.rolling(window=window).kurt()
