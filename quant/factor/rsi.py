import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    return 100 - 100 / (1 + rs)
