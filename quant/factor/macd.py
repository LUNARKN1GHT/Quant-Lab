import pandas as pd


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line: pd.Series = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return macd_line - signal_line
