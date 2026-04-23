import pandas as pd


def idiosyncratic_vol(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 20,
) -> pd.Series:
    r = close.pct_change()
    m = market_close.pct_change()

    roll_cov = r.rolling(window).cov(m)
    roll_var = m.rolling(window).var()
    beta = roll_cov / roll_var

    residuals = r - beta * m
    return residuals.rolling(window).std()
