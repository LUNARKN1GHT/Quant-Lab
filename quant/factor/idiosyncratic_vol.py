import pandas as pd


def idiosyncratic_vol(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 20,
) -> pd.Series:
    stock_returns = close.pct_change()
    market_returns = market_close.pct_change()

    def rolling_residual_std(i):
        if i < window:
            return float("nan")
        r = stock_returns.iloc[i - window : i]
        m = market_returns.iloc[i - window : i]
        beta = r.cov(m) / m.var()
        residuals = r - beta * m
        return residuals.std()

    return pd.Series(
        [rolling_residual_std(i) for i in range(len(close))], index=close.index
    )
