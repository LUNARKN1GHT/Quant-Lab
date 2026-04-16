import pandas as pd


def backtest(
    position: pd.DataFrame, returns: pd.DataFrame, shift: bool = True
) -> pd.Series:
    """单个因子因子收益率回测函数"""
    actual_returns = returns.shift(-1) if shift else returns
    return (position * actual_returns).sum(axis=1)
