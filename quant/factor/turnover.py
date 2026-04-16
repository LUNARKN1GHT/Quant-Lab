import pandas as pd


def turnover(volume: pd.Series, window: int):
    """计算换手率"""
    return volume / volume.rolling(window=window).mean()
