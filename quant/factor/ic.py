import pandas as pd
from scipy.stats import spearmanr


def calc_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """计算单期 IC"""
    ic, _ = spearmanr(factor, forward_return)
    return ic


def calc_icir(ic_series: pd.Series) -> float:
    """计算 ICIR"""
    return ic_series.mean() / ic_series.std()
