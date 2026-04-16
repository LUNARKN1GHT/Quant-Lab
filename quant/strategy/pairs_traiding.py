import pandas as pd
from statsmodels.tsa.stattools import coint


def check_cointegration(
    price_a: pd.Series, price_b: pd.Series, pvalue_threshold: float = 0.05
) -> bool:
    """检验两只股票是否协整，返回是否通过检验"""
    _, pvalue, _ = coint(price_a, price_b)
    return pvalue < pvalue_threshold
