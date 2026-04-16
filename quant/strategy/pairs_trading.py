import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def check_cointegration(
    price_a: pd.Series, price_b: pd.Series, pvalue_threshold: float = 0.05
) -> bool:
    """检验两只股票是否协整，返回是否通过检验"""
    _, pvalue, _ = coint(price_a, price_b)
    return pvalue < pvalue_threshold


def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """用 OLS 回归求对冲比例：price_a = hedge_ratio * price_b + intercept"""
    return np.polyfit(price_b, price_a, 1)[0]


def spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """计算标准化价差：price_a - hedge_ratio * price_b"""
    ratio = hedge_ratio(price_a, price_b)
    raw_spread = price_a - ratio * price_b
    # 标准化：均值为 0，标准差为 1，方便后续用 z-score 生成信号
    return (raw_spread - raw_spread.mean()) / raw_spread.std()


def generate_signal(spread: pd.Series) -> pd.Series:
    """根据价差生成信号"""

    # 初始化信号
    signal = pd.Series(0, index=spread.index)
    signal[spread > 2] = -1
    signal[spread < -2] = 1
    signal[spread.abs() < 0.5] = 0
    return signal
