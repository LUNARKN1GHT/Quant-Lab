"""配对交易策略

基于协整理论：若两只股票长期存在均衡关系（协整），其价差会均值回归。
当价差偏离均值过大时做多低估股、做空高估股，等待价差回归获利。
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def check_cointegration(
    price_a: pd.Series, price_b: pd.Series, pvalue_threshold: float = 0.05
) -> bool:
    """用 Engle-Granger 协整检验判断两只股票是否适合配对交易。

    Args:
        price_a: 股票 A 的价格序列
        price_b: 股票 B 的价格序列
        pvalue_threshold: 显著性水平，默认 0.05

    Returns:
        True 表示两股票协整（适合配对），False 表示不协整
    """
    _, pvalue, _ = coint(price_a, price_b)
    return pvalue < pvalue_threshold


def hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """用 OLS 回归估计对冲比例：price_a = hedge_ratio * price_b + intercept。

    对冲比例决定做多/做空的数量比，使组合价差平稳。
    """
    return np.polyfit(price_b, price_a, 1)[0]


def spread(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """计算两只股票的标准化价差（z-score）。

    标准化后价差均值为 0、标准差为 1，方便用固定阈值生成交易信号。
    """
    ratio = hedge_ratio(price_a, price_b)
    raw_spread = price_a - ratio * price_b
    # 标准化：均值为 0，标准差为 1，方便后续用 z-score 生成信号
    return (raw_spread - raw_spread.mean()) / raw_spread.std()


def generate_signal(spread: pd.Series) -> pd.Series:
    """根据价差 z-score 生成交易信号。

    入场阈值 ±2σ（价差偏离均值 2 个标准差时开仓），
    平仓阈值 ±0.5σ（价差回归至接近均值时平仓）。

    Returns:
        信号序列：1=做多价差（A 低 B 高），-1=做空价差，0=空仓
    """
    signal = pd.Series(0, index=spread.index)
    signal[spread > 2] = -1   # 价差过高：做空 A、做多 B
    signal[spread < -2] = 1   # 价差过低：做多 A、做空 B
    signal[spread.abs() < 0.5] = 0  # 价差回归均值附近，平仓
    return signal
