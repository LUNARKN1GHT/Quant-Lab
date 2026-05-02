"""因子有效性评估：IC（信息系数）与 ICIR（信息比率）

IC 衡量因子值与下期收益的截面相关性，使用 Spearman 秩相关（非参数，对极值更鲁棒）。
ICIR = IC均值 / IC标准差，衡量 IC 的稳定性，值越大说明因子信号越持续有效。
"""

import pandas as pd
from scipy.stats import spearmanr


def calc_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    """计算单期截面 IC（Spearman 秩相关系数）。

    Args:
        factor: 某一截面日期各股票的因子值，index 为股票代码
        forward_return: 对应的下期收益率，index 与 factor 一致

    Returns:
        IC 值，范围 [-1, 1]，绝对值越大说明因子预测能力越强
    """
    ic, _ = spearmanr(factor, forward_return)
    return ic


def calc_icir(ic_series: pd.Series) -> float:
    """计算 ICIR（IC 信息比率）= IC 均值 / IC 标准差。

    Args:
        ic_series: 时序 IC 序列，每个元素为一期截面 IC 值

    Returns:
        ICIR，|ICIR| > 0.5 通常认为因子具有较强的稳定性
    """
    return ic_series.mean() / ic_series.std()
