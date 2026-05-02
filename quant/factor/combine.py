"""多因子合成：将多个单因子加权合并为一个综合因子

两种合成方式：
- 等权：简单平均，无需历史数据，适合初步验证
- IC 加权：按各因子的历史 IC 分配权重，IC 越大的因子贡献越多
"""

import pandas as pd


def equal_weight(factors: pd.DataFrame) -> pd.Series:
    """等权合成：每个因子权重相同，取行均值。

    Args:
        factors: 每列为一个因子，行为同一时间截面的各股票

    Returns:
        合成因子值（各因子等权平均）
    """
    return factors.mean(axis=1)


def ic_weight(factors: pd.DataFrame, ic_scores: pd.Series) -> pd.Series:
    """IC 加权合成：按各因子历史 IC 绝对值分配权重。

    Args:
        factors: 每列为一个因子，行为同一时间截面的各股票
        ic_scores: 每个因子对应的 IC（或 ICIR），index 与 factors 列名一致

    Returns:
        IC 加权后的合成因子值
    """
    # 归一化权重，使所有因子权重之和为 1
    weights = ic_scores / ic_scores.sum()
    return factors.mul(weights, axis=1).sum(axis=1)
