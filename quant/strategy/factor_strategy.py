"""因子选股策略：基于截面因子值选出 Top-N 股票等权持仓"""

import pandas as pd


def factor_select(factor_scores: pd.Series, top_n: int = 10) -> pd.Series:
    """单截面因子选股，返回等权持仓权重。

    Args:
        factor_scores: 某日截面各股票的因子值，index 为股票代码
        top_n: 选取因子值最高的 N 只股票，默认 10

    Returns:
        持仓权重序列，被选中的股票权重为 1/top_n，其余为 0
    """
    # 取因子值最大的 top_n 只股票（因子值越大，预期收益越高）
    top_stocks = factor_scores.nlargest(top_n).index

    # 初始化全零权重，再对选中股票赋予等权
    weights = pd.Series(0.0, index=factor_scores.index)
    weights[top_stocks] = 1.0 / top_n

    return weights
