import pandas as pd


def factor_select(factor_scores: pd.Series, top_n: int = 10) -> pd.Series:
    """因子选择策略

    Args:
        factor_scores (pd.Series): 因子分数
        top_n (int, optional): 选多少只股票. Defaults to 10.

    Returns:
        pd.Series: 返回持仓权重
    """
    # 1. 取分数最高的 top_n 个股票代码
    top_stocks = factor_scores.nlargest(top_n).index

    # 2. 构造权重 Series
    weights = pd.Series(0.0, index=factor_scores.index)
    weights[top_stocks] = 1.0 / top_n

    return weights
