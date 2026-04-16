import pandas as pd


def layered_return(
    factor: pd.Series, forward_return: pd.Series, n_groups: int = 5
) -> pd.Series:
    """按因子值分层，返回每层的平均收益

    Args:
        factor: 截面因子值，index 为股票代码
        forward_return: 对应的下期收益，index 与 factor 一致
        n_groups: 分层数量，默认 5 分位

    Returns:
        每层的平均收益，index 为 1~n_groups（1 = 因子值最小组）
    """
    # 按因子值分成 n_groups 组，labels=False 返回 0~n_groups-1 的整数
    groups = pd.qcut(factor, q=n_groups, labels=False)

    # 把分组和下期收益合在一起，按组计算平均收益
    result = forward_return.groupby(groups).mean()

    # 把 index 从 0-based 改成 1-based，更直观
    result.index = result.index + 1

    return result
