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


def calc_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    min_stocks: int = 10,
) -> pd.Series:
    """逐日计算截面 IC，返回时序 IC 序列

    Args:
        factor (pd.DataFrame): 因子值宽表
        forward_return (pd.DataFrame): 远期收益宽表
        min_stocks (int, optional): 单日至少需要的有效股票数. Defaults to 10.

    Returns:
        pd.Series: 时序 IC 序列，index 为日期
    """
    ic_values: dict = {}
    common_dates = factor.index.intersection(forward_return.index)
    for date in common_dates:
        f_row = factor.loc[date].dropna()
        r_row = forward_return.loc[date].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < min_stocks:
            continue
        ic, _ = spearmanr(f_row[common], r_row[common])
        if pd.notna(ic):  # type: ignore
            ic_values[date] = ic
    return pd.Series(ic_values).sort_index()


def calc_ic_decay(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """计算因子在多个前瞻窗口下的 IC 衰减曲线

    Args:
        factor: 因子值宽表
        close: 收盘价宽表（与 factor 同 symbol 集）
        horizons: 前瞻窗口列表，默认 [1, 5, 10, 20, 40, 60]

    Returns:
        index=horizon, columns=[ic_mean, ic_std, icir, n_periods]
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 40, 60]

    rows = []
    for h in horizons:
        fwd_ret = close.pct_change(h).shift(-h)
        ic_series = calc_ic_series(factor, fwd_ret)
        rows.append(
            {
                "horizon": h,
                "ic_mean": ic_series.mean(),
                "ic_std": ic_series.std(),
                "icir": calc_icir(ic_series) if len(ic_series) > 1 else float("nan"),
                "n_periods": len(ic_series),
            }
        )
    return pd.DataFrame(rows).set_index("horizon")
