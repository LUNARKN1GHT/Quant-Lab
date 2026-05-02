"""宏观指标分析：滞后相关性计算与合成景气指数

宏观指标（PMI、M2、国债收益率、CPI）对大盘有一定的领先/滞后预测关系，
通过计算不同滞后期的相关性可以找到最优领先窗口。
合成景气指数将多个指标标准化后等权合并，形成单一的宏观信号。
"""

import pandas as pd


def calc_lag_corr(
    macro: pd.Series,
    market_ret: pd.Series,
    max_lag: int = 6,
) -> pd.Series:
    """计算宏观指标在 0~max_lag 个月滞后下与大盘月收益的相关性。

    Args:
        macro: 月频宏观指标序列（如 PMI）
        market_ret: 日频大盘收益率序列（函数内部会转换为月频）
        max_lag: 最大滞后月数，默认 6 个月

    Returns:
        各滞后期的相关系数序列，index 为滞后月数（0~max_lag）
    """
    # 将日频收益率转换为月频（累乘得到月度复合收益）
    market_monthly = (1 + market_ret).resample("ME").prod() - 1
    result = {}
    for lag in range(max_lag + 1):
        # shift(lag) 将宏观指标向前移 lag 个月，模拟宏观领先于大盘的效果
        shifted = macro.shift(lag).reindex(market_monthly.index, method="ffill")
        common = shifted.dropna().index.intersection(market_monthly.dropna().index)
        # 少于 12 个月的重叠数据相关性不可靠，置为 NaN
        if len(common) > 12:
            result[lag] = shifted[common].corr(market_monthly[common])
        else:
            result[lag] = float("nan")

    return pd.Series(result, name=macro.name)


def composite_index(macro_df: pd.DataFrame) -> pd.Series:
    """将多个宏观指标合成单一景气度指数。

    各指标先做 z-score 标准化，再乘以方向系数（经济扩张时 +1 / 紧缩时 -1），
    最后等权平均，使景气指数在经济好时偏高、差时偏低。

    directions 中 bond_yield=-1：利率上升通常对股市负面，因此取反。
    """
    directions = {"pmi": 1, "bond_yield": -1, "m2_yoy": 1, "cpi_yoy": 1}
    normalized = {}
    for col in macro_df.columns:
        s = macro_df[col].dropna()
        z = (s - s.mean()) / s.std()  # z-score 标准化，消除量纲差异
        normalized[col] = z * directions.get(col, 1)
    aligned = pd.DataFrame(normalized).ffill()  # 月频数据前向填充到每日

    return aligned.mean(axis=1).rename("macro_score")
