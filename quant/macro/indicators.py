"""计算各宏观指标对大盘的滞后相关性，以及合成景气度指数"""

import pandas as pd


def calc_lag_corr(
    macro: pd.Series,
    market_ret: pd.Series,
    max_lag: int = 6,
) -> pd.Series:
    """计算宏观指标之后 0～max_lag 月对大盘月收益的相关性"""
    market_monthly = (1 + market_ret).resample("ME").prod() - 1
    result = {}
    for lag in range(max_lag + 1):
        shifted = macro.shift(lag).reindex(market_monthly.index, method="ffill")
        common = shifted.dropna().index.intersection(market_monthly.dropna().index)
        if len(common) > 12:
            result[lag] = shifted[common].corr(market_monthly[common])
        else:
            result[lag] = float("nan")

    return pd.Series(result, name=macro.name)


def composite_index(macro_df: pd.DataFrame) -> pd.Series:
    """合成景气指数"""
    directions = {"pmi": 1, "bond_yield": -1, "m2_yoy": 1, "cpi_yoy": 1}
    normalized = {}
    for col in macro_df.columns:
        s = macro_df[col].dropna()
        z = (s - s.mean()) / s.std()
        normalized[col] = z * directions.get(col, 1)
    aligned = pd.DataFrame(normalized).ffill()

    return aligned.mean(axis=1).rename("macro_score")
