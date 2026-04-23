"""RS计算"""

import numpy as np
import pandas as pd


def calc_rs(
    sector_close: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """滚动 RS = 行业N日涨幅 / 基准N日涨幅"""
    sector_ret = sector_close.pct_change(window)
    bench_ret = benchmark.pct_change(window)
    # 避免除以零
    rs = sector_ret.div(bench_ret.replace(0, float("nan")), axis=0)
    return rs


def calc_rs_momentum(rs: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """最新截面的 RS 动量：过去 lookback 日 RS 的斜率（线性回归）"""
    recent = rs.iloc[-lookback:]
    result = {}
    for col in recent.columns:
        y = recent[col].dropna().values
        if len(y) < lookback // 2:
            result[col] = float("nan")
            continue
        slope = np.polyfit(range(len(y)), y, 1)[0]  # type: ignore
        result[col] = slope
    return pd.Series(result)


def get_suggestions(
    rs_latest: pd.Series,
    rs_momentum: pd.Series,
    top_n: int = 3,
) -> pd.DataFrame:
    """综合 RS 水平 + 动量，输出超配/低配建议"""
    df = pd.DataFrame({"RS": rs_latest, "RS动量": rs_momentum}).dropna()
    df["RS排名"] = df["RS"].rank(ascending=False)
    df["动量排名"] = df["RS动量"].rank(ascending=False)
    df["综合排名"] = (df["RS排名"] + df["动量排名"]) / 2
    df = df.sort_values("综合排名")
    df["建议"] = "中性"
    df.iloc[:top_n, df.columns.get_loc("建议")] = "超配 ▲"  # type: ignore
    df.iloc[-top_n:, df.columns.get_loc("建议")] = "低配 ▼"  # type: ignore
    return df
