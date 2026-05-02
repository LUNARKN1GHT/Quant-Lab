"""行业轮动分析：相对强度（RS）计算与配置建议

RS（Relative Strength）= 行业 N 日涨幅 / 基准 N 日涨幅，
大于 1 表示行业跑赢大盘（相对强势），小于 1 表示跑输。
RS 动量（RS Momentum）= RS 序列的线性回归斜率，衡量相对强势是否在加速。
"""

import numpy as np
import pandas as pd


def calc_rs(
    sector_close: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """计算各行业相对基准的滚动 RS = 行业 N 日涨幅 / 基准 N 日涨幅。

    Args:
        sector_close: 行业收盘价宽表（行=日期，列=行业名称）
        benchmark: 基准指数收盘价（如沪深300）
        window: 计算区间涨幅的窗口，默认 20 日

    Returns:
        RS 宽表，值>1 代表相对强势，<1 代表相对弱势
    """
    sector_ret = sector_close.pct_change(window)
    bench_ret = benchmark.pct_change(window)
    # 基准涨幅为 0 时除法无意义，替换为 NaN 避免 inf
    rs = sector_ret.div(bench_ret.replace(0, float("nan")), axis=0)
    return rs


def calc_rs_momentum(rs: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """计算最新截面各行业的 RS 动量（RS 序列过去 lookback 日的线性回归斜率）。

    斜率为正且较大说明相对强势仍在加速，是行业轮动的买入信号。
    数据点不足 lookback//2 时置为 NaN，避免用极少数据拟合产生噪声。
    """
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
    """综合 RS 水平与 RS 动量，输出行业超配/低配建议。

    综合排名 = (RS 排名 + RS 动量排名) / 2，前 top_n 超配，后 top_n 低配。

    Args:
        rs_latest: 各行业最新 RS 值
        rs_momentum: 各行业 RS 动量（斜率）
        top_n: 超配/低配各选几个行业，默认 3

    Returns:
        含排名与建议的 DataFrame
    """
    df = pd.DataFrame({"RS": rs_latest, "RS动量": rs_momentum}).dropna()
    df["RS排名"] = df["RS"].rank(ascending=False)
    df["动量排名"] = df["RS动量"].rank(ascending=False)
    df["综合排名"] = (df["RS排名"] + df["动量排名"]) / 2
    df = df.sort_values("综合排名")
    df["建议"] = "中性"
    df.iloc[:top_n, df.columns.get_loc("建议")] = "超配 ▲"  # type: ignore
    df.iloc[-top_n:, df.columns.get_loc("建议")] = "低配 ▼"  # type: ignore
    return df
