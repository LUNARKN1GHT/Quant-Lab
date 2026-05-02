"""季频财务数据拉取与 PIT（Point-in-Time）对齐工具

核心问题：回测中直接使用报告期数据会引入前视偏差（look-ahead bias），
因为季报在报告期结束后数周才披露。本模块通过 disclose_date 记录最晚披露日，
确保回测时每个时间点只能使用当时已公开的数据。
"""

import akshare as ak
import pandas as pd

# akshare 返回约 86 列财务指标，从中选取核心字段并重命名为英文
FUNDAMENTAL_COLS = {
    "日期": "report_date",
    "净资产收益率(%)": "roe",
    "总资产利润率(%)": "roa",
    "销售毛利率(%)": "gross_margin",
    "销售净利率(%)": "net_margin",
    "主营业务收入增长率(%)": "revenue_yoy",
    "净利润增长率(%)": "profit_yoy",
    "资产负债率(%)": "debt_ratio",
    "经营现金净流量与净利润的比率(%)": "cfo_to_profit",
}

# 中国 A 股强制披露截止日（保守估计上界）
# Q1(3月底) → 4月30日前，Q2(6月底) → 8月31日前
# Q3(9月底) → 10月31日前，Q4(12月底) → 次年4月30日前
QUARTER_DISCLOSE_LAG = {3: 30, 6: 62, 9: 31, 12: 120}


def _estimate_disclose_date(report_date: pd.Timestamp) -> pd.Timestamp:
    """根据报告期末月份估算最晚披露日期。

    取保守上界（实际可能更早），确保回测中不会提前使用尚未公开的数据。
    """
    lag = QUARTER_DISCLOSE_LAG.get(report_date.month, 60)
    return report_date + pd.Timedelta(days=lag)


def fetch_fundamental(symbol: str, start_year: str = "2015") -> pd.DataFrame:
    """获取季频财务数据，新增 disclose_date 列（估算）用于 PIT 对齐"""
    df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
    df = df[list(FUNDAMENTAL_COLS.keys())].rename(columns=FUNDAMENTAL_COLS)
    df["report_date"] = pd.to_datetime(df["report_date"])
    # 估算每期财报的最晚可用日期，后续因子计算以此为准
    df["disclose_date"] = df["report_date"].apply(_estimate_disclose_date)
    df = df.sort_values("report_date").reset_index(drop=True)
    return df


def align_fundamental_to_daily(
    fundamental: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Point-in-Time 对齐：将季频财务数据前向填充到每个交易日。

    使用 merge_asof（按 disclose_date 做 backward join），
    对每个交易日只保留 disclose_date <= 当日的最新一期财报，
    彻底消除前视偏差。
    """
    fund = (
        fundamental.sort_values("disclose_date")
        .drop(columns=["report_date"])
        .reset_index(drop=True)
    )

    price_df = pd.DataFrame({"date": price_dates.sort_values()})

    # merge_asof 默认 direction="backward"，即取 <= 当日的最近披露记录
    merged = pd.merge_asof(
        price_df,
        fund,
        left_on="date",
        right_on="disclose_date",
    )
    return merged.set_index("date").drop(columns=["disclose_date"])
