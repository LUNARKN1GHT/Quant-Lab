"""盈利质量因子：经营现金流 / 净利润"""

import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位数截断极值"""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def earnings_quality_pit(
    fundamental: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    """Point-in-Time 对齐，返回每个交易日每只股票的盈利质量因子值

    Args:
        fundamental: fundamental_quarterly 表，含 symbol/disclose_date/cfo_to_profit
        price_dates: 交易日序列
        symbols: 股票列表

    Returns:
        DataFrame，index=date，columns=symbol，值为因子值
    """
    fundamental = fundamental.copy()
    fundamental["disclose_date"] = pd.to_datetime(fundamental["disclose_date"])

    results = {}
    for symbol in symbols:
        sub = (
            fundamental[fundamental["symbol"] == symbol]
            .sort_values(["disclose_date", "report_date"])  # 先按披露日，再按报告期排序
            .drop_duplicates(subset="disclose_date", keep="last")[  # 同日保留最新报告期
                ["disclose_date", "cfo_to_profit"]
            ]
            .dropna()
        )
        if sub.empty:
            continue

        # 对每个交易日，找截止当日最新披露的报告
        sub = sub.set_index("disclose_date")["cfo_to_profit"]
        # reindex 到所有交易日，前向填充（只用已披露数据）
        aligned = sub.reindex(price_dates.union(sub.index)).sort_index()
        aligned = aligned.ffill().reindex(price_dates)
        results[symbol] = aligned

    factor = pd.DataFrame(results)
    factor.index.name = "date"

    # 截面截断极值（每个交易日横截面）
    factor = factor.apply(
        lambda row: winsorize(row.dropna()).reindex(row.index), axis=1
    )
    return factor
