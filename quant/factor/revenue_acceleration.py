"""财务加速度因子：营收增速的二阶导（环比变化）"""

import pandas as pd


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def revenue_acceleration_pit(
    fundamental: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    symbols: list[str],
) -> pd.DataFrame:
    """计算营收增速二阶导，Point-in-Time 对齐到每个交易日

    Args:
        fundamental: fundamental_quarterly 表，
                    含 symbol/report_date/disclose_date/revenue_yoy
        price_dates: 交易日序列
        symbols: 股票列表

    Returns:
        DataFrame，index=date（名为 'date'），columns=symbol
    """
    fundamental = fundamental.copy()
    fundamental["disclose_date"] = pd.to_datetime(fundamental["disclose_date"])

    results = {}
    for symbol in symbols:
        sub = (
            fundamental[fundamental["symbol"] == symbol]
            .sort_values(["disclose_date", "report_date"])
            .drop_duplicates(subset="disclose_date", keep="last")[
                ["disclose_date", "revenue_yoy"]
            ]
            .dropna()
        )
        if len(sub) < 2:
            continue

        sub = sub.set_index("disclose_date")["revenue_yoy"]

        # 二阶导：相邻两期披露的营收增速差值
        acceleration = sub.diff()

        # 前向填充到每个交易日
        aligned = acceleration.reindex(
            price_dates.union(acceleration.index)
        ).sort_index()
        aligned = aligned.ffill().reindex(price_dates)
        results[symbol] = aligned

    factor = pd.DataFrame(results)
    factor.index.name = "date"
    factor.columns.name = "symbol"

    # 截面截断极值
    factor = factor.apply(
        lambda row: _winsorize(row.dropna()).reindex(row.index), axis=1
    )
    return factor
