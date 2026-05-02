"""估值动量因子：PE 变化率"""

import pandas as pd


def valuation_momentum(
    valuation: pd.DataFrame,
    price_dates: pd.DatetimeIndex,
    symbols: list[str],
    window: int = 20,
) -> pd.DataFrame:
    """计算 PE 变化率因子，做 Point-in-Time 日频对齐

    Args:
        valuation: valuation_daily 表，含 symbol/date/pe_ttm
        price_dates: 交易日序列
        symbols: 股票列表
        window: 计算 PE 变化的回看窗口（交易日）

    Returns:
        DataFrame，index=date，columns=symbol，值为因子值
    """
    valuation = valuation.copy()
    valuation["date"] = pd.to_datetime(valuation["date"])
    # 负 PE 无意义，置为 NaN
    valuation.loc[valuation["pe_ttm"] <= 0, "pe_ttm"] = None

    results = {}
    for symbol in symbols:
        sub = (
            valuation[valuation["symbol"] == symbol]
            .sort_values("date")
            .drop_duplicates(subset="date", keep="last")
            .set_index("date")["pe_ttm"]
            .dropna()
        )
        if sub.empty:
            continue

        # 前向填充到每个交易日
        aligned = sub.reindex(price_dates.union(sub.index)).sort_index()
        aligned = aligned.ffill().reindex(price_dates)

        # PE 变化率（分母取绝对值防止符号翻转）
        pe_change = (aligned - aligned.shift(window)) / aligned.shift(window).abs()
        results[symbol] = pe_change

    factor = pd.DataFrame(results)
    factor.index.name = "date"
    factor.columns.name = "symbol"
    return factor
