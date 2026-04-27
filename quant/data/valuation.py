"""获取市场估值数据"""

from datetime import datetime

import akshare as ak
import pandas as pd


def fetch_valuation(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """获取个股日频估值数据"""
    indicators = {
        "pe_ttm": "市盈率(TTM)",
        "pb": "市净率",
        "total_mv": "总市值",
    }

    frames = {}
    for col, indicator in indicators.items():
        df = ak.stock_zh_valuation_baidu(
            symbol=symbol, indicator=indicator, period="全部"
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").rename(columns={"value": col})
        frames[col] = df[col]

    result: pd.DataFrame = pd.DataFrame(frames).sort_index()
    result = result.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
    return result
