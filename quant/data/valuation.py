"""获取市场估值数据（PE-TTM / PB / 总市值）"""

from datetime import datetime

import akshare as ak
import pandas as pd


def fetch_valuation(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """从百度股市通拉取个股日频估值数据。

    分三次 API 调用分别获取 PE-TTM、PB、总市值，再横向合并成宽表。
    每次都拉取"全部"历史，最后按时间范围截取，避免多次调用时的日期对齐问题。
    """
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

    # 三列按日期对齐后合并，sort_index 确保时序正确
    result: pd.DataFrame = pd.DataFrame(frames).sort_index()
    result = result.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
    return result
