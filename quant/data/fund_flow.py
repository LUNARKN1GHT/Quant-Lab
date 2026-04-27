from datetime import datetime

import akshare as ak
import pandas as pd

FUND_FLOW_COLS = {
    "日期": "date",
    "主力净流入-净额": "main_net_inflow",
    "主力净流入-净占比": "main_net_pct",
    "超大单净流入-净额": "xlarge_net_inflow",
    "大单净流入-净额": "large_net_inflow",
}


def _get_market(symbol: str) -> str:
    """根据股票代码前缀判断市场"""
    return "sh" if symbol.startswith("6") else "sz"


def fetch_fund_flow(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """获取个股日频资金流向数据"""
    market = _get_market(symbol)
    df = ak.stock_individual_fund_flow(stock=symbol, market=market)
    df = df[list(FUND_FLOW_COLS.keys())].rename(columns=FUND_FLOW_COLS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
