"""个股日频资金流向数据拉取"""

from datetime import datetime

import akshare as ak
import pandas as pd

# akshare 中文列名 → 标准英文列名映射
FUND_FLOW_COLS = {
    "日期": "date",
    "主力净流入-净额": "main_net_inflow",
    "主力净流入-净占比": "main_net_pct",
    "超大单净流入-净额": "xlarge_net_inflow",
    "大单净流入-净额": "large_net_inflow",
}


def _get_market(symbol: str) -> str:
    """根据股票代码前缀判断交易所：6 开头为沪市（sh），否则为深市（sz）"""
    return "sh" if symbol.startswith("6") else "sz"


def fetch_fund_flow(
    symbol: str, start_time: datetime, end_time: datetime
) -> pd.DataFrame:
    """获取个股日频资金流向数据。

    akshare 一次返回全部历史，本函数最后按时间范围截取目标区间。
    """
    market = _get_market(symbol)
    df = ak.stock_individual_fund_flow(stock=symbol, market=market)
    df = df[list(FUND_FLOW_COLS.keys())].rename(columns=FUND_FLOW_COLS)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df.loc[pd.Timestamp(start_time) : pd.Timestamp(end_time)]
