from datetime import datetime
from typing import Literal, Protocol

import akshare
import pandas as pd


class DataFetcher(Protocol):
    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """获取历史数据的价格"""
        ...


COLUMN_MAP = {"open": "开盘", "close": "收盘", "volume": "成交量"}


class AKShareAdapter:
    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """获取历史数据的价格"""
        price_df = akshare.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_time.strftime("%Y%m%d"),
            end_date=end_time.strftime("%Y%m%d"),
            adjust=adjust,
        )
        price_df["日期"] = pd.to_datetime(price_df["日期"])
        price_df = price_df.set_index("日期")

        cn_columns = [COLUMN_MAP[col] for col in columns_ask]
        return price_df[cn_columns]
