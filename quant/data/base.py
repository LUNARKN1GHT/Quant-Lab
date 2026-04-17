from datetime import datetime
from typing import Literal, Protocol

import akshare
import pandas as pd
import yfinance as yf


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


COLUMN_MAP_AKSHARE = {"open": "开盘", "close": "收盘", "volume": "成交量"}
COLUMN_MAP_YFINANCE = {"open": "Open", "close": "Close", "volume": "Volume"}

PERIOD_MAP_YFINANCE = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}


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

        cn_columns = [COLUMN_MAP_AKSHARE[col] for col in columns_ask]
        return price_df[cn_columns].rename(columns=dict(zip(cn_columns, columns_ask)))


class YFinanceAdapter:
    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """获取历史数据的价格，_adjust 参数对美股无效，仅为保持接口一致"""
        price_df = yf.download(
            tickers=symbol,
            start=start_time.strftime("%Y-%m-%d"),
            end=end_time.strftime("%Y-%m-%d"),
            interval=PERIOD_MAP_YFINANCE[period],
            progress=False,
            auto_adjust=True,
        )
        # yfinance 返回 MultiIndex 列，压平取第一层
        price_df.columns = price_df.columns.get_level_values(0)

        en_columns = [COLUMN_MAP_YFINANCE[col] for col in columns_ask]
        return price_df[en_columns].rename(columns=dict(zip(en_columns, columns_ask)))
