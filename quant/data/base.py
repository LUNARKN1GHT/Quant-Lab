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
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """获取历史价格数据，返回标准化列名"""
        ...


COLUMN_MAP_YFINANCE = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

PERIOD_MAP_YFINANCE = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}


class AKShareAdapter:
    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        df = akshare.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_time.strftime("%Y%m%d"),
            end_date=end_time.strftime("%Y%m%d"),
            adjust=adjust,
        )
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期")
        df = df.rename(
            columns={
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "change_pct",
                "换手率": "turnover_rate",
            }
        )
        return df[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
                "change_pct",
                "turnover_rate",
            ]
        ]


class YFinanceAdapter:
    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ) -> pd.DataFrame:
        """adjust 参数对美股无效，仅为保持接口一致"""
        price_df = yf.download(
            tickers=symbol,
            start=start_time.strftime("%Y-%m-%d"),
            end=end_time.strftime("%Y-%m-%d"),
            interval=PERIOD_MAP_YFINANCE[period],
            progress=False,
            auto_adjust=True,
        )
        if price_df is None or price_df.empty:
            raise ValueError(f"yfinance returned no data for {symbol}")
        price_df.columns = price_df.columns.get_level_values(0)
        return price_df.rename(columns=COLUMN_MAP_YFINANCE)[
            ["open", "high", "low", "close", "volume"]
        ]
