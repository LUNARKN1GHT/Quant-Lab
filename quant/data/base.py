"""数据处理基础模块

定义统一的 DataFetcher Protocol 接口，以及 A 股（akshare）和美股（yfinance）适配器。
所有适配器输出相同的列名（open/high/low/close/volume），下游代码与数据源解耦。
"""

from datetime import datetime
from typing import Literal, Protocol

import akshare
import pandas as pd
import yfinance as yf


class DataFetcher(Protocol):
    """数据拉取适配器的统一接口（结构子类型）。

    使用 typing.Protocol 而非抽象基类，适配器无需显式继承即可满足接口约束（鸭子类型）。
    CachedFetcher 等上层组件统一接受 DataFetcher，运行时可自由替换底层数据源。
    """

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


# yfinance 返回大写列名，统一转小写以匹配标准 schema
COLUMN_MAP_YFINANCE = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}

# yfinance interval 参数与通用 period 枚举的映射
PERIOD_MAP_YFINANCE = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}


class AKShareAdapter:
    """A 股历史数据适配器，底层调用 akshare.stock_zh_a_hist。

    输出 schema：DatetimeIndex（日期名称）+ 8 个标准英文列。
    adjust 支持 qfq（前复权）/ hfq（后复权）/ ''（不复权），默认前复权。
    """

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
        # 将 akshare 中文列名映射为标准英文列名
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
        # 显式指定列顺序，过滤 akshare 可能额外返回的字段
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
    """美股历史数据适配器，底层调用 yfinance.download。

    adjust 参数在此处无意义（yfinance 通过 auto_adjust=True 自动复权），
    保留该参数仅为满足 DataFetcher Protocol，使两个适配器可互换。
    """

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
        # 多 ticker 时 yfinance 返回 MultiIndex 列，取第 0 层降维为普通列名
        price_df.columns = price_df.columns.get_level_values(0)
        return price_df.rename(columns=COLUMN_MAP_YFINANCE)[
            ["open", "high", "low", "close", "volume"]
        ]
