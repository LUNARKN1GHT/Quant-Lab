from datetime import datetime
from typing import Literal

import duckdb

from quant.data.base import DataFetcher


class CachedFetcher:
    """数据缓存类"""

    def __init__(self, fetcher: DataFetcher, db_path: str = "data/quant.duckdb"):
        self.fetcher = fetcher
        self.db_path = db_path

        # 建立与数据库的连接
        self.connection = duckdb.connect(self.db_path)
        # 建表
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS price_cache (
                symbol VARCHAR,
                period VARCHAR,
                date DATE,
                open DOUBLE,
                close DOUBLE,
                volume BIGINT,
                adjust VARCHAR
            )
            """
        )

    def get_price(
        self,
        symbol: str,
        period: Literal["daily", "weekly", "monthly"],
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
        adjust: Literal["qfq", "hfq", ""] = "qfq",
    ):

        # 查表
        cached = self.connection.execute(
            """
            SELECT date, open, close, volume FROM price_cache
            WHERE symbol = ? AND period = ? AND adjust = ?
            AND date >= ? AND date <= ?
            """,
            [symbol, period, adjust, start_time.date(), end_time.date()],
        ).df()

        if len(cached) > 0:
            # 如果查找成功，可以直接返回这个表格
            cols: list[str] = ["date", *columns_ask]
            return cached[cols].set_index("date")
        else:
            # 查找失败，从API获取新的数据，并存入
            df = self.fetcher.get_price(
                symbol=symbol,
                period=period,
                start_time=start_time,
                end_time=end_time,
                columns_ask=["open", "close", "volume"],
                adjust=adjust,
            )

            # 加上元素列
            df["symbol"] = symbol
            df["period"] = period
            df["adjust"] = adjust
            df["date"] = df.index
            df = df.rename(
                columns={"开盘": "open", "收盘": "close", "成交量": "volume"}
            )

            # 写入数据库
            self.connection.execute(
                """INSERT INTO price_cache
                SELECT
                    symbol,
                    period,
                    date,
                    open,
                    close,
                    volume,
                    adjust
                FROM df
                """
            )

            return df[columns_ask]
