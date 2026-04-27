from datetime import datetime
from typing import Literal

import duckdb
import pandas as pd

from quant.data.base import DataFetcher

RETURN_COLS = [
    "report_date",
    "disclose_date",
    "roe",
    "roa",
    "gross_margin",
    "net_margin",
    "revenue_yoy",
    "profit_yoy",
    "debt_ratio",
    "cfo_to_profit",
]


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
        self._init_tables()

    def _init_tables(self) -> None:
        # 原有 price_cache 保持不变（或按上次方案扩列）
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS price_daily (
                symbol        VARCHAR,
                period        VARCHAR,
                date          DATE,
                open          DOUBLE,
                high          DOUBLE,
                low           DOUBLE,
                close         DOUBLE,
                volume        BIGINT,
                amount        DOUBLE,
                change_pct    DOUBLE,
                turnover_rate DOUBLE,
                adjust        VARCHAR,
                PRIMARY KEY (symbol, period, date, adjust)
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS valuation_daily (
                symbol   VARCHAR,
                date     DATE,
                pe_ttm   DOUBLE,
                pb       DOUBLE,
                total_mv DOUBLE,
                PRIMARY KEY (symbol, date)
            )
        """)
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_quarterly (
                symbol       VARCHAR,
                report_date  DATE,
                disclose_date DATE,
                roe          DOUBLE,
                roa          DOUBLE,
                gross_margin DOUBLE,
                net_margin   DOUBLE,
                revenue_yoy  DOUBLE,
                profit_yoy   DOUBLE,
                debt_ratio   DOUBLE,
                cfo_to_profit DOUBLE,
                PRIMARY KEY (symbol, report_date)
            )
        """)

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
                adjust=adjust,
            )

            # 加上元素列
            df["symbol"] = symbol
            df["period"] = period
            df["adjust"] = adjust
            df["date"] = df.index

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

    def get_valuation(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        cached = self.connection.execute(
            """
            SELECT date, pe_ttm, pb, total_mv FROM valuation_daily
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """,
            [symbol, start_time.date(), end_time.date()],
        ).df()

        if len(cached) > 0:
            return cached.set_index("date")

        from quant.data.valuation import fetch_valuation

        df = fetch_valuation(symbol, start_time, end_time)
        df["symbol"] = symbol
        df["date"] = df.index
        self.connection.execute("""
            INSERT OR IGNORE INTO valuation_daily
            SELECT symbol, date, pe_ttm, pb, total_mv FROM df
        """)
        return df[["pe_ttm", "pb", "total_mv"]]

    def get_fundamental(self, symbol: str, start_year: str = "2015") -> pd.DataFrame:
        cached = self.connection.execute(
            """
            SELECT * FROM fundamental_quarterly
            WHERE symbol = ?
            ORDER BY report_date
        """,
            [symbol],
        ).df()

        if len(cached) > 0:
            return cached[RETURN_COLS]  # 统一过滤

        from quant.data.fundamental import fetch_fundamental

        df = fetch_fundamental(symbol, start_year)
        df["symbol"] = symbol
        self.connection.execute("""
            INSERT OR IGNORE INTO fundamental_quarterly
            SELECT symbol, report_date, disclose_date,
                roe, roa, gross_margin, net_margin,
                revenue_yoy, profit_yoy, debt_ratio, cfo_to_profit
            FROM df
        """)
        return df[RETURN_COLS]
