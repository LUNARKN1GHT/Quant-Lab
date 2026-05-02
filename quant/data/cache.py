"""数据缓存模块

使用 DuckDB 本地文件数据库缓存行情、估值、财务、资金流向四类数据，
避免重复调用 akshare API，同时为因子计算提供统一的数据入口。
"""

from datetime import datetime
from typing import Literal

import duckdb
import pandas as pd

from quant.data.base import DataFetcher

# get_fundamental 统一返回的列，过滤掉数据库内部的 symbol 元数据列
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
    """带本地 DuckDB 缓存的数据拉取器。

    对上层调用透明：命中缓存直接返回，未命中则调用底层 fetcher 拉取并写入缓存。
    支持价格、估值、财务（含 PIT 披露日）、资金流向四类数据表。
    """

    def __init__(self, fetcher: DataFetcher, db_path: str = "data/quant.duckdb"):
        self.fetcher = fetcher
        self.db_path = db_path

        # 建立持久化连接（文件不存在时自动创建）
        self.connection = duckdb.connect(self.db_path)
        # 初始化旧版 price_cache 表，保留向后兼容
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
        """初始化四张业务数据表，IF NOT EXISTS 保证幂等，重复启动不会报错"""

        # 完整开高低收量 + 换手率，adjust 纳入主键允许同时缓存多种复权方式
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
        # 日频估值：PE-TTM、市净率、总市值
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
        # 季频财务：含 disclose_date 披露日，用于 PIT 因子对齐（防前视偏差）
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
        # 日频资金流向：主力、超大单、大单净流入
        self.connection.execute("""
        CREATE TABLE IF NOT EXISTS fund_flow_daily (
            symbol            VARCHAR,
            date              DATE,
            main_net_inflow   DOUBLE,
            main_net_pct      DOUBLE,
            xlarge_net_inflow DOUBLE,
            large_net_inflow  DOUBLE,
            PRIMARY KEY (symbol, date)
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
        """获取价格数据，优先读取旧版 price_cache 缓存表"""

        # 先查缓存（旧版 price_cache 表）
        cached = self.connection.execute(
            """
            SELECT date, open, close, volume FROM price_cache
            WHERE symbol = ? AND period = ? AND adjust = ?
            AND date >= ? AND date <= ?
            """,
            [symbol, period, adjust, start_time.date(), end_time.date()],
        ).df()

        if len(cached) > 0:
            # 缓存命中，只返回调用方需要的列
            cols: list[str] = ["date", *columns_ask]
            return cached[cols].set_index("date")
        else:
            # 缓存未命中，调用底层适配器拉取数据
            df = self.fetcher.get_price(
                symbol=symbol,
                period=period,
                start_time=start_time,
                end_time=end_time,
                adjust=adjust,
            )

            # 补充元数据列，INSERT 时作为查询条件
            df["symbol"] = symbol
            df["period"] = period
            df["adjust"] = adjust
            df["date"] = df.index

            # 写入缓存，供下次请求复用
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
        """获取日频估值数据（PE-TTM / PB / 总市值），优先读缓存"""
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

        # 懒加载 fetch_valuation，避免模块顶层循环依赖
        from quant.data.valuation import fetch_valuation

        df = fetch_valuation(symbol, start_time, end_time)
        df["symbol"] = symbol
        df["date"] = df.index
        # INSERT OR IGNORE：主键冲突（同一 symbol+date）时静默跳过，防止重复写入
        self.connection.execute("""
            INSERT OR IGNORE INTO valuation_daily
            SELECT symbol, date, pe_ttm, pb, total_mv FROM df
        """)
        return df[["pe_ttm", "pb", "total_mv"]]

    def get_fundamental(self, symbol: str, start_year: str = "2015") -> pd.DataFrame:
        """获取季频财务数据，含 disclose_date 披露日，用于 PIT 因子对齐"""
        cached = self.connection.execute(
            """
            SELECT * FROM fundamental_quarterly
            WHERE symbol = ?
            ORDER BY report_date
        """,
            [symbol],
        ).df()

        if len(cached) > 0:
            return cached[RETURN_COLS]  # 统一列顺序，过滤掉 symbol 元数据列

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

    def get_fund_flow(
        self, symbol: str, start_time: datetime, end_time: datetime
    ) -> pd.DataFrame:
        """获取日频资金流向：主力净流入额/占比、超大单、大单"""
        cached = self.connection.execute(
            """
            SELECT date, main_net_inflow, main_net_pct,
                xlarge_net_inflow, large_net_inflow
            FROM fund_flow_daily
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date
        """,
            [symbol, start_time.date(), end_time.date()],
        ).df()

        if len(cached) > 0:
            return cached.set_index("date")

        from quant.data.fund_flow import fetch_fund_flow

        df = fetch_fund_flow(symbol, start_time, end_time)
        # 单独复制一份用于插入，避免在原始 df 上添加额外列影响返回值
        df_insert = df.copy()
        df_insert["symbol"] = symbol
        df_insert["date"] = df_insert.index
        self.connection.execute("""
            INSERT OR IGNORE INTO fund_flow_daily
            SELECT symbol, date, main_net_inflow, main_net_pct,
                xlarge_net_inflow, large_net_inflow
            FROM df_insert
        """)
        return df
