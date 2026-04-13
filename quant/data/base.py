from datetime import datetime
from typing import Literal, Protocol

import pandas as pd


class DataFetcher(Protocol):
    def get_price(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
    ) -> pd.DataFrame:
        """获取历史数据的价格"""
        pass


class AKShareAdapter:
    def get_price(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        columns_ask: list[Literal["open", "close", "volume"]],
    ) -> pd.DataFrame:
        """获取历史数据的价格"""
        pass
