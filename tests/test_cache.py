from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from quant.data.cache import CachedFetcher


def make_mock_fetcher():
    mock_df = pd.DataFrame(
        {"open": [10.0], "close": [11.0], "volume": [1000]},
        index=pd.to_datetime(["2024-01-02"]),
    )
    mock_df.index.name = "date"
    fetcher = MagicMock()
    fetcher.get_price.return_value = mock_df
    return fetcher


def test_cache_miss(tmp_path):
    fetcher = make_mock_fetcher()
    cached = CachedFetcher(fetcher, db_path=str(tmp_path / "test.duckdb"))

    result = cached.get_price(
        symbol="000001",
        period="daily",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 31),
        columns_ask=["close"],
    )

    # 验证调用了真实 fetcher
    fetcher.get_price.assert_called_once()
    assert "close" in result.columns


def test_cache_hit(tmp_path):
    fetcher = make_mock_fetcher()
    cached = CachedFetcher(fetcher, db_path=str(tmp_path / "test.duckdb"))

    kwargs = dict(
        symbol="000001",
        period="daily",
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 31),
        columns_ask=["close"],
    )

    cached.get_price(**kwargs)  # 第一次：cache miss，写入 DB
    cached.get_price(**kwargs)  # 第二次：cache hit，从 DB 读取

    # fetcher 只应该被调用一次
    fetcher.get_price.assert_called_once()
