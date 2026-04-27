from datetime import datetime
from unittest.mock import patch

import pandas as pd

from quant.data.base import AKShareAdapter, YFinanceAdapter
from quant.data.quality import check_price_quality


def test_get_price_returns_correct_columns():
    # 写一个假的 akshare 返回值，模拟真实列名
    fake_df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.5],
            "最高": [10.8, 11.0],
            "最低": [9.9, 10.2],
            "收盘": [10.2, 10.8],
            "成交量": [1000, 1200],
            "成交额": [10200.0, 12960.0],
            "涨跌幅": [0.5, 0.6],
            "换手率": [0.3, 0.4],
        }
    )

    # patch 把真实的 akshare 替换成返回 fake_df 的假函数
    with patch("quant.data.base.akshare.stock_zh_a_hist", return_value=fake_df):
        adapter = AKShareAdapter()
        result = adapter.get_price(
            symbol="000001",
            period="daily",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 10),
        )

    assert set(result.columns) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "change_pct",
        "turnover_rate",
    }
    assert len(result) == 2


def test_yfinance_returns_correct_columns():
    # 写一个假的 akshare 返回值，模拟真实列名
    cols = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ]
    )
    fake_df = pd.DataFrame(
        [[150.0, 152.0, 149.0, 151.0, 1000000], [151.0, 153.0, 150.0, 152.0, 1100000]],
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        columns=cols,
    )

    with patch("quant.data.base.yf.download", return_value=fake_df):
        adapter = YFinanceAdapter()
        result = adapter.get_price(
            symbol="APPL",
            period="daily",
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 10),
        )

    assert set(result.columns) == {"open", "high", "low", "close", "volume"}
    assert len(result) == 2


def test_data_cached_Fetcher_price_check():
    # 构造一个有问题的 df，来看看报告是否符合预期
    fake_df = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 10.5],
            "close": [-1, 10.8],
            "volume": [0, -1],
            "high": [100, 200],
            "low": [200, 100],
        }
    )

    report = check_price_quality(fake_df)

    assert report["zero_volume_days"] == 1
    assert report["high_lt_low"] == 1
