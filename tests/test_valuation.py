from datetime import datetime
from unittest.mock import patch

import pandas as pd

from quant.data.cache import CachedFetcher
from quant.data.fundamental import align_fundamental_to_daily


def _make_fake_valuation():
    idx = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
    return pd.DataFrame(
        {
            "pe_ttm": [10.0, 10.1, 10.2],
            "pb": [1.0, 1.1, 1.2],
            "total_mv": [2000.0, 2010.0, 2020.0],
        },
        index=idx,
    )


def _make_fake_fundamental():
    return pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2022-09-30", "2022-12-31"]),
            "disclose_date": pd.to_datetime(["2022-10-31", "2023-04-30"]),
            "roe": [12.0, 13.0],
            "roa": [1.5, 1.6],
            "gross_margin": [30.0, 31.0],
            "net_margin": [10.0, 11.0],
            "revenue_yoy": [5.0, 6.0],
            "profit_yoy": [8.0, 9.0],
            "debt_ratio": [60.0, 61.0],
            "cfo_to_profit": [90.0, 95.0],
        }
    )


def test_get_valuation_cache_miss(tmp_path):
    fetcher = CachedFetcher(object(), db_path=str(tmp_path / "test.duckdb"))  # type: ignore

    with patch(
        "quant.data.valuation.fetch_valuation", return_value=_make_fake_valuation()
    ):
        result = fetcher.get_valuation(
            symbol="000001",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 31),
        )

    assert set(result.columns) == {"pe_ttm", "pb", "total_mv"}
    assert len(result) == 3


def test_get_valuation_cache_hit(tmp_path):
    fetcher = CachedFetcher(object(), db_path=str(tmp_path / "test.duckdb"))  # type: ignore
    kwargs = dict(
        symbol="000001", start_time=datetime(2023, 1, 1), end_time=datetime(2023, 1, 31)
    )

    with patch(
        "quant.data.valuation.fetch_valuation", return_value=_make_fake_valuation()
    ) as mock_fetch:
        fetcher.get_valuation(**kwargs)  # type: ignore
        fetcher.get_valuation(**kwargs)  # 第二次应命中缓存 # type: ignore

    mock_fetch.assert_called_once()  # API 只调用一次


def test_get_fundamental_cache_miss(tmp_path):
    fetcher = CachedFetcher(object(), db_path=str(tmp_path / "test.duckdb"))  # type: ignore

    with patch(
        "quant.data.fundamental.fetch_fundamental",
        return_value=_make_fake_fundamental(),
    ):
        result = fetcher.get_fundamental(symbol="000001")

    assert "roe" in result.columns
    assert len(result) == 2


def test_align_fundamental_to_daily():
    fund = _make_fake_fundamental()
    price_dates = pd.to_datetime(["2023-01-03", "2023-03-01", "2023-05-01"])

    aligned = align_fundamental_to_daily(fund, price_dates)

    # 2023-01-03：disclose_date 最早是 2022-10-31，已披露，应有值
    assert not pd.isna(aligned.loc["2023-01-03", "roe"])
    # 2023-05-01：两期均已披露，应填充最新一期（2022-12-31 那期）
    assert aligned.loc["2023-05-01", "roe"] == 13.0
