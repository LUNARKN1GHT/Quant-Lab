from datetime import datetime
from unittest.mock import patch

import pandas as pd

from quant.data.cache import CachedFetcher
from quant.data.quality import check_fundamental_quality, check_valuation_quality


def _make_fake_fund_flow():
    idx = pd.to_datetime(["2023-01-03", "2023-01-04"])
    return pd.DataFrame(
        {
            "main_net_inflow": [1e8, -2e8],
            "main_net_pct": [1.5, -2.0],
            "xlarge_net_inflow": [5e7, -1e8],
            "large_net_inflow": [5e7, -1e8],
        },
        index=idx,
    )


def test_get_fund_flow_cache_miss(tmp_path):
    fetcher = CachedFetcher(object(), db_path=str(tmp_path / "test.duckdb"))  # type: ignore[arg-type]

    with patch(
        "quant.data.fund_flow.fetch_fund_flow", return_value=_make_fake_fund_flow()
    ):
        result = fetcher.get_fund_flow(
            symbol="000001",
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 31),
        )

    assert "main_net_inflow" in result.columns
    assert len(result) == 2


def test_get_fund_flow_cache_hit(tmp_path):
    fetcher = CachedFetcher(object(), db_path=str(tmp_path / "test.duckdb"))  # type: ignore[arg-type]
    kwargs = dict(
        symbol="000001", start_time=datetime(2023, 1, 1), end_time=datetime(2023, 1, 31)
    )

    with patch(
        "quant.data.fund_flow.fetch_fund_flow", return_value=_make_fake_fund_flow()
    ) as mock_fetch:
        fetcher.get_fund_flow(**kwargs)  # type: ignore
        fetcher.get_fund_flow(**kwargs)  # type: ignore

    mock_fetch.assert_called_once()


def test_check_valuation_quality():
    df = pd.DataFrame(
        {
            "pe_ttm": [10.0, -5.0, None],
            "pb": [1.0, 1.2, None],
            "total_mv": [2000.0, 0.0, 1500.0],
        },
        index=pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"]),
    )
    report = check_valuation_quality(df)

    assert report["negative_pe_days"] == 1
    assert report["zero_mv_days"] == 1
    assert report["null_counts"]["pe_ttm"] == 1


def test_check_fundamental_quality():
    df = pd.DataFrame(
        {
            "report_date": pd.to_datetime(["2022-09-30", "2022-12-31"]),
            "roe": [12.0, 12.0],
            "roa": [1.5, 1.5],  # 所有数值列都相同，才算 stale
        }
    )
    report = check_fundamental_quality(df)

    assert report["total_quarters"] == 2
    assert report["stale_quarters"] == 1
