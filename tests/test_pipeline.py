from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from quant.pipeline import run_pipeline


def _mock_fetcher(symbols: list[str], n: int = 60) -> MagicMock:
    """返回一个模拟的 fetcher，每次 get_price 返回随机价格 DataFrame"""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")

    fetcher = MagicMock()
    fetcher.get_price.side_effect = lambda symbol, **kwargs: pd.DataFrame(
        {"close": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))},
        index=idx,
    )
    return fetcher


def test_run_pipeline_returns_dict():
    fetcher = _mock_fetcher(["000001", "000002"])
    result = run_pipeline(
        symbols=["000001", "000002"],
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        fetcher=fetcher,
    )
    assert isinstance(result, dict)
    assert "returns" in result
    assert "metrics" in result
    assert "close" in result


def test_run_pipeline_metrics_keys():
    fetcher = _mock_fetcher(["000001", "000002"])
    result = run_pipeline(
        symbols=["000001", "000002"],
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        fetcher=fetcher,
    )
    assert "Sharpe Ratio" in result["metrics"]
    assert "Max Drawdown" in result["metrics"]
    assert "Calmar Ratio" in result["metrics"]
    assert "Sortino Ratio" in result["metrics"]


def test_run_pipeline_close_columns():
    symbols = ["000001", "000002", "000003"]
    fetcher = _mock_fetcher(symbols)
    result = run_pipeline(
        symbols=symbols,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        fetcher=fetcher,
    )
    assert set(result["close"].columns) == set(symbols)


def test_run_pipeline_returns_series():
    fetcher = _mock_fetcher(["000001", "000002"])
    result = run_pipeline(
        symbols=["000001", "000002"],
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        fetcher=fetcher,
    )
    assert isinstance(result["returns"], pd.Series)
