import os
import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── 在模块被 import 前注入 mock ────────────────────────────────────────
os.environ.setdefault("TUSHARE_TOKEN", "fake_token")
_mock_ts = MagicMock()
_mock_ts.set_token = MagicMock()
_mock_ts.pro_api = MagicMock(return_value=MagicMock())
sys.modules.setdefault("tushare", _mock_ts)
sys.modules.pop("quant.fund.loader", None)

import quant.fund.loader as fund_loader  # noqa: E402

# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pro():
    """每个测试独立控制 _pro 返回值"""
    mock = MagicMock()
    original = fund_loader._pro
    fund_loader._pro = mock
    yield mock
    fund_loader._pro = original


def test_load_fund_nav_returns_series(mock_pro):
    mock_pro.fund_nav.return_value = pd.DataFrame(
        {
            "end_date": ["20230101", "20230102", "20230103"],
            "unit_nav": [1.0, 1.01, 1.02],
        }
    )
    result = fund_loader.load_fund_nav("000001")
    assert isinstance(result, pd.Series)
    assert result.name == "000001"
    assert len(result) == 3
    assert result.dtype == float


def test_load_fund_nav_appends_of_suffix(mock_pro):
    mock_pro.fund_nav.return_value = pd.DataFrame(
        {
            "end_date": ["20230101"],
            "unit_nav": [1.5],
        }
    )
    fund_loader.load_fund_nav("000001")
    mock_pro.fund_nav.assert_called_once_with(
        ts_code="000001.OF", fields="end_date,unit_nav"
    )


def test_load_fund_nav_already_has_suffix(mock_pro):
    mock_pro.fund_nav.return_value = pd.DataFrame(
        {
            "end_date": ["20230101"],
            "unit_nav": [1.5],
        }
    )
    fund_loader.load_fund_nav("000001.OF")
    mock_pro.fund_nav.assert_called_once_with(
        ts_code="000001.OF", fields="end_date,unit_nav"
    )


def test_load_funds_success(mock_pro):
    mock_pro.fund_nav.return_value = pd.DataFrame(
        {
            "end_date": ["20230101", "20230102"],
            "unit_nav": [1.0, 1.01],
        }
    )
    result = fund_loader.load_funds(["000001", "000002"])
    assert isinstance(result, pd.DataFrame)
    assert "000001" in result.columns
    assert "000002" in result.columns


def test_load_funds_handles_error(mock_pro):
    mock_pro.fund_nav.side_effect = Exception("API 失败")
    result = fund_loader.load_funds(["000001"])
    # 出错时跳过该基金，不抛异常
    assert isinstance(result, pd.DataFrame)
