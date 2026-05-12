import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from quant.config import Config
from quant.fund.advisor_signal import (
    FUND_TYPE_SCALE,
    fund_position_advice,
    latest_signal,
    load_local_close,
)


# ── 测试数据工厂 ──────────────────────────────────────────────────────────────

def _make_close(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, name="date")
    data = {
        sym: 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
        for sym in ["A", "B", "C", "D", "E"]
    }
    return pd.DataFrame(data, index=idx)


def _make_mock_con(close_df: pd.DataFrame) -> MagicMock:
    """返回模拟 duckdb 连接，execute().df() 返回 long-format 价格表"""
    long = close_df.reset_index().melt(id_vars="date", var_name="symbol", value_name="close")
    long["adjust"] = "qfq"
    mock_con = MagicMock()
    mock_con.execute.return_value.df.return_value = long
    return mock_con


# ── load_local_close ─────────────────────────────────────────────────────────

def test_load_local_close_returns_wide_dataframe():
    close_df = _make_close()
    mock_con = _make_mock_con(close_df)
    with patch("quant.fund.advisor_signal.duckdb.connect", return_value=mock_con):
        result = load_local_close()
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(close_df.columns)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_load_local_close_closes_connection():
    close_df = _make_close()
    mock_con = _make_mock_con(close_df)
    with patch("quant.fund.advisor_signal.duckdb.connect", return_value=mock_con):
        load_local_close()
    mock_con.close.assert_called_once()


# ── latest_signal ────────────────────────────────────────────────────────────

def test_latest_signal_keys():
    close = _make_close()
    cfg = Config()
    sig = latest_signal(cfg, close=close)
    expected = {
        "regime", "regime_label", "regime_emoji",
        "regime_scale", "vol_scale", "macro_multiplier",
        "position", "date",
    }
    assert expected == set(sig.keys())


def test_latest_signal_regime_is_valid():
    close = _make_close()
    cfg = Config()
    sig = latest_signal(cfg, close=close)
    assert sig["regime"] in ("BULL", "RANGE", "BEAR")


def test_latest_signal_position_in_range():
    close = _make_close()
    cfg = Config()
    sig = latest_signal(cfg, close=close)
    assert cfg.advisor.min_position <= sig["position"] <= cfg.advisor.max_position


def test_latest_signal_emoji_matches_regime():
    close = _make_close()
    cfg = Config()
    sig = latest_signal(cfg, close=close)
    emoji_map = {"BULL": "🟢", "RANGE": "🟡", "BEAR": "🔴"}
    assert sig["regime_emoji"] == emoji_map[sig["regime"]]


def test_latest_signal_uses_local_close_when_none(monkeypatch):
    """close=None 时应自动调用 load_local_close()"""
    close = _make_close()
    mock_con = _make_mock_con(close)
    cfg = Config()
    with patch("quant.fund.advisor_signal.duckdb.connect", return_value=mock_con):
        sig = latest_signal(cfg, close=None)
    assert "regime" in sig


# ── fund_position_advice ──────────────────────────────────────────────────────

def _make_holdings(symbols=("F1", "F2", "F3"), fund_types=("equity", "bond", "balanced")):
    rows = [
        {"symbol": s, "name": f"基金{s}", "mkt": 10000.0, "fund_type": ft}
        for s, ft in zip(symbols, fund_types)
    ]
    return pd.DataFrame(rows)


def _make_signal(regime: str = "RANGE", position: float = 0.6) -> dict:
    labels = {"BULL": ("BULL — 趋势上行", "🟢"),
               "RANGE": ("RANGE — 震荡整理", "🟡"),
               "BEAR": ("BEAR — 趋势下行", "🔴")}
    label, emoji = labels[regime]
    return {
        "regime": regime,
        "regime_label": label,
        "regime_emoji": emoji,
        "regime_scale": 0.6,
        "vol_scale": 1.0,
        "macro_multiplier": 1.0,
        "position": position,
        "date": pd.Timestamp("2024-01-01"),
    }


def test_fund_position_advice_returns_all_holdings():
    holdings = _make_holdings()
    sig = _make_signal()
    result = fund_position_advice(sig, holdings, total_capital=30000.0)
    assert len(result) == len(holdings)


def test_fund_position_advice_columns():
    holdings = _make_holdings()
    sig = _make_signal()
    result = fund_position_advice(sig, holdings, total_capital=30000.0)
    for col in ["名称", "代码", "类型", "当前仓位", "建议仓位", "偏差金额", "操作建议"]:
        assert col in result.columns


def test_fund_position_advice_action_add():
    """仓位明显低于建议时应给出加仓建议"""
    holdings = pd.DataFrame([{"symbol": "F1", "name": "基金F1", "mkt": 100.0, "fund_type": "equity"}])
    sig = _make_signal(position=0.9)
    result = fund_position_advice(sig, holdings, total_capital=10000.0)
    assert "加仓" in result.iloc[0]["操作建议"]


def test_fund_position_advice_action_hold():
    """仓位与建议相近时应保持不动"""
    # total_capital = mkt / suggested_weight 使偏差为零
    sig = _make_signal(regime="RANGE", position=0.5)
    mkt = 5000.0
    total = mkt / 0.5  # suggested_mkt == current_mkt
    holdings = pd.DataFrame([{"symbol": "F1", "name": "基金F1", "mkt": mkt, "fund_type": "equity"}])
    result = fund_position_advice(sig, holdings, total_capital=total)
    assert "持有" in result.iloc[0]["操作建议"]


def test_fund_position_advice_bond_scale_bull():
    """债基在牛市建议仓位应低于权益基（bond BULL scale=0.5 < equity BULL scale=1.0）"""
    sig = _make_signal(regime="BULL", position=0.8)
    holdings = pd.DataFrame([
        {"symbol": "EQ", "name": "权益基", "mkt": 8000.0, "fund_type": "equity"},
        {"symbol": "BD", "name": "债基",   "mkt": 8000.0, "fund_type": "bond"},
    ])
    result = fund_position_advice(sig, holdings, total_capital=16000.0).set_index("代码")
    assert result.loc["BD", "建议仓位"] < result.loc["EQ", "建议仓位"]


def test_fund_position_advice_default_fund_type():
    """fund_type 缺失时默认视为 equity"""
    holdings = pd.DataFrame([{"symbol": "F1", "name": "基金F1", "mkt": 5000.0}])
    sig = _make_signal(position=0.6)
    result = fund_position_advice(sig, holdings, total_capital=10000.0)
    expected = sig["position"] * FUND_TYPE_SCALE["equity"][sig["regime"]]
    assert abs(result.iloc[0]["建议仓位"] - expected) < 1e-9
