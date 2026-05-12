import pandas as pd
import pytest
import yaml

import quant.fund.dca as dca_mod
from quant.fund.dca import (
    load_dca_plans,
    next_trading_day,
    save_dca_plans,
)

# ── next_trading_day ──────────────────────────────────────────────────────────


def test_next_trading_day_exact_match():
    idx = pd.date_range("2024-01-01", periods=5)
    nav = pd.Series([1.0] * 5, index=idx)
    result = next_trading_day(pd.Timestamp("2024-01-03"), nav)
    assert result == pd.Timestamp("2024-01-03")


def test_next_trading_day_skips_to_next():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])
    nav = pd.Series([1.0] * 3, index=idx)
    # 2024-01-01 不在序列里，应跳到 2024-01-02
    result = next_trading_day(pd.Timestamp("2024-01-01"), nav)
    assert result == pd.Timestamp("2024-01-02")


def test_next_trading_day_beyond_series():
    idx = pd.date_range("2024-01-01", periods=3)
    nav = pd.Series([1.0] * 3, index=idx)
    result = next_trading_day(pd.Timestamp("2025-01-01"), nav)
    assert result is None


def test_next_trading_day_ignores_nan():
    idx = pd.date_range("2024-01-01", periods=4)
    nav = pd.Series([float("nan"), 1.0, 1.1, float("nan")], index=idx)
    result = next_trading_day(pd.Timestamp("2024-01-01"), nav)
    # 第一个非 NaN 值在 2024-01-02
    assert result == pd.Timestamp("2024-01-02")


# ── load_dca_plans ────────────────────────────────────────────────────────────


def test_load_dca_plans_file_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(dca_mod, "DCA_PATH", tmp_path / "nonexistent.yaml")
    result = load_dca_plans()
    assert result.empty


def test_load_dca_plans_empty_list(tmp_path, monkeypatch):
    p = tmp_path / "dca.yaml"
    p.write_text(yaml.dump({"dca_plans": []}))
    monkeypatch.setattr(dca_mod, "DCA_PATH", p)
    result = load_dca_plans()
    assert result.empty


def test_load_dca_plans_parses_items(tmp_path, monkeypatch):
    p = tmp_path / "dca.yaml"
    p.write_text(
        yaml.dump(
            {
                "dca_plans": [
                    {
                        "symbol": "009610",
                        "name": "基金A",
                        "amount": 500.0,
                        "period": "monthly",
                        "note": "",
                    },
                    {
                        "symbol": "110022",
                        "name": "基金B",
                        "amount": 1000.0,
                        "period": "weekly",
                        "note": "定投",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(dca_mod, "DCA_PATH", p)
    result = load_dca_plans()
    assert len(result) == 2
    assert result.iloc[0]["symbol"] == "009610"
    assert float(result.iloc[0]["amount"]) == 500.0


def test_load_dca_plans_bad_yaml(tmp_path, monkeypatch):
    p = tmp_path / "dca.yaml"
    p.write_text(": bad: [yaml")
    monkeypatch.setattr(dca_mod, "DCA_PATH", p)
    with pytest.raises(RuntimeError, match="损坏"):
        load_dca_plans()


# ── save_dca_plans ────────────────────────────────────────────────────────────


def test_save_and_reload_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "dca.yaml"
    monkeypatch.setattr(dca_mod, "DCA_PATH", p)
    df = pd.DataFrame(
        [
            {
                "symbol": "009610",
                "name": "基金A",
                "amount": 500.0,
                "period": "monthly",
                "note": "",
            },
        ]
    )
    save_dca_plans(df)
    result = load_dca_plans()
    assert result.iloc[0]["symbol"] == "009610"
    assert float(result.iloc[0]["amount"]) == 500.0


def test_save_dca_plans_yaml_structure(tmp_path, monkeypatch):
    p = tmp_path / "dca.yaml"
    monkeypatch.setattr(dca_mod, "DCA_PATH", p)
    df = pd.DataFrame(
        [
            {
                "symbol": "009610",
                "name": "基金A",
                "amount": 300.0,
                "period": "weekly",
                "note": "测试",
            },
        ]
    )
    save_dca_plans(df)
    raw = yaml.safe_load(p.read_text())
    assert "dca_plans" in raw
    assert raw["dca_plans"][0]["period"] == "weekly"
