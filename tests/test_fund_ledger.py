import pandas as pd
import pytest
import yaml

from quant.fund.ledger import (
    compute_holdings,
    load_transactions,
    next_id,
    transaction_returns,
)

# --- 测试数据生成 ---------


def _txns(*rows) -> pd.DataFrame:
    """快速构造流水 DataFrame"""
    df = pd.DataFrame(
        rows, columns=["id", "symbol", "name", "date", "type", "shares", "nav", "note"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


# --- next_id ---------


def test_next_id_empty():
    assert next_id(pd.DataFrame()) == 1


def test_next_id_increments():
    df = pd.DataFrame({"id": [1, 3, 2]})
    assert next_id(df) == 4


# --- compute_holdings --------


def test_compute_holdings_empty():
    result = compute_holdings(pd.DataFrame())
    assert result.empty
    assert list(result.columns) == ["symbol", "name", "shares", "avg_cost_nav"]


def test_compute_holdings_single_buy():
    df = _txns((1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.5, ""))
    result = compute_holdings(df)
    assert len(result) == 1
    assert result.iloc[0]["shares"] == pytest.approx(1000.0)
    assert result.iloc[0]["avg_cost_nav"] == pytest.approx(1.5)


def test_compute_holdings_two_buys_avg_cost():
    df = _txns(
        (1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.0, ""),
        (2, "F1", "基金A", "2024-02-01", "buy", 1000.0, 2.0, ""),
    )
    result = compute_holdings(df)
    # 平均成本 = (1000*1.0 + 1000*2.0) / 2000 = 1.5
    assert result.iloc[0]["avg_cost_nav"] == pytest.approx(1.5)
    assert result.iloc[0]["shares"] == pytest.approx(2000.0)


def test_compute_holdings_partial_sell():
    df = _txns(
        (1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.0, ""),
        (2, "F1", "基金A", "2024-06-01", "sell", 400.0, 1.2, ""),
    )
    result = compute_holdings(df)
    assert result.iloc[0]["shares"] == pytest.approx(600.0)


def test_compute_holdings_full_sell_removed():
    df = _txns(
        (1, "F1", "基金A", "2024-01-01", "buy", 500.0, 1.0, ""),
        (2, "F1", "基金A", "2024-06-01", "sell", 500.0, 1.2, ""),
    )
    result = compute_holdings(df)
    assert result.empty


def test_compute_holdings_multiple_symbols():
    df = _txns(
        (1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.0, ""),
        (2, "F2", "基金B", "2024-01-02", "buy", 500.0, 2.0, ""),
    )
    result = compute_holdings(df)
    assert set(result["symbol"]) == {"F1", "F2"}


# --- transaction_returns ---------


def _make_nav(symbols, n=100):
    idx = pd.date_range("2024-01-01", periods=n, name="date")
    return pd.DataFrame(
        {s: [1.0 + i * 0.005 for i in range(n)] for s in symbols}, index=idx
    )


def test_transaction_returns_columns():
    df = _txns((1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.0, ""))
    nav = _make_nav(["F1"])
    result = transaction_returns(df, nav)
    for col in [
        "交易ID",
        "基金",
        "买入日",
        "买入净值",
        "最新净值",
        "持有天数",
        "收益率",
        "盈亏金额",
    ]:
        assert col in result.columns


def test_transaction_returns_positive_return():
    df = _txns((1, "F1", "基金A", "2024-01-01", "buy", 1000.0, 1.0, ""))
    nav = _make_nav(["F1"])
    result = transaction_returns(df, nav)
    assert result.iloc[0]["收益率"] > 0


def test_transaction_returns_skips_missing_symbol():
    df = _txns((1, "F9", "基金X", "2024-01-01", "buy", 1000.0, 1.0, ""))
    nav = _make_nav(["F1"])  # F9 不在 nav 里
    result = transaction_returns(df, nav)
    assert result.empty


# --- load_transactions ----------


def test_load_transactions_file_not_exist(tmp_path, monkeypatch):
    import quant.fund.ledger as ledger_mod

    monkeypatch.setattr(ledger_mod, "TRANSACTIONS_PATH", tmp_path / "nonexistent.yaml")
    result = load_transactions()
    assert result.empty


def test_load_transactions_parses_yaml(tmp_path, monkeypatch):
    import quant.fund.ledger as ledger_mod

    p = tmp_path / "txns.yaml"
    p.write_text(
        yaml.dump(
            {
                "transactions": [
                    {
                        "id": 1,
                        "symbol": "F1",
                        "name": "基金A",
                        "date": "2024-01-01",
                        "type": "buy",
                        "shares": 500.0,
                        "nav": 1.2,
                        "note": "",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(ledger_mod, "TRANSACTIONS_PATH", p)
    result = load_transactions()
    assert len(result) == 1
    assert result.iloc[0]["symbol"] == "F1"


def test_load_transactions_bad_yaml(tmp_path, monkeypatch):
    import quant.fund.ledger as ledger_mod

    p = tmp_path / "txns.yaml"
    p.write_text(": bad: yaml: [")
    monkeypatch.setattr(ledger_mod, "TRANSACTIONS_PATH", p)
    with pytest.raises(RuntimeError, match="损坏"):
        load_transactions()
