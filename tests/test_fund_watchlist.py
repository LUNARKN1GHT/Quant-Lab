import pandas as pd
import pytest
import yaml

import quant.fund.watchlist as wl_mod
from quant.fund.watchlist import load_watchlist, save_watchlist


def test_load_watchlist_file_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", tmp_path / "nonexistent.yaml")
    result = load_watchlist()
    assert result.empty
    assert list(result.columns) == ["symbol", "name", "added_date", "note"]


def test_load_watchlist_empty_list(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.yaml"
    p.write_text(yaml.dump({"watchlist": []}))
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", p)
    result = load_watchlist()
    assert result.empty


def test_load_watchlist_parses_items(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.yaml"
    p.write_text(
        yaml.dump(
            {
                "watchlist": [
                    {
                        "symbol": "009610",
                        "name": "基金A",
                        "added_date": "2024-01-01",
                        "note": "",
                    },
                    {
                        "symbol": "110022",
                        "name": "基金B",
                        "added_date": "2024-03-15",
                        "note": "定投",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", p)
    result = load_watchlist()
    assert len(result) == 2
    assert result.iloc[0]["symbol"] == "009610"
    assert isinstance(result.iloc[0]["added_date"], pd.Timestamp)


def test_load_watchlist_bad_yaml(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.yaml"
    p.write_text(": bad: [yaml")
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", p)
    with pytest.raises(RuntimeError, match="损坏"):
        load_watchlist()


def test_save_and_reload_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.yaml"
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", p)
    df = pd.DataFrame(
        [
            {
                "symbol": "009610",
                "name": "基金A",
                "added_date": pd.Timestamp("2024-01-01"),
                "note": "",
            },
        ]
    )
    save_watchlist(df)
    result = load_watchlist()
    assert result.iloc[0]["symbol"] == "009610"
    assert result.iloc[0]["added_date"] == pd.Timestamp("2024-01-01")


def test_save_watchlist_date_formatting(tmp_path, monkeypatch):
    p = tmp_path / "watchlist.yaml"
    monkeypatch.setattr(wl_mod, "WATCHLIST_PATH", p)
    df = pd.DataFrame(
        [
            {
                "symbol": "009610",
                "name": "基金A",
                "added_date": pd.Timestamp("2024-06-15"),
                "note": "",
            },
        ]
    )
    save_watchlist(df)
    raw = yaml.safe_load(p.read_text())
    assert raw["watchlist"][0]["added_date"] == "2024-06-15"
