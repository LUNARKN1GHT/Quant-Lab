from unittest.mock import patch

import pandas as pd


def test_fetch_sector_close_cache_hit(tmp_path, monkeypatch):
    import quant.sector.loader as loader

    monkeypatch.setattr(loader, "SECTOR_DIR", tmp_path)

    idx = pd.date_range("2023-01-01", periods=20)
    df = pd.DataFrame({"收盘": [100.0 + i for i in range(20)]}, index=idx)
    df.index.name = "日期"
    df.to_csv(tmp_path / "801010.csv")

    result = loader.fetch_sector_close("801010")
    assert isinstance(result, pd.Series)
    assert len(result) == 20


def test_fetch_sector_close_cache_miss(tmp_path, monkeypatch):
    import quant.sector.loader as loader

    monkeypatch.setattr(loader, "SECTOR_DIR", tmp_path)

    mock_df = pd.DataFrame(
        {
            "日期": pd.date_range("2023-01-01", periods=5).strftime("%Y-%m-%d"),
            "收盘": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    with patch("quant.sector.loader.ak.index_hist_sw", return_value=mock_df):
        result = loader.fetch_sector_close("801010")

    assert isinstance(result, pd.Series)
    assert len(result) == 5
    assert (tmp_path / "801010.csv").exists()  # 已写入缓存


def test_get_sector_names():
    import quant.sector.loader as loader

    mock_df = pd.DataFrame(
        {
            "行业代码": ["801010.SI", "801020.SI", "801030.SI"],
            "行业名称": ["农林牧渔", "采掘", "化工"],
        }
    )
    with patch("quant.sector.loader.ak.sw_index_first_info", return_value=mock_df):
        result = loader.get_sector_names()

    assert result == {"801010": "农林牧渔", "801020": "采掘", "801030": "化工"}


def test_load_sector_close(tmp_path, monkeypatch):
    import quant.sector.loader as loader

    monkeypatch.setattr(loader, "SECTOR_DIR", tmp_path)

    idx = pd.date_range("2020-01-01", periods=30)
    for code in ["801010", "801020"]:
        df = pd.DataFrame({"收盘": [100.0] * 30}, index=idx)
        df.index.name = "日期"
        df.to_csv(tmp_path / f"{code}.csv")

    mock_names = pd.DataFrame(
        {
            "行业代码": ["801010.SI", "801020.SI"],
            "行业名称": ["农林牧渔", "采掘"],
        }
    )
    with patch("quant.sector.loader.ak.sw_index_first_info", return_value=mock_names):
        result = loader.load_sector_close(start="20200101")

    assert isinstance(result, pd.DataFrame)
    assert "农林牧渔" in result.columns
    assert "采掘" in result.columns
