import pandas as pd


def test_load_or_fetch_cache_hit(tmp_path, monkeypatch):
    import quant.macro.loader as loader

    monkeypatch.setattr(loader, "MACRO_DIR", tmp_path)

    idx = pd.date_range("2020-01-31", periods=5, freq="ME")
    s = pd.Series([3.0, 3.1, 3.2, 3.3, 3.4], index=idx, name="bond_yield")
    s.to_csv(tmp_path / "cached.csv", header=True)

    # fetch_fn 不应被调用
    result = loader._load_or_fetch(
        "cached", lambda: (_ for _ in ()).throw(AssertionError("不应调用 fetch"))
    )
    assert isinstance(result, pd.Series)
    assert len(result) == 5


def test_load_or_fetch_cache_miss_saves_file(tmp_path, monkeypatch):
    import quant.macro.loader as loader

    monkeypatch.setattr(loader, "MACRO_DIR", tmp_path)

    idx = pd.date_range("2021-01-31", periods=4, freq="ME")
    fetched = pd.Series([50.0, 51.0, 52.0, 53.0], index=idx, name="pmi")

    result = loader._load_or_fetch("pmi_test", lambda: fetched)

    assert isinstance(result, pd.Series)
    assert len(result) == 4
    assert (tmp_path / "pmi_test.csv").exists()  # 已写入缓存


def test_load_bond_yield_from_cache(tmp_path, monkeypatch):
    import quant.macro.loader as loader

    monkeypatch.setattr(loader, "MACRO_DIR", tmp_path)

    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    s = pd.Series([3.0] * 10, index=idx, name="bond_yield")
    s.to_csv(tmp_path / "bond_yield.csv", header=True)

    result = loader.load_bond_yield()
    assert isinstance(result, pd.Series)
    assert len(result) == 10


def test_load_pmi_from_cache(tmp_path, monkeypatch):
    import quant.macro.loader as loader

    monkeypatch.setattr(loader, "MACRO_DIR", tmp_path)

    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    s = pd.Series([50.5] * 12, index=idx, name="pmi")
    s.to_csv(tmp_path / "pmi.csv", header=True)
