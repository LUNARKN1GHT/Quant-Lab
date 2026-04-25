import numpy as np
import pandas as pd

from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions


def _sectors(n: int = 60, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n)
    return pd.DataFrame(
        {
            f"行业{i + 1}": 100 * np.cumprod(1 + rng.normal(0.001, 0.02, n))
            for i in range(k)
        },
        index=idx,
    )


def test_calc_rs_returns_dataframe():
    df = _sectors()
    bench = df.mean(axis=1)
    result = calc_rs(df, bench, window=20)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape


def test_calc_rs_no_inf():
    df = _sectors()
    bench = df.mean(axis=1)
    result = calc_rs(df, bench, window=20)
    assert not result.isin([float("inf"), float("-inf")]).any().any()


def test_calc_rs_benchmark_zero_replaced():
    # 基准为 0 时不应报错，应返回 NaN
    df = _sectors(n=30, k=2)
    bench = pd.Series(0.0, index=df.index)
    result = calc_rs(df, bench, window=5)
    assert isinstance(result, pd.DataFrame)


def test_calc_rs_momentum_returns_series():
    df = _sectors(n=80)
    bench = df.mean(axis=1)
    rs = calc_rs(df, bench, window=10)
    result = calc_rs_momentum(rs, lookback=20)
    assert isinstance(result, pd.Series)
    assert len(result) == df.shape[1]


def test_calc_rs_momentum_insufficient_data():
    # 数据不足 lookback//2 时返回 nan
    df = _sectors(n=10, k=3)
    bench = df.mean(axis=1)
    rs = calc_rs(df, bench, window=5)
    # 只有 10 行，pct_change 后有 NaN，dropna 后 < lookback//2=10
    result = calc_rs_momentum(rs, lookback=20)
    assert isinstance(result, pd.Series)


def test_get_suggestions_structure():
    rng = np.random.default_rng(1)
    sectors = [f"行业{i}" for i in range(6)]
    rs_latest = pd.Series(rng.normal(1, 0.1, 6), index=sectors)
    rs_momentum = pd.Series(rng.normal(0, 0.01, 6), index=sectors)
    result = get_suggestions(rs_latest, rs_momentum, top_n=2)
    assert isinstance(result, pd.DataFrame)
    assert "建议" in result.columns
    assert "RS" in result.columns
    assert "综合排名" in result.columns


def test_get_suggestions_top_n_counts():
    rng = np.random.default_rng(2)
    sectors = [f"S{i}" for i in range(10)]
    rs_latest = pd.Series(rng.normal(1, 0.2, 10), index=sectors)
    rs_momentum = pd.Series(rng.normal(0, 0.1, 10), index=sectors)
    result = get_suggestions(rs_latest, rs_momentum, top_n=3)
    assert (result["建议"] == "超配 ▲").sum() == 3
    assert (result["建议"] == "低配 ▼").sum() == 3


def test_get_suggestions_default_top_n():
    rng = np.random.default_rng(3)
    sectors = [f"X{i}" for i in range(8)]
    rs_latest = pd.Series(rng.normal(1, 0.1, 8), index=sectors)
    rs_momentum = pd.Series(rng.normal(0, 0.05, 8), index=sectors)
    result = get_suggestions(rs_latest, rs_momentum)  # top_n=3 默认
    assert (result["建议"] == "超配 ▲").sum() == 3
    assert (result["建议"] == "低配 ▼").sum() == 3
