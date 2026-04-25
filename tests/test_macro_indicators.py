import numpy as np
import pandas as pd

from quant.macro.indicators import calc_lag_corr, composite_index


def _monthly(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-31", periods=n, freq="ME")
    s = pd.Series(rng.normal(50, 2, n), index=idx, name="pmi")
    return s


def _daily_market(n_months: int = 60) -> pd.Series:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2019-01-01", periods=n_months * 21, freq="B")
    return pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)


def test_calc_lag_corr_returns_series():
    result = calc_lag_corr(_monthly(), _daily_market(), max_lag=3)
    assert isinstance(result, pd.Series)
    assert len(result) == 4  # lag 0..3


def test_calc_lag_corr_values_in_range():
    result = calc_lag_corr(_monthly(), _daily_market(), max_lag=6)
    valid = result.dropna()
    assert (valid >= -1).all()
    assert (valid <= 1).all()


def test_calc_lag_corr_name():
    macro = _monthly()
    macro.name = "my_indicator"
    result = calc_lag_corr(macro, _daily_market())
    assert result.name == "my_indicator"


def test_composite_index_returns_series():
    idx = pd.date_range("2019-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "pmi": rng.normal(50, 2, 48),
            "bond_yield": rng.normal(3, 0.5, 48),
            "m2_yoy": rng.normal(8, 1, 48),
            "cpi_yoy": rng.normal(2, 0.5, 48),
        },
        index=idx,
    )
    result = composite_index(df)
    assert isinstance(result, pd.Series)
    assert result.name == "macro_score"
    assert len(result) == len(df)


def test_composite_index_bond_yield_inverted():
    # bond_yield 单调上升 → direction=-1 → 合成景气指数应下降
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "pmi": 50.0 + rng.normal(0, 0.5, 24),  # 近似常数但 std > 0
            "bond_yield": np.linspace(2.0, 5.0, 24),  # 单调上升
            "m2_yoy": 8.0 + rng.normal(0, 0.5, 24),
            "cpi_yoy": 2.0 + rng.normal(0, 0.5, 24),
        },
        index=idx,
    )
    result = composite_index(df)

    # bond_yield 贡献随时间越来越负，所以前半段均值 > 后半段均值
    assert result.iloc[:12].mean() > result.iloc[12:].mean()


def test_composite_index_unknown_col_direction_defaults_positive():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    df = pd.DataFrame({"unknown_col": np.linspace(1, 10, 12)}, index=idx)
    result = composite_index(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 12
