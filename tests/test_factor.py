import pandas as pd
import pytest

from quant.factor.ic import calc_ic, calc_icir
from quant.factor.momentum import momentum
from quant.factor.turnover import turnover
from quant.factor.volatility import volatility


def test_momentum_factor():
    # 一个假的价格序列
    fake_close_series = pd.Series([1, 2, 3, 4, 5, 6])
    window: int = 2

    # 序列有限，我们使用小一点的窗口即可
    result = momentum(fake_close_series, window)

    assert result.iloc[:window].isna().all()
    # iloc[2] = close[2]/close[0] - 1 = 3/1 - 1 = 2.0
    assert result.iloc[2] == pytest.approx(2.0)
    # iloc[3] = close[3]/close[1] - 1 = 4/2 - 1 = 1.0
    assert result.iloc[3] == pytest.approx(1.0)
    # iloc[5] = close[5]/close[3] - 1 = 6/4 - 1 = 0.5
    assert result.iloc[5] == pytest.approx(0.5)


def test_volatility_factor():
    fake_close_series = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41])
    window = 3

    result = volatility(fake_close_series, window)

    assert result.iloc[:window].isna().all()
    # 收益率固定，波动率应该为0
    assert result.iloc[window] == pytest.approx(0.0, abs=1e-10)


def test_turnover_factor():
    fake_volume_series = pd.Series([100, 200, 300, 400, 500])
    window: int = 3

    result = turnover(fake_volume_series, window)

    assert result.iloc[: window - 1].isna().all()
    assert result.iloc[2] == pytest.approx(1.5)
    assert result.iloc[3] == pytest.approx(1.333, rel=1e-3)


def test_ic():
    fake_factor_series = pd.Series([100, 200, 300, 400])
    fake_return_series = pd.Series([5, 6, 8, 10])

    ic = calc_ic(fake_factor_series, fake_return_series)

    assert ic == pytest.approx(1.0)


def test_icir():
    # 构造一个 IC 序列
    ic_series = pd.Series([0.1, 0.2, 0.3, 0.2, 0.1])
    result = calc_icir(ic_series=ic_series)
    assert result == pytest.approx(ic_series.mean() / ic_series.std())
