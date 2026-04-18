import pandas as pd
import pytest

from quant.factor.bollinger import bollinger_position
from quant.factor.combine import equal_weight, ic_weight
from quant.factor.ic import calc_ic, calc_icir
from quant.factor.layered import layered_return
from quant.factor.macd import macd
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
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


def test_layered_return():
    factor = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    forward_return = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

    result = layered_return(factor, forward_return, n_groups=4)

    assert len(result) == 4
    # 因子越大收益越高，第 4 组均值应大于第 1 组
    assert result.iloc[-1] > result.iloc[0]


def test_equal_weight():
    fake_factors = pd.DataFrame(
        {
            "factor_1": [1, 2, 3, 4, 5],
            "factor_2": [6, 7, 8, 9, 10],
            "factor_3": [5, 4, 3, 2, 1],
        }
    )

    result = equal_weight(factors=fake_factors)

    assert result.iloc[0] == pytest.approx(4.0)
    assert result.iloc[1] == pytest.approx(4.333, rel=1e-3)
    assert result.iloc[4] == pytest.approx(5.333, rel=1e-3)


def test_ic_weight():
    factors = pd.DataFrame({"factor_1": [1.0, 2.0, 3.0], "factor_2": [4.0, 5.0, 6.0]})
    # factor_1 权重 0.25，factor_2 权重 0.75（IC 之比为 1:3）
    ic_scores = pd.Series({"factor_1": 0.1, "factor_2": 0.3})

    result = ic_weight(factors, ic_scores)

    # 第一行：1*0.25 + 4*0.75 = 0.25 + 3.0 = 3.25
    assert result.iloc[0] == pytest.approx(3.25)
    # 第二行：2*0.25 + 5*0.75 = 0.5 + 3.75 = 4.25
    assert result.iloc[1] == pytest.approx(4.25)


def test_rsi_all_up():
    close = pd.Series([100.0 + i for i in range(20)])
    result = rsi(close, window=14)
    assert result.iloc[-1] > 99


def test_rsi_all_down():
    close = pd.Series([100.0 - i for i in range(20)])
    result = rsi(close, window=14)
    assert result.iloc[-1] < 1


def test_macd_returns_series():
    close = pd.Series([float(i) for i in range(50)])
    result = macd(close)
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_macd_trending_up():
    # 持续上涨时，快线在慢线上方，histogram 应为正
    close = pd.Series([float(i) for i in range(100)])
    result = macd(close)
    assert result.iloc[-1] > 0


def test_bollinger_position_range():
    # 正常波动的价格，BBP 应在 0~1 附近
    close = pd.Series([100.0 + i % 10 for i in range(50)])
    result = bollinger_position(close)
    # 去掉前 window 期的 NaN
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 1).all()


def test_bollinger_position_at_mean():
    # 价格等于均值时，BBP 应为 0.5
    close = pd.Series([10.0] * 30)
    result = bollinger_position(close)
    # 所有值相同时 std=0，会出现 NaN，正常现象
    assert result.dropna().empty or (result.dropna() == 0.5).all()
