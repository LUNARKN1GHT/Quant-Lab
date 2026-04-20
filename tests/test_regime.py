import numpy as np
import pandas as pd

from quant.regime.detector import Regime, detect_regime

# 用小窗口避免构造大量数据
SMALL_WINDOWS = dict(ma_window=10, breadth_window=5, vol_short=5, vol_long=10)
WARMUP = 9


def _make_close(daily_return: float, n: int = 50, n_stocks: int = 3) -> pd.DataFrame:
    """构造 n 只股票以固定日收益率涨跌的收盘价宽表"""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * (1 + daily_return) ** np.arange(n)
    return pd.DataFrame(
        {f"S{i}": prices * (1 + i * 0.05) for i in range(n_stocks)},
        index=dates,
    )


def test_detect_regime_returns_series():
    close = _make_close(0.01)
    result = detect_regime(close, **SMALL_WINDOWS)
    assert isinstance(result, pd.Series)
    assert len(result) == len(close)


def test_detect_regime_bull():
    # 价格持续上涨
    close = _make_close(0.01, n=50)
    result = detect_regime(close, **SMALL_WINDOWS)

    assert (result.iloc[WARMUP:] == Regime.BULL).all()


def test_detect_regime_bear():
    # 价格持续下跌
    close = _make_close(-0.01, n=50)
    result = detect_regime(close, **SMALL_WINDOWS)

    assert (result.iloc[WARMUP:] == Regime.BEAR).all()


def test_detect_regime_warmup_is_range():
    # 预热期信号为 NaN
    close = _make_close(0.01, n=50)
    result = detect_regime(close, **SMALL_WINDOWS)
    assert (result.iloc[:WARMUP] == Regime.RANGE).all()
