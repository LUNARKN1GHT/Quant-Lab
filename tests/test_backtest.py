import pandas as pd
import pytest

from quant.backtest.engine import backtest


def test_backtest_no_shift_no_cost():
    position = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    returns = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.02, 0.04, 0.06]})

    result = backtest(position, returns, shift=False, commission_rate=0.0)

    assert result.iloc[0] == pytest.approx(0.02)
    assert result.iloc[1] == pytest.approx(0.04)
    assert result.iloc[2] == pytest.approx(0.06)


def test_backtest_with_shift_no_cost():
    position = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    returns = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.02, 0.04, 0.06]})

    result = backtest(position, returns, shift=True, commission_rate=0.0)

    assert result.iloc[0] == pytest.approx(0.04)
    assert result.iloc[1] == pytest.approx(0.06)
    assert result.iloc[2] == pytest.approx(0.0)


def test_backtest_commission():
    # 第 1 天调仓：A 从 0.5→0.8，B 从 0.5→0.2，换手量 = 0.3+0.3 = 0.6
    # 手续费 = 0.6 * 0.0003 = 0.00018
    position = pd.DataFrame({"A": [0.5, 0.8, 0.8], "B": [0.5, 0.2, 0.2]})
    returns = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.02, 0.04, 0.06]})

    result = backtest(position, returns, shift=False, commission_rate=0.0003)

    assert result.iloc[0] == pytest.approx(0.02)
    assert result.iloc[1] == pytest.approx(0.04 - 0.00018)
    assert result.iloc[2] == pytest.approx(0.06)
