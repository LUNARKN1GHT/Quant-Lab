import pandas as pd
import pytest

from quant.backtest.engine import backtest


def test_backtest_no_shift():
    # 两只股票，三天
    position = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    returns = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.02, 0.04, 0.06]})

    result = backtest(position, returns, shift=False)

    # 每天：0.5*return_A + 0.5*return_B = return（因为两只股票收益相同）
    assert result.iloc[0] == pytest.approx(0.02)
    assert result.iloc[1] == pytest.approx(0.04)
    assert result.iloc[2] == pytest.approx(0.06)


def test_backtest_with_shift():
    position = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    returns = pd.DataFrame({"A": [0.02, 0.04, 0.06], "B": [0.02, 0.04, 0.06]})

    result = backtest(position, returns, shift=True)

    # shift=-1 后 returns 变成 [0.04, 0.06, NaN]
    # 第 0 天持仓对应第 1 天收益
    assert result.iloc[0] == pytest.approx(0.04)
    assert result.iloc[1] == pytest.approx(0.06)
    # 最后一天没有下期收益，结果是 NaN
    assert result.iloc[2] == pytest.approx(0.0)
