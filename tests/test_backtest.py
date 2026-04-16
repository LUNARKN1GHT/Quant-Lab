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


def test_rebalance_weekly():
    from quant.backtest.engine import rebalance

    idx = pd.date_range("2024-01-01", "2024-01-14", freq="D")
    position = pd.DataFrame(
        {"A": [float(i) / 10 for i in range(len(idx))]},
        index=idx,
    )

    result = rebalance(position, freq="W")

    # 第一周没有前向填充来源，是 NaN
    assert result.iloc[0].isna().all()
    # 1/7（周日）是第一周末，持仓锁定为当天的值 0.6
    assert result.loc["2024-01-07", "A"] == pytest.approx(0.6)
    # 1/8~1/13 持仓保持 0.6（前向填充）
    assert result.loc["2024-01-10", "A"] == pytest.approx(0.6)
    # 1/14（第二周末）更新为当天的值 1.3
    assert result.loc["2024-01-14", "A"] == pytest.approx(1.3)
