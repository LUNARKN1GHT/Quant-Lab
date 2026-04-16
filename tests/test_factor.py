import pandas as pd
import pytest

from quant.factor.momentum import momentum


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
