import pandas as pd
import pytest

from quant.strategy.factor_strategy import factor_select


def test_factor_select():
    scores = pd.Series({"A": 0.1, "B": 0.5, "C": 0.3, "D": 0.8})
    result = factor_select(scores, top_n=2)

    # 权重之和为 1
    assert result.sum() == pytest.approx(1.0)
    # 最高分的两只（B、D）权重各为 0.5
    assert result["D"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.5)
    # 其余为 0
    assert result["A"] == pytest.approx(0.0)
