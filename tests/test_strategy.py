import numpy as np
import pandas as pd
import pytest

from quant.strategy.factor_strategy import factor_select
from quant.strategy.ml_alpha import walk_forward_predict
from quant.strategy.pairs_trading import generate_signal


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


def test_generate_signal():
    spread = pd.Series([-3.0, -1.0, 0.3, 1.0, 3.0])
    result = generate_signal(spread)

    assert result.iloc[0] == 1  # < -2，做多
    assert result.iloc[2] == 0  # abs < 0.5，平仓
    assert result.iloc[4] == -1  # > 2，做空


def test_walk_forward_predict_structure():
    X = pd.DataFrame({"f1": np.random.randn(50), "f2": np.random.randn(50)})
    y = pd.Series(np.random.randn(50))

    result = walk_forward_predict(X, y, train_window=20, predict_window=5)

    assert len(result) == len(y)
    assert result.index.equals(y.index)
    # 前 train_window 期没有预测值，应为 NaN
    assert result.iloc[:20].isna().all()
