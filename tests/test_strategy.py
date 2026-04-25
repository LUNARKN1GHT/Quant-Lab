import numpy as np
import pandas as pd
import pytest

from quant.strategy.compare import compare_strategies
from quant.strategy.factor_strategy import factor_select
from quant.strategy.ml_alpha import walk_forward_predict
from quant.strategy.pairs_trading import (
    check_cointegration,
    generate_signal,
    hedge_ratio,
    spread,
)


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


def test_check_cointegration():
    import numpy as np

    np.random.seed(42)
    # 共同趋势 + 小噪声，真实协整场景
    trend = np.cumsum(np.random.randn(100))
    price_a = pd.Series(trend + np.random.randn(100) * 0.1)
    price_b = pd.Series(trend * 1.1 + np.random.randn(100) * 0.1)
    assert check_cointegration(price_a, price_b)


def test_spread_zero_mean():
    # 标准化后均值应接近 0，std 接近 1
    a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    b = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
    result = spread(a, b)
    assert result.mean() == pytest.approx(0.0, abs=0.02)
    assert result.std() == pytest.approx(1.0, rel=1e-3)


def test_hedge_ratio():
    # price_a = 2 * price_b，hedge ratio 应接近 2.0
    price_b = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    price_a = price_b * 2.0

    result = hedge_ratio(price_a, price_b)

    assert result == pytest.approx(2.0, rel=1e-3)


def test_compare_strategies():
    np.random.seed(42)
    r1 = pd.Series(np.random.randn(252) * 0.01)
    r2 = pd.Series(np.random.randn(252) * 0.01)

    result = compare_strategies({"策略A": r1, "策略B": r2})

    # 验证结构
    assert list(result.columns) == ["策略A", "策略B"]
    assert "Sharpe Ratio" in result.index
    assert "Max Drawdown" in result.index

    # 验证 Max Drawdown 为负
    assert result.loc["Max Drawdown", "策略A"] < 0
    assert result.loc["Max Drawdown", "策略B"] < 0


def test_walk_forward_stack_structure():
    from sklearn.linear_model import Ridge

    from quant.strategy.ml_alpha import walk_forward_stack

    np.random.seed(0)
    X = pd.DataFrame({"f1": np.random.randn(80), "f2": np.random.randn(80)})
    y = pd.Series(np.random.randn(80))

    base_models = [Ridge(alpha=1.0), Ridge(alpha=10.0)]
    meta_model = Ridge()

    result = walk_forward_stack(
        X,
        y,
        base_models=base_models,  # type: ignore
        meta_model=meta_model,
        train_window=40,
        predict_window=10,
        holdout_ratio=0.2,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == len(y)
    # 前 train_window 期没有预测值
    assert result.iloc[:40].isna().all()
    # 之后应有预测值
    assert result.iloc[40:].notna().any()
