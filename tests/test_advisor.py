import numpy as np
import pandas as pd

from quant.advisor.position import compute_position, vol_target_scale
from quant.config import Config, SignalWeightsConfig


def _make_close(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n)
    return pd.DataFrame(
        {
            "A": 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n)),
            "B": 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n)),
            "C": 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n)),
        },
        index=idx,
    )


def test_vol_target_scale_returns_series():
    close = _make_close()
    cfg = Config()
    returns = close.mean(axis=1).pct_change()
    result = vol_target_scale(returns, cfg)
    assert isinstance(result, pd.Series)
    assert len(result) == len(returns)


def test_vol_target_scale_clipped():
    close = _make_close()
    cfg = Config()
    returns = close.mean(axis=1).pct_change()
    result = vol_target_scale(returns, cfg)
    valid = result.dropna()
    assert (valid >= cfg.advisor.min_position).all()
    assert (valid <= cfg.advisor.max_position).all()


def test_compute_position_structure():
    close = _make_close()
    cfg = Config()
    result = compute_position(close, cfg)
    expected_cols = {
        "regime",
        "regime_signal",
        "vol_signal",
        "macro_signal",
        "sector_signal",
        "position",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(close)


def test_compute_position_no_macro_is_neutral():
    close = _make_close()
    cfg = Config()
    result = compute_position(close, cfg)
    # 无宏观数据时，宏观信号应为中性值 0.5
    assert (result["macro_signal"] == 0.5).all()


def test_compute_position_with_macro():
    close = _make_close()
    cfg = Config()
    macro_score = pd.Series(np.linspace(-2, 2, len(close)), index=close.index)
    result = compute_position(close, cfg, macro_score=macro_score)
    ms = result["macro_signal"].dropna()
    # z-score [-2, 2] → 信号 [0, 1]
    assert (ms >= 0.0).all()
    assert (ms <= 1.0).all()


def test_compute_position_clipped():
    close = _make_close()
    cfg = Config()
    result = compute_position(close, cfg)
    valid = result["position"].dropna()
    assert (valid >= cfg.advisor.min_position).all()
    assert (valid <= cfg.advisor.max_position).all()


def test_compute_position_custom_weights():
    close = _make_close()
    cfg = Config()
    # 将权重全部给 regime，仓位应严格等于 regime_signal
    w = SignalWeightsConfig(regime=1.0, vol=0.0, macro=0.0, sector=0.0)
    result = compute_position(close, cfg, signal_weights=w)
    valid = result["position"].dropna()
    regime_valid = result["regime_signal"].dropna()
    pd.testing.assert_series_equal(
        valid, regime_valid.clip(0.0, 1.0), check_names=False
    )
