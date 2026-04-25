import numpy as np
import pandas as pd

from quant.advisor.position import compute_position, vol_target_scale
from quant.config import Config


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
    assert isinstance(result, pd.DataFrame)
    expected_cols = {
        "regime",
        "regime_scale",
        "vol_scale",
        "macro_multiplier",
        "position",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == len(close)


def test_compute_position_no_macro_multiplier_is_one():
    close = _make_close()
    cfg = Config()
    result = compute_position(close, cfg)
    assert (result["macro_multiplier"] == 1.0).all()


def test_compute_position_with_macro():
    close = _make_close()
    cfg = Config()
    macro_score = pd.Series(
        np.linspace(-2, 2, len(close)),
        index=close.index,
    )
    result = compute_position(close, cfg, macro_score=macro_score)
    mm = result["macro_multiplier"].dropna()
    # 宏观乘数由 clip(-2,2) 限制，范围应在 [0.6, 1.4]
    assert (mm >= 0.6).all()
    assert (mm <= 1.4).all()


def test_compute_position_clipped():
    close = _make_close()
    cfg = Config()
    result = compute_position(close, cfg)
    valid = result["position"].dropna()
    assert (valid >= cfg.advisor.min_position).all()
    assert (valid <= cfg.advisor.max_position).all()
