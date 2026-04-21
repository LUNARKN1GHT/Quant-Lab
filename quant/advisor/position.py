import numpy as np
import pandas as pd

from quant.config import Config
from quant.regime.detector import Regime, detect_regime


def vol_target_scale(returns: pd.Series, cfg: Config) -> pd.Series:
    """波动率目标法：当前波动率高于目标时自动调仓"""
    realized_vol = returns.rolling(cfg.advisor.vol_window).std() * np.sqrt(252)
    scale = (cfg.advisor.target_vol / realized_vol).clip(
        lower=cfg.advisor.min_position, upper=cfg.advisor.max_position
    )

    return scale


def compute_position(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """综合 Regime + 波动率目标，输出每日仓位建议"""
    index_returns = close.mean(axis=1).pct_change()

    regime = detect_regime(
        close,
        ma_window=cfg.regime.ma_window,
        breadth_window=cfg.regime.breadth_window,
        vol_short=cfg.regime.vol_short,
        vol_long=cfg.regime.vol_long,
    )

    scale_map = {
        Regime.BULL.value: cfg.regime.bull_scale,
        Regime.RANGE.value: cfg.regime.range_scale,
        Regime.BEAR.value: cfg.regime.bear_scale,
    }
    regime_scale = regime.map(scale_map)

    v_scale = vol_target_scale(index_returns, cfg)

    position = (regime_scale * v_scale).clip(
        lower=cfg.advisor.min_position, upper=cfg.advisor.max_position
    )

    return pd.DataFrame(
        {
            "regime": regime,
            "regime_scale": regime_scale,
            "vol_scale": v_scale,
            "position": position,
        }
    )
