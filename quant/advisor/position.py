"""仓位建议模块：使用波动率目标法 + 市场环境 + 宏观景气三层乘数动态调仓

最终仓位 = regime_scale × vol_scale × macro_multiplier，并截断到 [min, max]：
- regime_scale：根据牛/震荡/熊市设定基础仓位比例
- vol_scale：当前波动率高于目标时降仓（波动率目标法）
- macro_multiplier：宏观景气高时略微加仓，低时略微减仓
"""

import numpy as np
import pandas as pd

from quant.config import Config
from quant.regime.detector import Regime, detect_regime


def vol_target_scale(returns: pd.Series, cfg: Config) -> pd.Series:
    """波动率目标法：动态调整仓位使组合波动率趋向目标值。

    scale = target_vol / realized_vol，波动率越高则仓位越低。
    结果截断到 [min_position, max_position] 防止过度杠杆或空仓。
    """
    realized_vol = returns.rolling(cfg.advisor.vol_window).std() * np.sqrt(252)
    scale = (cfg.advisor.target_vol / realized_vol).clip(
        lower=cfg.advisor.min_position, upper=cfg.advisor.max_position
    )
    return scale


def compute_position(
    close: pd.DataFrame,
    cfg: Config,
    macro_score: pd.Series | None = None,
) -> pd.DataFrame:
    """综合市场环境 + 波动率目标 + 宏观景气，输出每日仓位建议。

    Args:
        close: 成分股收盘价宽表，用于计算等权指数和市场宽度
        cfg: 全局配置，包含各层参数
        macro_score: 宏观景气指数（z-score），None 时宏观乘数恒为 1

    Returns:
        DataFrame，列为各中间变量与最终仓位，便于归因分析
    """
    index_returns = close.mean(axis=1).pct_change()  # 等权指数日收益率

    regime = detect_regime(
        close,
        ma_window=cfg.regime.ma_window,
        breadth_window=cfg.regime.breadth_window,
        vol_short=cfg.regime.vol_short,
        vol_long=cfg.regime.vol_long,
    )

    # 将 Regime 枚举值映射为对应的仓位倍数
    scale_map = {
        Regime.BULL.value: cfg.regime.bull_scale,
        Regime.RANGE.value: cfg.regime.range_scale,
        Regime.BEAR.value: cfg.regime.bear_scale,
    }
    regime_scale = regime.map(scale_map)
    v_scale = vol_target_scale(index_returns, cfg)

    # 宏观景气乘数：z-score ∈ [-2, 2] 映射到 [0.6, 1.4]
    # clip(-2,2) 防止极端宏观数据造成过大调整
    if macro_score is not None:
        macro_aligned = macro_score.reindex(close.index, method="ffill")
        macro_multiplier = (1 + 0.2 * macro_aligned.clip(-2, 2)).fillna(1.0)
    else:
        macro_multiplier = pd.Series(1.0, index=close.index)

    position = (regime_scale * v_scale * macro_multiplier).clip(
        lower=cfg.advisor.min_position, upper=cfg.advisor.max_position
    )

    return pd.DataFrame(
        {
            "regime": regime,
            "regime_scale": regime_scale,
            "vol_scale": v_scale,
            "macro_multiplier": macro_multiplier,
            "position": position,
        }
    )
