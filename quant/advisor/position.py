"""仓位建议模块：思路信号甲醛融合动态调仓

最终仓位 = Σ(wᵢ × signalᵢ) / Σwᵢ，截断到 [min, max]：
- regime_signal：牛/震荡/熊市对应 1.0/0.6/0.2
- vol_signal：波动率目标法归一化到 [0, 1]
- macro_signal：宏观景气 z-score 线性映射到 [0, 1]
- sector_signal：行业动量占位符（恒为 0.5，下阶段接入行业 RS）
"""

import numpy as np
import pandas as pd

from quant.config import Config, SignalWeightsConfig
from quant.regime.detector import Regime, detect_regime


def vol_target_scale(returns: pd.Series, cfg: Config) -> pd.Series:
    """波动率目标法：target_vol / realized_vol, 截断到 [0, 1]"""
    realized_vol: pd.Series = returns.rolling(cfg.advisor.vol_window).std() * np.sqrt(
        252
    )
    return (cfg.advisor.target_vol / realized_vol).clip(0.0, 1.0)


def compute_position(
    close: pd.DataFrame,
    cfg: Config,
    macro_score: pd.Series | None = None,
    signal_weights: SignalWeightsConfig | None = None,
) -> pd.DataFrame:
    """四路信号加权融合，输出每日仓位建议。

    Args:
        close: 成分股收盘价宽表，用于计算等权指数和市场宽度
        cfg: 全局配置，包含各层参数
        macro_score: 宏观景气指数（z-score），None 时宏观信号取 0.5（中性）
        signal_weights: 信号融合权重，None 时使用 cfg.signal_weights

    Returns:
        DataFrame，列为各信号分量与最终仓位，便于归因分析
    """
    w = signal_weights if signal_weights is not None else cfg.signal_weights
    index_returns = close.mean(axis=1).pct_change()  # 等权指数日收益率

    # 信号 1: 市场环境
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
    regime_signal = regime.map(scale_map)

    # 信号 2: 波动率目标
    vol_signal = vol_target_scale(index_returns, cfg)

    # 信号 3: 宏观景气
    if macro_score is not None:
        macro_aligned = macro_score.reindex(close.index, method="ffill")
        macro_signal = ((macro_aligned.clip(-2, 2) + 2) / 4).fillna(0.5)
    else:
        macro_signal = pd.Series(0.5, index=close.index)

    # 信号 4: 行业动量
    sector_signal = pd.Series(0.5, index=close.index)

    # 加权融合
    total_w = w.regime + w.vol + w.macro + w.sector
    weighted = [
        (w.regime / total_w) * regime_signal,
        (w.vol / total_w) * vol_signal,
        (w.macro / total_w) * macro_signal,
        (w.sector / total_w) * sector_signal,
    ]
    position = sum(
        s if coef > 0 else pd.Series(0.0, index=close.index)
        for coef, s in zip([w.regime, w.vol, w.macro, w.sector], weighted)
    ).clip(lower=cfg.advisor.min_position, upper=cfg.advisor.max_position)  # type: ignore

    return pd.DataFrame(
        {
            "regime": regime,
            "regime_signal": regime_signal,
            "vol_signal": vol_signal,
            "macro_signal": macro_signal,
            "sector_signal": sector_signal,
            "position": position,
        }
    )
