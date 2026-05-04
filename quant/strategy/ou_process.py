# quant/strategy/ou_process.py
"""Ornstein-Uhlenbeck 过程建模：MLE 参数估计与半衰期计算"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUParams:
    """OU 过程参数：dX = θ(μ - X)dt + σdW"""

    theta: float  # 均值回归速度
    mu: float  # 长期均值
    sigma: float  # 波动率
    half_life: float  # 半衰期（交易日）
    log_likelihood: float


def fit_ou(spread: pd.Series) -> OUParams:
    """用 MLE 估计 OU 过程参数

    离散化形式：X_{t+1} = X_t + θ(μ - X_t)Δt + σε
    等价于 AR(1)：X_{t+1} = a + b * X_t + ε
    其中 b = 1 - θΔt，a = θμΔt

    Args:
        spread: 价差时间序列（已去均值的协整残差）

    Returns:
        OUParams 含 θ/μ/σ 和半衰期
    """
    x = spread.values
    x_lag = x[:-1]
    x_now = x[1:]

    # OLS 拟合 AR(1)
    n = len(x_lag)
    b = (n * np.dot(x_lag, x_now) - x_lag.sum() * x_now.sum()) / (
        n * np.dot(x_lag, x_lag) - x_lag.sum() ** 2
    )
    a = x_now.mean() - b * x_lag.mean()
    residuals = x_now - (a + b * x_lag)
    sigma_eps = residuals.std()

    # AR(1) 系数映射回 OU 参数（Δt = 1 交易日）
    dt = 1.0
    theta = (1 - b) / dt
    mu = a / (theta * dt) if theta != 0 else x.mean()
    sigma = sigma_eps / np.sqrt(dt)

    # 半衰期：价差偏离均值后回归一半所需天数
    half_life = np.log(2) / theta if theta > 0 else float("inf")

    # 对数似然
    log_lik = -n / 2 * np.log(2 * np.pi * sigma_eps**2) - np.sum(residuals**2) / (
        2 * sigma_eps**2
    )

    return OUParams(
        theta=theta,
        mu=mu,
        sigma=sigma,
        half_life=half_life,
        log_likelihood=log_lik,
    )


def ou_zscore(
    spread: pd.Series, params: OUParams, window: int | None = None
) -> pd.Series:
    """将价差转换为 OU z-score（相对长期均值的标准差倍数）

    Args:
        spread: 价差序列
        params: OU 参数
        window: 若指定，用滚动均值/标准差代替全局参数（更稳健）

    Returns:
        z-score 序列
    """
    if window:
        mu = spread.rolling(window).mean()
        sigma = spread.rolling(window).std()
    else:
        mu = params.mu
        sigma = params.sigma

    return (spread - mu) / sigma


def entry_exit_thresholds(params: OUParams) -> dict:
    """基于半衰期推导入场/出场阈值建议

    半衰期越短 → 均值回归越快 → 可以用更激进的阈值
    半衰期越长 → 回归慢 → 阈值需要更保守

    Returns:
        建议的开仓/平仓 z-score 阈值
    """
    hl = params.half_life
    if hl < 5:
        entry, exit_ = 1.5, 0.3
    elif hl < 15:
        entry, exit_ = 2.0, 0.5
    elif hl < 30:
        entry, exit_ = 2.5, 0.75
    else:
        entry, exit_ = 3.0, 1.0

    return {"entry": entry, "exit": exit_, "stop_loss": entry * 2}
