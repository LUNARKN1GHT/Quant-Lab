"""Kalman Filter 动态对冲比率估计

将配对交易的对冲比率建模为随机游走，KF 每日更新估计：
  price_a = β * price_b + α + ε
其中 β, α 随时间缓慢漂移，KF 给出后验估计及不确定度。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanResult:
    hedge_ratio: pd.Series  # β_t 时序
    intercept: pd.Series  # α_t 时序
    spread: pd.Series  # price_a - β_t * price_b - α_t
    variance: pd.Series  # 观测残差方差（用于动态 z-score）
    R: float  # 观测噪声方差（估计值）
    Q_scale: float  # 过程噪声缩放系数


def fit_kalman(
    price_a: pd.Series,
    price_b: pd.Series,
    delta: float = 1e-4,
    obs_noise_init: float = 1e-3,
) -> KalmanResult:
    """用 Kalman Filter 估计时变对冲比率。

    Args:
        price_a: 股票 A 价格序列
        price_b: 股票 B 价格序列
        delta:   过程噪声强度，控制 β 漂移速度；越大跟踪越快但噪声越大
        obs_noise_init: 观测噪声初始方差 R

    Returns:
        KalmanResult，含逐日 hedge_ratio / intercept / spread
    """
    n = len(price_a)
    assert len(price_b) == n

    # 状态向量 [β, α]，2 维
    # 过程噪声协方差 Q = delta / (1 - delta) * I
    Q = delta / (1 - delta) * np.eye(2)
    R = obs_noise_init

    # 初始化：用前 20 期 OLS 热启动
    init_n = min(20, n)
    X_init = np.column_stack([price_b.values[:init_n], np.ones(init_n)])
    y_init = price_a.values[:init_n]
    beta_ols, _, _, _ = np.linalg.lstsq(X_init, y_init, rcond=None)

    beta = beta_ols.copy()  # 状态均值 [β, α]
    P = np.eye(2)  # 状态协方差

    betas = np.zeros((n, 2))
    variances = np.zeros(n)

    for t in range(n):
        pb = price_b.values[t]
        H = np.array([[pb, 1.0]])  # 观测矩阵 (1×2)

        # 预测步
        beta_pred = beta
        P_pred = P + Q

        # 新息及其协方差
        y_hat = H @ beta_pred
        S = H @ P_pred @ H.T + R  # 标量

        # Kalman 增益
        K = P_pred @ H.T / S  # (2×1)

        # 更新步
        innovation = price_a.values[t] - y_hat[0]
        beta = beta_pred + K[:, 0] * innovation
        P = (np.eye(2) - K @ H) @ P_pred

        betas[t] = beta
        variances[t] = S[0, 0]

        # 在线估计 R（EM-like：用滑动窗口均方残差）
        if t > 0:
            R = float(np.mean(variances[max(0, t - 60) : t + 1]))

    idx = price_a.index
    hedge_ratio = pd.Series(betas[:, 0], index=idx, name="hedge_ratio")
    intercept = pd.Series(betas[:, 1], index=idx, name="intercept")
    spread = price_a - hedge_ratio * price_b - intercept
    variance = pd.Series(variances, index=idx, name="variance")

    return KalmanResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
        variance=variance,
        R=float(R),
        Q_scale=delta,
    )


def kalman_zscore(result: KalmanResult, window: int = 20) -> pd.Series:
    """用滚动标准差对 KF 价差做标准化。

    两种标准化方式：
    - 用 variance（KF 给出的预测不确定度）直接除
    - 用滚动窗口 std（更保守，本函数采用）
    """
    std = result.spread.rolling(window).std()
    return (result.spread / std).rename("kf_zscore")
