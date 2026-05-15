"""组合优化模块：MVO / Risk Parity / Equal Weight

所有方法统一接口：输入年化 mu 和 cov，返回权重向量 w（sum=1）。
"""

from typing import Literal

import numpy as np
from scipy.optimize import minimize


def _equal_weight(n: int) -> np.ndarray:
    """等权组合：所有资产权重 1/n"""
    return np.ones(n) / n


def _mvo(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    w_min: float,
    w_max: float,
    sector_constraints: list[tuple[np.ndarray, float, float]] | None = None,
) -> np.ndarray:
    """均值方差优化：min ½λ·w'Σw - μ'w，subject to Σw=1, w_min≤w≤w_max。

    显式提供梯度避免 SLSQP 数值梯度异常。
    """
    n = len(mu)

    def neg_utility(w: np.ndarray) -> float:
        return float(0.5 * risk_aversion * w @ cov @ w - mu @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return risk_aversion * (cov @ w) - mu

    cons: list[dict] = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    for vec, lo, hi in sector_constraints or []:
        cons.append({"type": "ineq", "fun": lambda w, v=vec, b=hi: float(b - v @ w)})
        cons.append({"type": "ineq", "fun": lambda w, v=vec, b=lo: float(v @ w - b)})

    result = minimize(  # type: ignore
        neg_utility,
        x0=np.ones(n) / n,
        jac=gradient,
        method="SLSQP",
        bounds=[(w_min, w_max)] * n,
        constraints=cons,  # type: ignore
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return np.asarray(result.x, dtype=float)


def _risk_parity(
    cov: np.ndarray,
    w_min: float,
    w_max: float,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """风险平价：Spinu (2013) 循环坐标下降。

    每次对单个资产求二次方程根，迭代到风险贡献均衡。比 SLSQP 稳定且无导数问题。
    每个资产风险贡献 RC_i = w_i·(Σw)_i，目标 RC_i = const = 1/n（归一化后）。
    """
    n = cov.shape[0]
    target = 1.0 / n
    w = np.ones(n) / n

    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # 推导：w_i·(Σw)_i = target·(w'Σw)
            # 展开 (Σw)_i = Σ_j cov[i,j]·w_j = cov[i,i]·w_i + Σ_{j≠i} cov[i,j]·w_j
            # 令 a = Σ_{j≠i} cov[i,j]·w_j，得二次方程：
            # cov[i,i]·w_i² + a·w_i - target·(w'Σw) = 0
            cov_w = cov @ w
            a = cov_w[i] - cov[i, i] * w[i]
            c = target * float(w @ cov_w)
            disc = a * a + 4 * cov[i, i] * c
            if cov[i, i] <= 0 or disc < 0:
                continue
            w[i] = (-a + np.sqrt(disc)) / (2 * cov[i, i])

        w = w / w.sum()
        w = np.clip(w, w_min, w_max)
        s = w.sum()
        if s > 0:
            w = w / s

        if np.max(np.abs(w - w_prev)) < tol:
            break

    return w


def optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    method: Literal["mvo", "risk_parity", "equal_weight"] = "mvo",
    risk_aversion: float = 1.0,
    w_min: float = 0.0,
    w_max: float = 1.0,
    sector_constraints: list[tuple[np.ndarray, float, float]] | None = None,
) -> np.ndarray:
    """统一入口：返回归一化权重向量。

    Args:
        mu: 年化预期收益向量，shape (n,)
        cov: 年化协方差矩阵，shape (n, n)
        method: 优化方法
        risk_aversion: 风险厌恶系数（仅 mvo 使用）
        w_min: 单资产权重下限
        w_max: 单资产权重上限

    Returns:
        权重向量，shape (n,)，sum=1
    """
    mu_arr = np.asarray(mu, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)
    n = len(mu_arr)

    if cov_arr.shape != (n, n):
        raise ValueError(f"cov 形状 {cov_arr.shape} 与 mu 长度 {n} 不匹配")

    if method == "equal_weight":
        return _equal_weight(n)
    if method == "mvo":
        return _mvo(mu_arr, cov_arr, risk_aversion, w_min, w_max, sector_constraints)
    return _risk_parity(cov_arr, w_min, w_max)


def portfolio_stats(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float = 0.02,
) -> dict[str, float]:
    """计算组合统计指标（年化）。

    Returns:
        包含 return / volatility / sharpe 的字典
    """
    w_arr = np.asarray(w, dtype=float).flatten()
    mu_arr = np.asarray(mu, dtype=float).flatten()
    cov_arr = np.asarray(cov, dtype=float)

    port_return = float(w_arr @ mu_arr)
    port_vol = float(np.sqrt(w_arr @ cov_arr @ w_arr))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0
    return {"return": port_return, "volatility": port_vol, "sharpe": sharpe}


def black_litterman(
    cov: np.ndarray,
    w_market: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float = 0.05,
    Omega: np.ndarray | None = None,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """Black-Litterman 后验预期收益

    Args:
        cov (np.ndarray): 年化协方差矩阵
        w_market (np.ndarray): 市值权重向量
        P (np.ndarray): 观点矩阵，每行一个观点
        Q (np.ndarray): 观点收益向量
        tau (float, optional): 先验不确定系数. Defaults to 0.05.
        Omega (np.ndarray | None, optional): 观点误差协方差. Defaults to None.
        risk_aversion (float, optional): 市场隐含风险厌恶系数. Defaults to 2.5.

    Returns:
        np.ndarray: 后验年化预期收益向量
    """
    cov_arr = np.asarray(cov, dtype=float)
    w_mkt = np.asarray(w_market, dtype=float).flatten()
    P_arr = np.asarray(P, dtype=float)
    Q_arr = np.asarray(Q, dtype=float).flatten()

    # 均衡隐含收益 π = λ Σ w_mkt
    pi = risk_aversion * cov_arr @ w_mkt

    # 观点误差协方差：默认 Ω = diag(P (τΣ) P^T)，与观点方差成比例
    tau_cov = tau * cov_arr
    if Omega is None:
        Omega = np.diag(np.diag(P_arr @ tau_cov @ P_arr.T))

    # BL 后验公式（矩阵形式）
    # μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1} π + P^T Ω^{-1} Q]
    tau_cov_inv = np.linalg.inv(tau_cov)
    omega_inv = np.linalg.inv(Omega)  # type:ignore

    M = tau_cov_inv + P_arr.T @ omega_inv @ P_arr
    rhs = tau_cov_inv @ pi + P_arr.T @ omega_inv @ Q_arr
    mu_bl = np.linalg.solve(M, rhs)
    return mu_bl
