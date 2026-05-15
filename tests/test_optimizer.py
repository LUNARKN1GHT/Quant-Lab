import numpy as np
import pytest

from quant.portfolio.optimizer import (
    black_litterman,
    optimize,
    portfolio_stats,
)


def _make_mu_cov(n: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.15, n)
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + np.eye(n) * 0.01
    return mu, cov


# ── optimize 基础 ──────────────────────────────────────────────────────────────


def test_equal_weight_sums_to_one():
    mu, cov = _make_mu_cov()
    w = optimize(mu, cov, method="equal_weight")
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w, w[0])


def test_mvo_sums_to_one_and_in_bounds():
    mu, cov = _make_mu_cov()
    w = optimize(mu, cov, method="mvo", w_max=0.6)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.6 + 1e-9).all()


def test_risk_parity_sums_to_one():
    mu, cov = _make_mu_cov()
    w = optimize(mu, cov, method="risk_parity")
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()


def test_optimize_raises_on_shape_mismatch():
    mu = np.array([0.1, 0.2])
    cov = np.eye(3)
    with pytest.raises(ValueError, match="cov 形状"):
        optimize(mu, cov)


# ── 行业暴露约束 ───────────────────────────────────────────────────────────────


def test_sector_constraint_respected():
    """前两只资产为同一行业，暴露上限 50%，MVO 结果应满足约束。"""
    mu, cov = _make_mu_cov(n=4)
    sector_vec = np.array([1.0, 1.0, 0.0, 0.0])
    w = optimize(mu, cov, method="mvo", sector_constraints=[(sector_vec, 0.0, 0.5)])
    assert float(sector_vec @ w) <= 0.5 + 1e-6


def test_sector_constraint_lower_bound():
    """某行业权重下限 30%，结果应不低于该值。"""
    mu, cov = _make_mu_cov(n=3)
    sector_vec = np.array([1.0, 0.0, 0.0])
    w = optimize(mu, cov, method="mvo", sector_constraints=[(sector_vec, 0.3, 1.0)])
    assert float(sector_vec @ w) >= 0.3 - 1e-6


# ── portfolio_stats ────────────────────────────────────────────────────────────


def test_portfolio_stats_keys():
    mu, cov = _make_mu_cov()
    w = optimize(mu, cov, method="equal_weight")
    stats = portfolio_stats(w, mu, cov)
    assert {"return", "volatility", "sharpe"} <= stats.keys()
    assert stats["volatility"] > 0


# ── Black-Litterman ────────────────────────────────────────────────────────────


def _simple_bl_setup():
    mu, cov = _make_mu_cov(n=3)
    w_mkt = np.array([0.5, 0.3, 0.2])
    return cov, w_mkt


def test_bl_returns_correct_shape():
    cov, w_mkt = _simple_bl_setup()
    P = np.array([[1.0, -1.0, 0.0]])  # 相对观点
    Q = np.array([0.05])
    mu_bl = black_litterman(cov, w_mkt, P, Q)
    assert mu_bl.shape == (3,)


def test_bl_view_shifts_return():
    """观点 F0 年化比 F1 高 10%，μ_BL[0] 应比无观点均衡收益更高。"""
    cov, w_mkt = _simple_bl_setup()
    pi = 2.5 * cov @ w_mkt  # 均衡收益

    P = np.array([[1.0, -1.0, 0.0]])
    Q = np.array([0.10])
    mu_bl = black_litterman(cov, w_mkt, P, Q)

    # BL 后验收益应在均衡收益与观点之间（向观点方向偏移）
    assert mu_bl[0] > pi[0]


def test_bl_no_view_close_to_equilibrium():
    """tau 极小、观点极不确定时，μ_BL 应接近均衡收益 π。"""
    cov, w_mkt = _simple_bl_setup()
    pi = 2.5 * cov @ w_mkt

    P = np.eye(3)
    Q = pi.copy()  # 观点等于均衡
    mu_bl = black_litterman(cov, w_mkt, P, Q, tau=0.001)
    assert np.allclose(mu_bl, pi, atol=1e-3)


def test_bl_custom_omega():
    """传入自定义 Omega 不应报错，结果 shape 正确。"""
    cov, w_mkt = _simple_bl_setup()
    P = np.array([[1.0, 0.0, -1.0]])
    Q = np.array([0.03])
    Omega = np.array([[0.01]])
    mu_bl = black_litterman(cov, w_mkt, P, Q, Omega=Omega)
    assert mu_bl.shape == (3,)


def test_bl_then_mvo_valid_weights():
    """BL 输出的 μ_BL 直接传给 MVO，结果应是合法权重。"""
    cov, w_mkt = _simple_bl_setup()
    P = np.array([[1.0, -1.0, 0.0]])
    Q = np.array([0.08])
    mu_bl = black_litterman(cov, w_mkt, P, Q)
    w = optimize(mu_bl, cov, method="mvo", w_max=0.6)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.6 + 1e-9).all()
