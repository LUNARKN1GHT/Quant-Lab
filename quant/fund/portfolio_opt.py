"""基金组合优化：从 NAV 历史估计 μ/Σ → 调用优化器 → 与实盘对账"""

from typing import Literal

import numpy as np
import pandas as pd

from quant.portfolio.optimizer import optimize, portfolio_stats


def estimate_mu_cov(
    nav: pd.DataFrame, lookback: int = 120
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """从基金 NAV 宽表估计年化 μ 和 Σ。

    Returns:
        (mu_annualized, cov_annualized, asset_symbols)
    """
    rets = nav.pct_change().dropna().tail(lookback)
    if len(rets) < 30:
        raise ValueError(f"可用数据点仅 {len(rets)} 条，建议至少 30 条")
    mu = rets.mean().values * 252  # type: ignore
    cov = rets.cov().values * 252  # type: ignore
    return mu, cov, rets.columns.tolist()  # type: ignore


def current_weights(
    holdings: pd.DataFrame, nav: pd.DataFrame
) -> tuple[dict[str, float], float]:
    """根据持仓和最新 NAV 计算实盘权重和总市值。"""
    mkts: dict[str, float] = {}
    for _, h in holdings.iterrows():
        sym = h["symbol"]
        if sym in nav.columns:
            mkts[sym] = float(h["shares"]) * float(nav[sym].dropna().iloc[-1])
        else:
            mkts[sym] = 0.0
    total = sum(mkts.values())
    weights = {k: (v / total if total > 0 else 0.0) for k, v in mkts.items()}
    return weights, total


def run_all_methods(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 2.0,
    w_max: float = 0.5,
) -> dict[str, tuple[np.ndarray, dict[str, float]]]:
    """三种方法批量运行，返回 {方法名: (权重, 统计指标)}"""
    method_map: dict[str, Literal["equal_weight", "mvo", "risk_parity"]] = {
        "等权": "equal_weight",
        "MVO": "mvo",
        "风险平价": "risk_parity",
    }
    results = {}
    for label, method in method_map.items():
        w = optimize(mu, cov, method=method, risk_aversion=risk_aversion, w_max=w_max)
        results[label] = (w, portfolio_stats(w, mu, cov))
    return results


def reconcile(
    assets: list[str],
    target_w: np.ndarray,
    current_w: dict[str, float],
    total_mkt: float,
    name_map: dict[str, str] | None = None,
    rebalance_threshold: float = 0.02,
) -> pd.DataFrame:
    """生成"建议 vs 实盘"对账表。

    Args:
        rebalance_threshold: 偏差金额占总市值低于此比例时视为"持有不动"
    """
    name_map = name_map or {}
    rows = []
    for sym, tgt in zip(assets, target_w):
        cur = current_w.get(sym, 0.0)
        diff_w = float(tgt) - cur
        diff_amt = diff_w * total_mkt
        if abs(diff_amt) < total_mkt * rebalance_threshold:
            action = "⚪ 持有不动"
        elif diff_amt > 0:
            action = "🟢 建议加仓"
        else:
            action = "🔴 建议减仓"
        rows.append(
            {
                "基金": name_map.get(sym, sym),
                "代码": sym,
                "当前权重": cur,
                "建议权重": float(tgt),
                "权重偏差": diff_w,
                "调仓金额": diff_amt,
                "操作": action,
            }
        )
    return pd.DataFrame(rows)
