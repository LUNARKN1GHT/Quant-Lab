"""统一风险报告模块

risk_report(returns) → 结构化字典，整合 metrics.py 全部指标。
可选传入 market_returns 以计算 beta / alpha。
"""

import pandas as pd

from quant.risk.metrics import (
    alpha,
    beta,
    calmar,
    cvar,
    drawdown_series,
    max_drawdown,
    sharpe,
    sortino,
    underwater_stats,
    var,
)


def risk_report(
    returns: pd.Series,
    market_returns: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict:
    """生成完整风险报告字典

    Args:
        returns (pd.Series): 日度收益率序列
        market_returns (pd.Series | None): 基准日度收益率，传入后计算beta/alpha
        risk_free (float, optional): 日度无风险利率（年化 3% 传 0.03/252）

    Returns:
        dict: 结构化字典，含绩效、风险、尾部、回撤、基准五组指标
    """
    r = returns.dropna()
    report: dict = {
        # 绩效指标
        "annual_return": float(r.mean() * 252),
        "annual_vol": float(r.std() * (252**0.5)),
        "sharpe": sharpe(r, risk_free),
        "sortino": sortino(r, risk_free),
        "calmar": calmar(r),
        # 尾部风险
        "var_95": var(r),
        "cvar_95": cvar(r),
        # 回撤
        "max_drawdown": max_drawdown(r),
        **underwater_stats(r),
        "drawdown_series": drawdown_series(r),
    }

    if market_returns is not None:
        mkt = market_returns.reindex(r.index).dropna()
        aligned = r.reindex(mkt.index).dropna()
        report["beta"] = beta(aligned, mkt)
        report["alpha_annual"] = float(alpha(aligned, mkt, risk_free))

    return report
