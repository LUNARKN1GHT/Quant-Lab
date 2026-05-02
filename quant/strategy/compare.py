"""多策略绩效对比"""

import pandas as pd

from quant.risk.metrics import calmar, cvar, max_drawdown, sharpe, sortino, var


def compare_strategies(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """汇总多个策略的核心风险指标，输出对比宽表。

    Args:
        strategy_returns: 策略名称 → 日度收益率序列的映射

    Returns:
        DataFrame，行为指标名称，列为策略名称，方便横向比较
    """
    records = {}
    for name, returns in strategy_returns.items():
        records[name] = {
            "Sharpe Ratio": sharpe(returns),
            "Sortino Ratio": sortino(returns),
            "Max Drawdown": max_drawdown(returns),
            "Calmar Ratio": calmar(returns),
            "VaR (95%)": var(returns),
            "CVaR (95%)": cvar(returns),
        }
    return pd.DataFrame(records)
