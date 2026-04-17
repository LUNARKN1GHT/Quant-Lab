import pandas as pd

from quant.risk.metrics import calmar, cvar, max_drawdown, sharpe, sortino, var


def compare_strategies(strategy_returns: dict[str, pd.Series]) -> pd.DataFrame:
    """比较各个策略"""
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
