import pandas as pd


def brinson(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Brinson 贡献分析

    Args:
        portfolio_weights (pd.Series): 组合权重
        benchmark_weights (pd.Series): 基准权重
        portfolio_returns (pd.Series): 组合各行业收益
        benchmark_returns (pd.Series): 基准各行业收益

    Returns:
        pd.DataFrame: 列名为 `["allocation", "selection", "interaction", "total"]`
    """

    # 基准整体收益
    R_b = (benchmark_weights * benchmark_returns).sum()

    allocation = (portfolio_weights - benchmark_weights) * (benchmark_returns - R_b)
    selection = benchmark_weights * (portfolio_returns - benchmark_returns)
    interaction = (portfolio_weights - benchmark_weights) * (
        portfolio_returns - benchmark_returns
    )

    return pd.DataFrame(
        {
            "allocation": allocation,
            "selection": selection,
            "interaction": interaction,
            "total": allocation + selection + interaction,
        }
    )
