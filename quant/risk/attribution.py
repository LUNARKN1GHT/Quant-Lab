"""收益归因分析

Brinson 模型将超额收益分解为三部分：
- Allocation（配置效应）：行业权重偏离基准带来的收益
- Selection（选股效应）：同行业内选股优于基准带来的收益
- Interaction（交互效应）：配置偏离与选股超额的联合项
"""

import pandas as pd


def brinson(
    portfolio_weights: pd.Series,
    benchmark_weights: pd.Series,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """Brinson 归因分析（单期截面，按行业分解超额收益）。

    Args:
        portfolio_weights: 组合各行业权重，index 为行业名称，合计为 1
        benchmark_weights: 基准各行业权重，index 与 portfolio_weights 一致
        portfolio_returns: 组合各行业当期收益率
        benchmark_returns: 基准各行业当期收益率

    Returns:
        DataFrame，列为 ["allocation", "selection", "interaction", "total"]，
        行为各行业，total = allocation + selection + interaction
    """
    # 基准整体收益（加权平均）
    R_b = (benchmark_weights * benchmark_returns).sum()

    # 配置效应：超配收益好于基准整体的行业获正贡献
    allocation = (portfolio_weights - benchmark_weights) * (benchmark_returns - R_b)
    # 选股效应：在基准权重下，组合收益超越同行业基准的部分
    selection = benchmark_weights * (portfolio_returns - benchmark_returns)
    # 交互效应：同时超配且选股超额的联合贡献
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
