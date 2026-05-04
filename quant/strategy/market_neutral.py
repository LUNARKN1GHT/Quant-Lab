"""市场中性组合构建

将有效因子（主力资金动量）的多空组合对冲掉系统风险：
  - Beta 中性：多空两腿加权后 Σw_i * β_i ≈ 0
  - 行业中性：因子在行业内去均值，消除行业暴露
"""

import numpy as np
import pandas as pd


def compute_rolling_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """用滚动 OLS 计算每只股票相对市场的 Beta。

    Args:
        stock_returns:  日收益率矩阵，index=date, columns=symbol
        market_returns: 市场基准日收益率序列
        window:         滚动窗口

    Returns:
        Beta 矩阵，shape 同 stock_returns
    """
    mkt_var = market_returns.rolling(window).var()
    # rolling().cov() 返回与 stock_returns 同 shape 的矩阵
    cov = stock_returns.rolling(window).cov(market_returns)
    betas = cov.div(mkt_var, axis=0)
    return betas


def sector_neutralize(
    factor: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """行业内因子去均值，消除行业暴露。

    Args:
        factor:     因子矩阵，index=date, columns=symbol
        sector_map: {symbol: sector_label}

    Returns:
        去均值后的因子矩阵
    """
    sector_series = pd.Series(sector_map)
    result = factor.copy()
    for date in factor.index:
        row = factor.loc[date].dropna()
        if row.empty:
            continue
        secs = sector_series.reindex(row.index).fillna("unknown")
        demeaned = row - row.groupby(secs).transform("mean")
        result.loc[date, demeaned.index] = demeaned.values
    return result


def build_portfolio(
    factor_scores: pd.Series,
    betas: pd.Series,
    n_long: int = 20,
    n_short: int = 20,
    beta_neutral: bool = True,
) -> dict[str, float]:
    """构建 Beta 中性多空组合，返回各股票权重。

    多头选因子得分最高的 n_long 只，空头选最低的 n_short 只。
    通过缩放空头规模使组合 Beta ≈ 0。

    Returns:
        {symbol: weight}，多头为正，空头为负，gross exposure = 1
    """
    valid = factor_scores.dropna()
    valid_beta = betas.reindex(valid.index).dropna()
    valid = valid.reindex(valid_beta.index)

    if len(valid) < n_long + n_short:
        return {}

    sorted_f = valid.sort_values(ascending=False)
    long_stocks = sorted_f.head(n_long).index.tolist()
    short_stocks = sorted_f.tail(n_short).index.tolist()

    beta_long = valid_beta[long_stocks].mean()
    beta_short = valid_beta[short_stocks].mean()

    # 缩放空头使 Beta 中性：long_size * β_L = short_size * β_S
    if not beta_neutral:
        scale = 1.0
    elif abs(beta_short) < 1e-6 or beta_long * beta_short < 0:
        scale = 1.0
    else:
        scale = beta_long / beta_short

    weights = {}
    for s in long_stocks:
        weights[s] = 1.0 / n_long
    for s in short_stocks:
        weights[s] = -scale / n_short

    # 归一化到 gross exposure = 1
    gross = sum(abs(v) for v in weights.values())
    return {k: v / gross for k, v in weights.items()}


def backtest_weights(
    weights_history: list[tuple],  # [(date, {symbol: weight})]
    returns: pd.DataFrame,
) -> pd.Series:
    """根据历史权重序列计算组合日收益。

    Args:
        weights_history: [(date, weights_dict)]，每个 date 对应当日开盘建仓
        returns:         日收益率矩阵

    Returns:
        组合日收益序列
    """
    pnl = []
    for i, (date, weights) in enumerate(weights_history):
        if not weights:
            continue
        idx = returns.index.get_loc(date)
        if idx + 1 >= len(returns):
            continue
        next_ret = returns.iloc[idx + 1]  # 次日收益（T+1 执行）
        port_ret = sum(
            w * next_ret.get(s, np.nan)
            for s, w in weights.items()
            if not np.isnan(next_ret.get(s, np.nan))
        )
        pnl.append({"date": returns.index[idx + 1], "ret": port_ret})

    return pd.DataFrame(pnl).set_index("date")["ret"] if pnl else pd.Series(dtype=float)
