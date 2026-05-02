"""端到端策略流水线示例

将数据拉取 → 因子计算 → 选股 → 回测 → 风险指标计算串联为一个函数，
用于快速验证单因子策略的整体流程。
"""

import time
from datetime import datetime

import pandas as pd

from quant.backtest.engine import backtest
from quant.data.base import AKShareAdapter
from quant.data.cache import CachedFetcher
from quant.factor.momentum import momentum
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select


def run_pipeline(
    symbols: list[str], start: datetime, end: datetime, fetcher=None
) -> dict:
    """运行动量因子策略的完整流水线。

    Args:
        symbols: 股票代码列表
        start: 回测起始日期
        end: 回测结束日期
        fetcher: 数据拉取器，默认使用带缓存的 AKShare 适配器

    Returns:
        dict，包含 'returns'（日度收益率）、'metrics'（风险指标）、'close'（收盘价宽表）
    """
    if fetcher is None:
        fetcher = CachedFetcher(AKShareAdapter())

    # 逐只股票拉取收盘价，sleep 2s 避免触发 akshare 频率限制
    close_dict = {}
    for symbol in symbols:
        df = fetcher.get_price(
            symbol=symbol,
            period="daily",
            start_time=start,
            end_time=end,
            columns_ask=["close"],
        )
        close_dict[symbol] = df["close"]
        time.sleep(2)

    # dropna 确保所有股票的日期对齐（去掉有任意股票缺失的交易日）
    close = pd.DataFrame(close_dict).dropna()

    # 计算 20 日动量因子
    factor = close.apply(lambda col: momentum(col, window=20))

    # 每日基于因子值选出 Top-2 股票等权持仓
    returns = close.pct_change().dropna()
    position = factor.apply(lambda row: factor_select(row.dropna(), top_n=2), axis=1)
    # reindex + ffill：因子信号比收益率少一天，前向填充补齐
    position = position.reindex(returns.index).ffill().fillna(0)

    # 向量化回测（shift=True，用今日仓位赚明日收益）
    strategy_returns = backtest(position, returns)

    metrics = {
        "Sharpe Ratio": sharpe(strategy_returns),
        "Max Drawdown": max_drawdown(strategy_returns),
        "Calmar Ratio": calmar(strategy_returns),
        "Sortino Ratio": sortino(strategy_returns),
    }

    return {
        "returns": strategy_returns,
        "metrics": metrics,
        "close": close,
    }
