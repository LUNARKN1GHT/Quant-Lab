import time
from datetime import datetime

import pandas as pd

from quant.backtest.engine import backtest
from quant.data.base import AKShareAdapter
from quant.data.cache import CachedFetcher
from quant.factor.ic import calc_ic
from quant.factor.momentum import momentum
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select


def run_pipeline(symbols: list[str], start: datetime, end: datetime) -> dict:
    fetcher = CachedFetcher(AKShareAdapter())

    # 拉取所有股票收盘价，拼成一张宽表
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
        time.sleep(1)  # 每只股票时间等待 1 秒

    close = pd.DataFrame(close_dict).dropna()

    # 计算动量因子
    factor = close.apply(lambda col: momentum(col, window=20))

    # 选出前两只股票，构建等权仓位
    returns = close.pct_change().dropna()
    position = factor.apply(lambda row: factor_select(row.dropna(), top_n=2), axis=1)
    position = position.reindex(returns.index).ffill().fillna(0)

    # 回测
    strategy_returns = backtest(position, returns)

    # 计算风险指标
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
