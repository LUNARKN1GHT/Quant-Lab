import os
import sys
import time
from pathlib import Path

os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import akshare as ak
import pandas as pd

from quant.backtest.engine import backtest
from quant.factor.momentum import momentum
from quant.strategy.compare import compare_strategies
from quant.strategy.factor_strategy import factor_select

SYMBOLS = ["600519", "600036", "601318", "000333", "000858"]


def run(symbols, start, end):
    cache_file = Path(f"data/{'_'.join(symbols)}_{start.year}_{end.year}.csv")
    if cache_file.exists():
        close = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        close_dict = {}
        for symbol in symbols:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            )
            df["日期"] = pd.to_datetime(df["日期"])
            df = df.set_index("日期")
            close_dict[symbol] = df["收盘"]
            time.sleep(2)
        close = pd.DataFrame(close_dict).dropna()
        cache_file.parent.mkdir(exist_ok=True)
        close.to_csv(cache_file)

    factor = close.apply(lambda col: momentum(col, window=20))
    returns = close.pct_change().dropna()
    position = factor.apply(lambda row: factor_select(row.dropna(), top_n=2), axis=1)
    position = position.reindex(returns.index).ffill().fillna(0)
    return backtest(position, returns)


bull = run(SYMBOLS, datetime(2019, 1, 1), datetime(2021, 12, 31))
print("牛市数据获取完毕，等待 5 秒...")
time.sleep(5)
bear = run(SYMBOLS, datetime(2022, 1, 1), datetime(2024, 12, 31))

result = compare_strategies({
    "牛市 2019-2021": bull,
    "熊市 2022-2024": bear,
})

print(result.to_string())
