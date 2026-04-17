import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from quant.data.base import YFinanceAdapter
from quant.data.cache import CachedFetcher
from quant.pipeline import run_pipeline
from quant.strategy.compare import compare_strategies

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
fetcher = CachedFetcher(YFinanceAdapter())

bull = run_pipeline(
    SYMBOLS, datetime(2019, 1, 1), datetime(2021, 12, 31), fetcher=fetcher
)
bear = run_pipeline(
    SYMBOLS, datetime(2022, 1, 1), datetime(2024, 12, 31), fetcher=fetcher
)

result = compare_strategies(
    {
        "牛市 2019-2021": bull["returns"],
        "熊市 2022-2024": bear["returns"],
    }
)

print(result.to_string())
