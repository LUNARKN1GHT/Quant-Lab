# scripts/download_enriched_data.py
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import akshare as ak

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.data.base import AKShareAdapter
from quant.data.cache import CachedFetcher

START = datetime(2019, 1, 1)
END = datetime.today()
DB_PATH = "data/quant.duckdb"
SLEEP = random.uniform(0.5, 1.5)  # 每只股票请求间隔（秒）


def get_csi300_symbols() -> list[str]:
    df = ak.index_stock_cons(symbol="000300")
    return df["品种代码"].tolist()


def is_cached(fetcher: CachedFetcher, table: str, symbol: str) -> bool:
    count = fetcher.connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE symbol = ?", [symbol]
    ).fetchone()[0]  # type: ignore
    return count > 0


def run_batch(
    name: str,
    table: str,
    symbols: list[str],
    fetch_fn,
    fetcher: CachedFetcher,  # ← 显式传入
) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {name}  ({len(symbols)} 只)")
    print(f"{'=' * 50}")
    counts = {"ok": 0, "skip": 0, "fail": 0}

    for i, symbol in enumerate(symbols, 1):
        prefix = f"[{i:3d}/{len(symbols)}] {symbol}"

        if is_cached(fetcher, table, symbol):
            counts["skip"] += 1
            print(f"{prefix}  skip")
            continue

        try:
            fetch_fn(symbol)
            counts["ok"] += 1
            print(f"{prefix}  ok")
        except Exception as e:
            counts["fail"] += 1
            print(f"{prefix}  FAIL: {e}")

        time.sleep(SLEEP)

    print(f"\n  下载 {counts['ok']}  已缓存 {counts['skip']}  失败 {counts['fail']}\n")


if __name__ == "__main__":
    symbols = get_csi300_symbols()
    print(f"沪深300成分股：{len(symbols)} 只")
    print(f"数据区间：{START.date()} ~ {END.date()}")
    print(f"写入数据库：{DB_PATH}")

    fetcher = CachedFetcher(AKShareAdapter(), db_path=DB_PATH)

    run_batch(
        name="估值数据  PE / PB / 总市值",
        table="valuation_daily",
        symbols=symbols,
        fetch_fn=lambda s: fetcher.get_valuation(s, START, END),
        fetcher=fetcher,
    )

    run_batch(
        name="财务数据  ROE / ROA / 毛利率等",
        table="fundamental_quarterly",
        symbols=symbols,
        fetch_fn=lambda s: fetcher.get_fundamental(s, start_year="2019"),
        fetcher=fetcher,
    )

    run_batch(
        name="资金流向  主力净流入 / 大单",
        table="fund_flow_daily",
        symbols=symbols,
        fetch_fn=lambda s: fetcher.get_fund_flow(s, START, END),
        fetcher=fetcher,
    )

    print("全部完成。")
