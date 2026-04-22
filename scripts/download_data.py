import os
import sys
import time
from pathlib import Path

import akshare as ak
import tushare as ts
from dotenv import load_dotenv

load_dotenv()
os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

TUSHARE_TOKEN = os.environ["TUSHARE_TOKEN"]
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

DATA_DIR = Path("data/csi300")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def to_ts_code(symbol: str) -> str:
    """000001 → 000001.SZ，600519 → 600519.SH"""
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def get_csi300_symbols() -> list[str]:
    df = ak.index_stock_cons(symbol="000300")
    return df["品种代码"].tolist()


def download_symbol(symbol: str, start: str, end: str) -> bool:
    """全量下载（文件不存在时）"""
    cache_file = DATA_DIR / f"{symbol}.csv"
    if cache_file.exists():
        return True
    try:
        ts_code = to_ts_code(symbol)
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            return False
        df = df.sort_values("trade_date")
        df.to_csv(cache_file, index=False)
        return True
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
        return False


def update_symbol(symbol: str, end: str) -> str:
    """增量更新：从已有数据的最后日期续下"""
    import pandas as pd

    cache_file = DATA_DIR / f"{symbol}.csv"
    if not cache_file.exists():
        ok = download_symbol(symbol, "20190101", end)
        return "new" if ok else "fail"

    try:
        existing = pd.read_csv(cache_file, usecols=["trade_date"])
        last_date = str(existing["trade_date"].max())
        if last_date >= end:
            return "skip"

        ts_code = to_ts_code(symbol)
        new_df = pro.daily(ts_code=ts_code, start_date=last_date, end_date=end)
        if new_df is None or new_df.empty:
            return "skip"

        new_df = new_df[new_df["trade_date"] > last_date].sort_values("trade_date")
        if new_df.empty:
            return "skip"

        new_df.to_csv(cache_file, mode="a", header=False, index=False)
        return "updated"
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
        return "fail"


if __name__ == "__main__":
    import datetime
    end = datetime.date.today().strftime("%Y%m%d")
    symbols = get_csi300_symbols()
    print(f"共 {len(symbols)} 只股票，增量更新至 {end}...")

    counts: dict[str, int] = {"new": 0, "updated": 0, "skip": 0, "fail": 0}
    for i, symbol in enumerate(symbols, 1):
        result = update_symbol(symbol, end)
        counts[result] += 1
        print(f"[{i}/{len(symbols)}] {symbol} {result}")
        time.sleep(0.5)

    print(f"\n完成：新增 {counts['new']}，更新 {counts['updated']}，"
          f"已最新 {counts['skip']}，失败 {counts['fail']}")
