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
    cache_file = DATA_DIR / f"{symbol}.csv"
    if cache_file.exists():
        return True

    try:
        ts_code = to_ts_code(symbol)
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        df = df.sort_values("trade_date")
        df.to_csv(cache_file, index=False)
        return True
    except Exception as e:
        print(f"  [FAIL] {symbol}: {e}")
        return False


if __name__ == "__main__":
    symbols = get_csi300_symbols()
    print(f"共 {len(symbols)} 只股票，开始下载...")

    success, failed = 0, []
    for i, symbol in enumerate(symbols, 1):
        ok = download_symbol(symbol, "20190101", "20241231")
        if ok:
            success += 1
            print(f"[{i}/{len(symbols)}] {symbol} OK")
        else:
            failed.append(symbol)
        time.sleep(1.5)

    print(f"\n完成：{success} 成功，{len(failed)} 失败")
    if failed:
        print("失败列表：", failed)
