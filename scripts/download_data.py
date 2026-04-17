import os
import sys
import time
from pathlib import Path

os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

import akshare as ak

DATA_DIR = Path("data/csi300")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_csi300_symbols() -> list[str]:
    df = ak.index_stock_cons(symbol="000300")
    return df["品种代码"].tolist()


def download_symbol(symbol: str, start: str, end: str) -> bool:
    cache_file = DATA_DIR / f"{symbol}.csv"
    if cache_file.exists():
        return True  #  已缓存，跳过

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        df.to_csv(cache_file, index=False)
        return True
    except Exception as e:
        print(f" [FAIL] {symbol}: {e}")
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
        time.sleep(10)

    print(f"\n完成：{success} 成功，{len(failed)} 失败")
    if failed:
        print("失败列表：", failed)
