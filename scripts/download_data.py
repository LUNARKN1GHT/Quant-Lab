"""沪深300成分股日频行情下载脚本

数据来源：Tushare Pro（需在 .env 中配置 TUSHARE_TOKEN）
存储格式：data/csi300/{symbol}.csv，每只股票一个文件，列名为 trade_date/close 等。

增量更新策略（update_symbol 内部四分支）：
1. 文件不存在或为空 → 全量下载（从 20190101 起）
2. 文件存在但无法解析（损坏）→ 删除后全量重下
3. 文件为旧格式（中文列名，akshare 历史遗留）→ 删除后全量重下统一为新格式
4. 文件为新格式 → 只追加最后日期之后的增量行
"""

import os
import sys
import time
from pathlib import Path

import akshare as ak
import tushare as ts
from dotenv import load_dotenv

load_dotenv()
# 绕过可能拦截 Tushare/akshare 请求的系统代理
os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

TUSHARE_TOKEN = os.environ["TUSHARE_TOKEN"]
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

DATA_DIR = Path("data/csi300")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def to_ts_code(symbol: str) -> str:
    """将纯数字股票代码转为 Tushare 格式，6 开头为沪市（SH），其余为深市（SZ）。

    示例：000001 → 000001.SZ，600519 → 600519.SH
    """
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def get_csi300_symbols() -> list[str]:
    """通过 akshare 获取当前沪深300成分股列表，返回纯6位数字代码。"""
    df = ak.index_stock_cons(symbol="000300")
    return df["品种代码"].tolist()


def download_symbol(symbol: str, start: str, end: str) -> bool:
    """全量下载单只股票日频行情并保存为 CSV。

    若文件已存在则直接跳过（幂等），适合首次批量下载场景。
    """
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
    """增量更新单只股票数据，返回操作结果标识。

    Returns:
        "new"     — 文件不存在，全量下载成功
        "updated" — 文件存在，追加了新数据
        "skip"    — 数据已最新，无需更新
        "fail"    — 下载或解析失败
    """
    import pandas as pd

    cache_file = DATA_DIR / f"{symbol}.csv"

    # 分支一：文件不存在或为空，全量重下
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        cache_file.unlink(missing_ok=True)
        ok = download_symbol(symbol, "20190101", end)
        return "new" if ok else "fail"

    try:
        try:
            # nrows=0 只读列名，快速判断文件格式
            header = pd.read_csv(cache_file, nrows=0).columns.tolist()
        except Exception:
            # 分支二：文件损坏无法解析，删除后全量重下
            cache_file.unlink(missing_ok=True)
            ok = download_symbol(symbol, "20190101", end)
            return "new" if ok else "fail"

        if "trade_date" not in header:
            # 分支三：旧格式（中文列名），删除后全量重下统一为 Tushare 新格式
            cache_file.unlink()
            ok = download_symbol(symbol, "20190101", end)
            return "new" if ok else "fail"

        # 分支四：新格式文件，读取最后一条日期判断是否需要续下
        existing = pd.read_csv(cache_file, usecols=["trade_date"])
        last_date = str(existing["trade_date"].max())
        if last_date >= end:
            return "skip"

        ts_code = to_ts_code(symbol)
        new_df = pro.daily(ts_code=ts_code, start_date=last_date, end_date=end)
        if new_df is None or new_df.empty:
            return "skip"

        # 过滤掉 last_date 当天（文件中已存在），只追加更新的行
        new_df = new_df[new_df["trade_date"] > last_date].sort_values("trade_date")
        if new_df.empty:
            return "skip"

        # mode="a" + header=False：追加写入，不重复写列名行
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
