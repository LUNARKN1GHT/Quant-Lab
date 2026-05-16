"""基准指数数据加载器：从 akshare 拉取并本地 CSV 缓存

data/benchmark/ 目录下每个指数存储为独立 CSV。
benchmark 指数被多个模块使用（基金风格归因、Beta/Alpha 计算等），
故归入数据层，避免在业务模块内散落 akshare 调用。
"""

import os
from pathlib import Path

import akshare as ak
import pandas as pd

os.environ["no_proxy"] = "*"

BENCHMARK_DIR = Path(__file__).parent.parent.parent / "data/benchmark"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# 新浪源符号前缀：sh = 沪市，sz = 深市
_BENCHMARKS: dict[str, str] = {
    "CSI300": "sh000300",
    "CSI500": "sh000905",
}


def _fetch_index_returns(name: str, symbol: str) -> pd.Series:
    """从新浪拉取指数日收盘价，转换为日收益率"""
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["close"].pct_change().dropna().rename(name)


def _load_or_fetch(name: str, symbol: str) -> pd.Series:
    path = BENCHMARK_DIR / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0].rename(name)
    s = _fetch_index_returns(name, symbol)
    s.to_csv(path, header=True)
    return s


def load_benchmark(name: str) -> pd.Series:
    if name not in _BENCHMARKS:
        raise ValueError(f"未注册的基准：{name}，可选 {list(_BENCHMARKS)}")
    return _load_or_fetch(name, _BENCHMARKS[name])


def load_benchmarks(names: list[str] | None = None) -> pd.DataFrame:
    target = names or list(_BENCHMARKS.keys())
    return pd.DataFrame({n: load_benchmark(n) for n in target}).dropna()
