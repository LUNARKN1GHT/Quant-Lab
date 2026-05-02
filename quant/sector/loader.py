"""申万行业数据加载与 CSV 本地缓存"""

import os
from pathlib import Path

import akshare as ak
import pandas as pd

os.environ["no_proxy"] = "*"

# 各行业历史行情存储目录，每个行业一个 CSV 文件
SECTOR_DIR = Path(__file__).parent.parent.parent / "data/sectors"
SECTOR_DIR.mkdir(parents=True, exist_ok=True)


def get_sector_names() -> dict[str, str]:
    """获取申万一级行业代码与名称的映射，返回 {代码: 行业名称}。

    akshare 返回的代码带 .SI 后缀，此处去掉以匹配 index_hist_sw 的参数格式。
    """
    info = ak.sw_index_first_info()
    return {
        row["行业代码"].replace(".SI", ""): row["行业名称"]
        for _, row in info.iterrows()
    }


def fetch_sector_close(code: str) -> pd.Series:
    """拉取单个申万行业的历史日频收盘价，命中 CSV 缓存则直接读取。"""
    cache = SECTOR_DIR / f"{code}.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col="日期", parse_dates=True)
        return df["收盘"]
    df = ak.index_hist_sw(symbol=code, period="day")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.set_index("日期").sort_index()
    df[["收盘"]].to_csv(cache)
    return df["收盘"]


def load_sector_close(start: str = "20190101") -> pd.DataFrame:
    """批量加载所有申万一级行业收盘价，返回日期×行业名称的宽表。

    单个行业拉取失败时跳过（akshare 偶发超时），不中断整体加载。
    """
    names = get_sector_names()
    frames = {}
    for code, name in names.items():
        try:
            s = fetch_sector_close(code)
            frames[name] = s[s.index >= pd.Timestamp(start)]
        except Exception:
            continue
    return pd.DataFrame(frames).sort_index()
