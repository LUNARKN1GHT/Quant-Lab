"""数据加载与缓存"""

import os
from pathlib import Path

import akshare as ak
import pandas as pd

os.environ["no_proxy"] = "*"

SECTOR_DIR = Path(__file__).parent.parent.parent / "data/sectors"
SECTOR_DIR.mkdir(parents=True, exist_ok=True)


def get_sector_names() -> dict[str, str]:
    """返回 {代码: 行业名称}，代码不含 .SI"""
    info = ak.sw_index_first_info()
    return {
        row["行业代码"].replace(".SI", ""): row["行业名称"]
        for _, row in info.iterrows()
    }


def fetch_sector_close(code: str) -> pd.Series:
    """拉单个行业历史收盘价，缓存到 CSV"""
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
    """返回宽表，列为行业名称，行为日期"""
    names = get_sector_names()
    frames = {}
    for code, name in names.items():
        try:
            s = fetch_sector_close(code)
            frames[name] = s[s.index >= pd.Timestamp(start)]
        except Exception:
            continue
    return pd.DataFrame(frames).sort_index()
