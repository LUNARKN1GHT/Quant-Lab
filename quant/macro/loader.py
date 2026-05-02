"""宏观数据加载器：从 akshare 拉取并本地 CSV 缓存

data/macro/ 目录下每个指标存储为独立 CSV，避免重复调用 akshare API。
no_proxy 设置是为了解决部分环境下 akshare 请求被代理拦截的问题。
"""

import os
from pathlib import Path

import akshare as ak
import pandas as pd

# 部分环境下系统代理会拦截 akshare 的国内 HTTP 请求，设置 no_proxy 绕过
os.environ["no_proxy"] = "*"

MACRO_DIR = Path(__file__).parent.parent.parent / "data/macro"
MACRO_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_fetch(name: str, fetch_fn) -> pd.Series:
    """通用缓存加载器：CSV 存在则直接读取，否则调用 fetch_fn 拉取并保存。"""
    path = MACRO_DIR / f"{name}.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df.iloc[:, 0]
    s = fetch_fn()
    s.to_csv(path, header=True)
    return s


def load_bond_yield() -> pd.Series:
    """十年期国债收益率"""

    def fetch():
        df = ak.bond_zh_us_rate(start_date="20050101")
        df = df[["日期", "中国国债收益率10年"]].dropna()
        df["日期"] = pd.to_datetime(df["日期"])
        return df.set_index("日期")["中国国债收益率10年"].rename("bond_yield")

    return _load_or_fetch("bond_yield", fetch)


def load_pmi() -> pd.Series:
    """制造业 PMI"""

    def fetch():
        df = ak.macro_china_pmi_yearly()
        df = df[df["商品"] == "中国官方制造业PMI"][["日期", "今值"]].dropna()
        df["日期"] = pd.to_datetime(df["日期"])
        return df.set_index("日期")["今值"].rename("pmi").astype(float)

    return _load_or_fetch("pmi", fetch)


def load_m2() -> pd.Series:
    """M2 同比增速"""

    def fetch():
        df = ak.macro_china_money_supply()
        df = df[["月份", "货币和准货币(M2)-同比增长"]].dropna()
        df["月份"] = pd.to_datetime(
            df["月份"].str.replace("年", "-").str.replace("月份", ""),
            format="%Y-%m",
        )
        return (
            df.set_index("月份")["货币和准货币(M2)-同比增长"]
            .rename("m2_yoy")
            .astype(float)
        )

    return _load_or_fetch("m2", fetch)


def load_cpi() -> pd.Series:
    """CPI 同比"""

    def fetch():
        df = ak.macro_china_cpi_yearly()
        df = df[df["商品"] == "中国CPI年率报告"][["日期", "今值"]].dropna()
        df["日期"] = pd.to_datetime(df["日期"])
        return df.set_index("日期")["今值"].rename("cpi_yoy").astype(float)

    return _load_or_fetch("cpi", fetch)


def load_all_macro() -> pd.DataFrame:
    """加载全部宏观指标，按月对齐，向前填充"""
    series = {
        "bond_yield": load_bond_yield(),
        "pmi": load_pmi(),
        "m2_yoy": load_m2(),
        "cpi_yoy": load_cpi(),
    }
    df = pd.DataFrame(series)
    df = df.resample("ME").last().ffill()
    df = df.dropna()  # 只保留四个指标都有数据的月份
    return df
