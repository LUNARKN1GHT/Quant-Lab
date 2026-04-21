import os

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.environ["TUSHARE_TOKEN"])
_pro = ts.pro_api()


def load_fund_nav(fund_code: str) -> pd.Series:
    """拉取基金单位净值历史"""
    code = fund_code if fund_code.endswith(".OF") else f"{fund_code}.OF"
    df = _pro.fund_nav(ts_code=code, fields="end_date,unit_nav")
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.set_index("end_date").sort_index()
    return df["unit_nav"].astype(float).rename(fund_code)


def load_funds(fund_codes: list[str]) -> pd.DataFrame:
    """批量拉取，返回 date × fund 的净值宽表"""
    series = {}
    for code in fund_codes:
        try:
            series[code] = load_fund_nav(code)
            print(f"  {code} ✓")
        except Exception as e:
            print(f"  {code} 失败: {e}")
    return pd.DataFrame(series)
