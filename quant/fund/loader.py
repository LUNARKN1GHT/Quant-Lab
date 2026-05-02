"""公募基金净值数据加载器（基于 Tushare Pro）

需要在 .env 文件中设置 TUSHARE_TOKEN，或通过环境变量传入。
Tushare Pro 免费账户对调用频率有限制，批量拉取时如遇限频可加 sleep。
"""

import os

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.environ["TUSHARE_TOKEN"])
_pro = ts.pro_api()  # 模块级单例，避免每次调用重新初始化


def load_fund_nav(fund_code: str) -> pd.Series:
    """拉取单只基金的历史单位净值序列。

    Args:
        fund_code: 基金代码，有无 .OF 后缀均可（函数内部自动补全）

    Returns:
        以交易日为 index、单位净值为值的时序序列
    """
    # Tushare 基金净值接口要求代码带 .OF 后缀
    code = fund_code if fund_code.endswith(".OF") else f"{fund_code}.OF"
    df = _pro.fund_nav(ts_code=code, fields="end_date,unit_nav")
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.set_index("end_date").sort_index()
    return df["unit_nav"].astype(float).rename(fund_code)


def load_funds(fund_codes: list[str]) -> pd.DataFrame:
    """批量拉取多只基金净值，返回 date × fund 宽表。

    单只基金拉取失败时打印错误并跳过，不中断整体流程。
    """
    series = {}
    for code in fund_codes:
        try:
            series[code] = load_fund_nav(code)
            print(f"  {code} ✓")
        except Exception as e:
            print(f"  {code} 失败: {e}")
    return pd.DataFrame(series)
