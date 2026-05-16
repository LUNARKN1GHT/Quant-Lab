"""基金风格归因：将基金 NAV 收益回归到基准指数，输出 Beta 暴露与 Alpha"""

import pandas as pd

from quant.fund.portfolio import style_attribution


def fund_style_report(
    nav_series: pd.Series,
    benchmarks: pd.DataFrame,
    name: str = "",
) -> dict:
    """对单只基金做风格回归，返回结构化结果。

    Args:
        nav_series: 基金净值序列（非收益率）
        benchmarks: 基准日收益率宽表，由 quant.data.benchmark.load_benchmarks() 提供
        name: 基金名称，仅用于结果展示

    Returns:
        含 {基准名} Beta / Alpha（年化） / R² / name 的字典；
        数据不足 60 日时返回 {"name": name, "error": "数据不足"}
    """
    ret = nav_series.pct_change().dropna()
    attr = style_attribution(ret, benchmarks)
    if not attr:
        return {"name": name, "error": "数据不足（需 ≥ 60 个交易日）"}

    result: dict = {"name": name}
    for col in benchmarks.columns:
        result[f"{col} Beta"] = attr.get(col, float("nan"))
    result["Alpha（年化）"] = attr.get("Alpha（日）", float("nan")) * 252
    result["R²"] = attr.get("R²", float("nan"))
    return result
