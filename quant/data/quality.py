import pandas as pd


def check_price_quality(df: pd.DataFrame) -> dict:
    """返回数据质量报告"""
    # 报告字典初始化与定义
    report: dict = {}

    # 1. 缺失值，每列有多少个 NaN
    report["null_counts"] = df.isnull().sum().to_dict()

    # 2. 价格异常
    for col in ["open", "close"]:
        if col in df.columns:
            report[f"{col}_invalid"] = int((df[col] <= 0).sum())

    # 3. 成交量为 0
    if "volume" in df.columns:
        report["zero_volume_days"] = int((df["volume"] == 0).sum())

    # 价格逻辑矛盾
    if "high" in df.columns and "low" in df.columns:
        report["high_lt_low"] = int((df["high"] < df["low"]).sum())

    # 5. 重复日期
    report["duplicate_dates"] = int(df.index.duplicated().sum())

    # 6. 总体摘要
    report["total_rows"] = len(df)
    report["date_range"] = (str(df.index.min()), str(df.index.max()))

    return report
