import pandas as pd


def check_price_quality(df: pd.DataFrame) -> dict:
    """返回价格数据质量报告"""
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
    report["total_rows"] = len(df)
    report["date_range"] = (str(df.index.min()), str(df.index.max()))
    return report


def check_valuation_quality(df: pd.DataFrame) -> dict:
    """返回估值数据质量报告"""
    report: dict = {}
    report["null_counts"] = df.isnull().sum().to_dict()
    if "pe_ttm" in df.columns:
        # PE 为负表示亏损，不是错误，但值得统计
        report["negative_pe_days"] = int((df["pe_ttm"] < 0).sum())
    if "total_mv" in df.columns:
        report["zero_mv_days"] = int((df["total_mv"] <= 0).sum())
    report["total_rows"] = len(df)
    report["date_range"] = (str(df.index.min()), str(df.index.max()))
    return report


def check_fundamental_quality(df: pd.DataFrame) -> dict:
    """返回财务数据质量报告"""
    report: dict = {}
    report["null_counts"] = df.isnull().sum().to_dict()
    report["total_quarters"] = len(df)
    if "report_date" in df.columns:
        report["date_range"] = (
            str(df["report_date"].min()),
            str(df["report_date"].max()),
        )
    # 检测连续两期数值完全相同（可能是数据源未更新）
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols and len(df) > 1:
        stale = df[numeric_cols].eq(df[numeric_cols].shift()).all(axis=1).sum()
        report["stale_quarters"] = int(stale)
    return report
