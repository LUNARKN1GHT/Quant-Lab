"""定投计划管理"""

from pathlib import Path

import pandas as pd
import yaml

DCA_PATH = Path("configs/fund_dca_plans.yaml")
_COLS = ["symbol", "name", "amount", "period", "note"]

PERIOD_DAYS = {
    "daily": 1,
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
    "quarterly": 91,
}
PERIOD_LABELS = {
    "daily": "每日",
    "weekly": "每周",
    "biweekly": "每两周",
    "monthly": "每月",
    "quarterly": "每季",
}


def next_trading_day(date: pd.Timestamp, nav_series: pd.Series) -> pd.Timestamp | None:
    """返回 date 当天或之后最近的交易日（净值序列中存在的日期）。

    节假日、周末均无净值，跳过即可。
    """
    candidates = nav_series.dropna().index
    future = candidates[candidates >= date]
    return future[0] if len(future) else None


def load_dca_plans() -> pd.DataFrame:
    empty = pd.DataFrame(columns=_COLS)
    if not DCA_PATH.exists():
        return empty
    try:
        with open(DCA_PATH) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"定投计划文件损坏，请检查 {DCA_PATH}: {e}") from e
    items = data.get("dca_plans", [])
    if not items:
        return empty
    return pd.DataFrame(items)


def save_dca_plans(df: pd.DataFrame) -> None:
    DCA_PATH.parent.mkdir(exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    records = []
    for row_vals in df[_COLS].values:
        r = {}
        for col, val in zip(_COLS, row_vals):
            if hasattr(val, "item"):
                val = val.item()
            r[col] = val
        records.append(r)
    with open(DCA_PATH, "w") as f:
        yaml.dump({"dca_plans": records}, f, allow_unicode=True, sort_keys=False)
