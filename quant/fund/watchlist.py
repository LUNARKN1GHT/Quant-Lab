"""自选基金关注列表管理"""

from pathlib import Path

import pandas as pd
import yaml

WATCHLIST_PATH = Path("configs/fund_watchlist.yaml")


def load_watchlist() -> pd.DataFrame:
    empty = pd.DataFrame(columns=["symbol", "name", "added_date", "note"])

    if not WATCHLIST_PATH.exists():
        return empty
    try:
        with open(WATCHLIST_PATH) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"自选列表文件损坏，请检查 {WATCHLIST_PATH}: {e}") from e
    items = data.get("watchlist", [])
    if not items:
        return empty
    df = pd.DataFrame(items)
    df["added_date"] = pd.to_datetime(df["added_date"])
    return df


def save_watchlist(df) -> None:
    WATCHLIST_PATH.parent.mkdir(exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    cols = list(df.columns)
    records = []
    for row_vals in df.values:
        r = {}
        for col, val in zip(cols, row_vals):
            if col == "added_date":
                if hasattr(val, "strftime"):
                    val = val.strftime("%Y-%m-%d")
                else:
                    val = str(val)[:10]
            elif hasattr(val, "item"):
                val = val.item()
            r[col] = val
        records.append(r)
    with open(WATCHLIST_PATH, "w") as f:
        yaml.dump({"watchlist": records}, f, allow_unicode=True, sort_keys=False)
