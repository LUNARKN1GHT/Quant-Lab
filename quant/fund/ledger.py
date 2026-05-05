"""基金交易流水管理：持仓从流水自动计算"""

from pathlib import Path

import pandas as pd
import yaml

TRANSACTIONS_PATH = Path("configs/fund_transactions.yaml")


def load_transactions() -> pd.DataFrame:
    empty = pd.DataFrame(
        columns=["id", "symbol", "name", "date", "type", "shares", "nav", "note"]
    )
    if not TRANSACTIONS_PATH.exists():
        return empty
    try:
        with open(TRANSACTIONS_PATH) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"交易流水文件损坏，请检查 {TRANSACTIONS_PATH}: {e}") from e
    txns = data.get("transactions", [])
    if not txns:
        return empty
    df = pd.DataFrame(txns)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def save_transactions(df) -> None:
    """将流水写回 YAML（兼容 DataFrame / session_state 序列化对象）"""
    TRANSACTIONS_PATH.parent.mkdir(exist_ok=True)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    cols = list(df.columns)
    records = []
    # 用 numpy values 避免 pandas 行迭代器的类型依赖
    for row_vals in df.values:
        r = {}
        for col, val in zip(cols, row_vals):
            if col == "date":
                if hasattr(val, "strftime"):
                    val = val.strftime("%Y-%m-%d")
                else:
                    val = str(val)[:10]
            elif hasattr(val, "item"):
                val = val.item()  # numpy scalar → Python 原生类型
            r[col] = val
        records.append(r)
    with open(TRANSACTIONS_PATH, "w") as f:
        yaml.dump({"transactions": records}, f, allow_unicode=True, sort_keys=False)


def next_id(df: pd.DataFrame) -> int:
    if df.empty or "id" not in df.columns:
        return 1
    return int(df["id"].max()) + 1


def compute_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """从流水计算当前持仓"""
    if df.empty:
        return pd.DataFrame(columns=["symbol", "name", "shares", "avg_cost_nav"])

    result = {}
    for _, row in df.iterrows():
        sym = row["symbol"]
        if sym not in result:
            result[sym] = {
                "symbol": sym,
                "name": row["name"],
                "shares": 0.0,
                "cost_total": 0.0,
            }
        if row["type"] == "buy":
            result[sym]["cost_total"] += row["shares"] * row["nav"]
            result[sym]["shares"] += row["shares"]
        elif row["type"] == "sell":
            sell_ratio = (
                min(row["shares"] / result[sym]["shares"], 1.0)
                if result[sym]["shares"] > 0
                else 0
            )
            result[sym]["cost_total"] *= 1 - sell_ratio
            result[sym]["shares"] -= row["shares"]

    rows = []
    for v in result.values():
        if v["shares"] > 1e-4:
            rows.append(
                {
                    "symbol": v["symbol"],
                    "name": v["name"],
                    "shares": v["shares"],
                    "avg_cost_nav": v["cost_total"] / v["shares"]
                    if v["shares"] > 0
                    else 0,
                }
            )
    return pd.DataFrame(rows)


def transaction_returns(df: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """计算每笔买入交易从买入日至今的收益"""
    buys = df[df["type"] == "buy"].copy()
    records = []
    for _, row in buys.iterrows():
        sym = row["symbol"]
        if sym not in nav.columns:
            continue
        nav_series = nav[sym].dropna()
        latest_nav = nav_series.iloc[-1]
        actual_nav = row["nav"]  # 以录入净值为准
        ret = latest_nav / actual_nav - 1
        hold_days = (nav_series.index[-1] - row["date"]).days
        records.append(
            {
                "交易ID": row["id"],
                "基金": row["name"],
                "买入日": row["date"].date(),
                "买入净值": actual_nav,
                "最新净值": latest_nav,
                "持有天数": hold_days,
                "收益率": ret,
                "盈亏金额": row["shares"] * (latest_nav - actual_nav),
                "备注": row.get("note", ""),
            }
        )
    return pd.DataFrame(records)
