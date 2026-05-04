"""下载持仓基金历史净值，存入 DuckDB fund_nav 表"""

import akshare as ak
import duckdb
import pandas as pd
import yaml

DB_PATH = "data/quant.duckdb"
HOLDINGS_PATH = "configs/my_holdings.yaml"


def fetch_nav(symbol: str) -> pd.DataFrame:
    df = ak.fund_open_fund_info_em(
        symbol=symbol, indicator="单位净值走势", period="成立来"
    )
    df = df.rename(
        columns={"净值日期": "date", "单位净值": "nav", "日增长率": "daily_pct"}
    )
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = symbol
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["daily_pct"] = pd.to_numeric(df["daily_pct"], errors="coerce")
    return df[["symbol", "date", "nav", "daily_pct"]].dropna(subset=["nav"])


def main():
    with open(HOLDINGS_PATH) as f:
        cfg = yaml.safe_load(f)

    symbols = [h["symbol"] for h in cfg["holdings"]]
    con = duckdb.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS fund_nav (
            symbol  VARCHAR,
            date    DATE,
            nav     DOUBLE,
            daily_pct DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)

    for symbol in symbols:
        print(f"下载 {symbol} ...")
        df = fetch_nav(symbol)
        # 增量写入：删除已有数据再插入（简单策略）
        con.execute(f"DELETE FROM fund_nav WHERE symbol = '{symbol}'")
        con.execute("INSERT INTO fund_nav SELECT * FROM df")
        print(
            f"{symbol}: {len(df)} 条，{df['date'].min().date()} ~ {df['date'].max().date()}"
        )

    con.close()
    print("\n完成！")


if __name__ == "__main__":
    main()
