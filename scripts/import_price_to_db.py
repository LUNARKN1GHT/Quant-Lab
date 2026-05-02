# scripts/import_price_to_db.py
"""将 data/csi300/ 下的 CSV 批量导入 price_daily 表"""

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = "data/quant.duckdb"
CSV_DIR = Path("data/csi300")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"trade_date": str})
    symbol = df["ts_code"].iloc[0].split(".")[0]
    df = df.rename(
        columns={
            "trade_date": "date",
            "pct_chg": "change_pct",
            "vol": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    df["symbol"] = symbol
    df["period"] = "daily"
    df["turnover_rate"] = None
    df["adjust"] = "qfq"
    return df[
        [
            "symbol",
            "period",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "change_pct",
            "turnover_rate",
            "adjust",
        ]
    ]


def main():
    con = duckdb.connect(DB_PATH)
    con.execute("DELETE FROM price_daily")

    files = sorted(CSV_DIR.glob("*.csv"))
    print(f"共 {len(files)} 个文件")

    for i, f in enumerate(files, 1):
        df = load_csv(f)
        con.execute("INSERT INTO price_daily SELECT * FROM df")
        if i % 50 == 0:
            print(f"  {i}/{len(files)}")

    total = con.execute("SELECT COUNT(*) FROM price_daily").fetchone()[0]
    print(f"导入完成，price_daily 共 {total:,} 行")


if __name__ == "__main__":
    main()
