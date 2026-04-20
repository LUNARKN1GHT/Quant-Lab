"""在沪深300成分股真实数据上运行市场状态检测"""

import os
import sys
from pathlib import Path

os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from quant.config import Config
from quant.regime.detector import detect_regime

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")
DATA_DIR = Path(__file__).parent.parent / cfg.data.data_dir


def load_csi300_close() -> pd.DataFrame:
    """读取 data/csi300/ 下所有 CSV，拼成日期 x 股票的收盘价宽表"""
    series: dict[str, pd.Series] = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        symbol = csv_path.stem  # 文件名即股票代码
        try:
            df = pd.read_csv(csv_path, usecols=["trade_date", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.set_index("trade_date").sort_index()
            series[symbol] = df["close"]
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df.empty:
            continue
    close = pd.DataFrame(series)
    close.index.name = "date"
    return close


def main() -> None:
    close = load_csi300_close()
    print(f"\n数据范围: {close.index[0].date()} ~ {close.index[-1].date()}")
    print(f"股票数量: {close.shape[1]}，交易日: {close.shape[0]}\n")

    regime = detect_regime(close)

    # 按年统计各状态天数占比
    print("=== 各年度市场状态统计 ===")
    regime_df = regime.to_frame("regime")
    regime_df["year"] = regime_df.index.year
    yearly = regime_df.groupby(["year", "regime"]).size().unstack(fill_value=0)
    for col in ["BULL", "RANGE", "BEAR"]:
        if col not in yearly.columns:
            yearly[col] = 0
    yearly = yearly[["BULL", "RANGE", "BEAR"]]
    yearly_pct = (yearly.div(yearly.sum(axis=1), axis=0) * 100).round(1)
    print(yearly_pct.to_string())

    # 最近 60 个交易日的状态变化
    print("\n=== 最近 60 个交易日市场状态 ===")
    recent = regime.tail(60)
    changes = recent[recent != recent.shift()]
    print(changes.to_string())

    # 当前状态
    latest_date = regime.index[-1]
    latest_regime = regime.iloc[-1]
    print(f"\n当前市场状态（{latest_date.date()}）: {latest_regime}")


if __name__ == "__main__":
    main()
