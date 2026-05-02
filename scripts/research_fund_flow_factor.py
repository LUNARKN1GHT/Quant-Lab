# scripts/research_fund_flow_factor.py
"""主力资金趋势因子 IC 验证"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.factor.fund_flow import fund_flow_momentum

DB_PATH = "data/quant.duckdb"
FORWARD_DAYS = 20  # 预测未来 20 日收益
WINDOWS = [5, 10, 20]


def load_data(con) -> tuple[pd.DataFrame, pd.DataFrame]:
    flow = con.execute("""
        SELECT symbol, date, main_net_pct
        FROM fund_flow_daily
        ORDER BY symbol, date
    """).df()

    price = con.execute("""
        SELECT symbol, date, close
        FROM price_daily
        WHERE adjust = 'qfq'
        ORDER BY symbol, date
    """).df()

    return flow, price


def calc_forward_return(price: pd.DataFrame, n: int) -> pd.DataFrame:
    """计算每只股票未来 n 日收益率"""
    pivot = price.pivot(index="date", columns="symbol", values="close")
    fwd = pivot.shift(-n) / pivot - 1
    return fwd.stack().rename("fwd_ret").reset_index()


def rolling_ic(factor_df: pd.DataFrame, fwd_df: pd.DataFrame, window: int) -> pd.Series:
    """按截面日期滚动计算 IC"""
    merged = factor_df.merge(fwd_df, on=["date", "symbol"])
    ic_list = []
    for date, grp in merged.groupby("date"):
        grp = grp.dropna(subset=["factor", "fwd_ret"])
        if len(grp) < 30:
            continue
        ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
        ic_list.append({"date": date, "ic": ic})
    return pd.DataFrame(ic_list).set_index("date")["ic"]


def main():
    con = duckdb.connect(DB_PATH)
    flow, price = load_data(con)

    print(f"资金流向数据: {flow['date'].min()} ~ {flow['date'].max()}")
    print(f"价格数据行数: {len(price)}")

    fwd_df = calc_forward_return(price, FORWARD_DAYS)

    results = {}
    for w in WINDOWS:
        # 计算因子
        flow[f"factor_{w}"] = flow.groupby("symbol")["main_net_pct"].transform(
            lambda s: fund_flow_momentum(s, w)
        )

        factor_df = flow[["date", "symbol", f"factor_{w}"]].rename(
            columns={f"factor_{w}": "factor"}
        )

        ic_series = rolling_ic(factor_df, fwd_df, w)
        icir = ic_series.mean() / ic_series.std()
        results[w] = {
            "IC均值": ic_series.mean(),
            "ICIR": icir,
            "IC>0占比": (ic_series > 0).mean(),
        }
        print(
            f"\nwindow={w:2d}d  IC均值={ic_series.mean():.4f}  ICIR={icir:.3f}  IC>0占比={results[w]['IC>0占比']:.1%}"
        )

    best_w = max(results, key=lambda w: abs(results[w]["ICIR"]))
    print(f"\n最佳窗口: {best_w} 天（ICIR={results[best_w]['ICIR']:.3f}）")


if __name__ == "__main__":
    main()
