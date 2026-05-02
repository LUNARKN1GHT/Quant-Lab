# scripts/research_fund_flow_factor.py
"""主力资金趋势因子 IC 验证"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.factor.fund_flow import fund_flow_momentum
from quant.factor.layered import layered_return

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

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
    return fwd.stack().rename("fwd_ret").reset_index()  # type: ignore


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


def layered_backtest(
    flow: pd.DataFrame, fwd_df: pd.DataFrame, window: int, n_groups: int = 5
):
    """对指定窗口因子做分层回测，画累计收益曲线"""
    col = f"factor_{window}"
    factor_df = flow[["date", "symbol", col]].rename(columns={col: "factor"})
    merged = factor_df.merge(fwd_df, on=["date", "symbol"]).dropna()

    # 每个截面日期分层，记录各组当期收益
    records = []
    for date, grp in merged.groupby("date"):
        if len(grp) < n_groups * 5:
            continue
        try:
            layer = layered_return(
                grp.set_index("symbol")["factor"],
                grp.set_index("symbol")["fwd_ret"],
                n_groups=n_groups,
            )
            for g, ret in layer.items():
                records.append({"date": date, "group": f"G{g}", "ret": ret})
        except Exception:
            continue

    df = pd.DataFrame(records)
    pivot = df.pivot(index="date", columns="group", values="ret").sort_index()

    # 平均分层收益柱状图
    mean_ret = pivot.mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    mean_ret.plot(
        kind="bar",
        ax=axes[0],
        color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"],
    )
    axes[0].set_title(f"各层平均收益（window={window}d, fwd={FORWARD_DAYS}d）")
    axes[0].set_xlabel("因子分层（G1最低 G5最高）")
    axes[0].set_ylabel("平均收益率")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].tick_params(axis="x", rotation=0)

    # 多空累计收益（G5 - G1）
    ls = (pivot["G5"] - pivot["G1"]).cumsum()
    ls.plot(ax=axes[1], color="#1f77b4")
    axes[1].set_title("多空累计收益（G5 - G1）")
    axes[1].set_ylabel("累计收益率")
    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = f"output/factor_research/factor_layered_fund_flow_w{window}.png"
    plt.savefig(out, dpi=150)
    print(f"图表已保存: {out}")

    # 打印各层平均收益
    print(f"\n=== 分层平均收益（window={window}d）===")
    for g, r in mean_ret.items():
        bar = "█" * int(abs(r) * 500)
        sign = "+" if r > 0 else ""
        print(f"  {g}: {sign}{r:.4f}  {bar}")
    spread = mean_ret["G5"] - mean_ret["G1"]
    print(f"  多空价差 G5-G1: {spread:+.4f}")


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
            f"\nwindow={w:2d}d. "
            + f"IC均值={ic_series.mean():.4f}. "
            + f"ICIR={icir:.3f}. "
            + f"IC>0占比={results[w]['IC>0占比']:.1%}. "
        )

    best_w = max(results, key=lambda w: abs(results[w]["ICIR"]))
    print(f"\n最佳窗口: {best_w} 天（ICIR={results[best_w]['ICIR']:.3f}）")

    # 对最佳窗口做分层回测
    layered_backtest(flow, fwd_df, window=best_w)


if __name__ == "__main__":
    main()
