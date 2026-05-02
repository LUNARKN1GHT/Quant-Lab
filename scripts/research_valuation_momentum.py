# scripts/research_valuation_momentum.py
"""估值动量因子 IC 验证"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.factor.layered import layered_return
from quant.factor.valuation_momentum import valuation_momentum

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
FORWARD_DAYS = 20
WINDOWS = [10, 20, 40]
OUT_DIR = Path("output/factor_research")


def load_data(con):
    valuation = con.execute(
        "SELECT symbol, date, pe_ttm FROM valuation_daily ORDER BY symbol, date"
    ).df()
    price = con.execute(
        "SELECT symbol, date, close FROM price_daily WHERE adjust = 'qfq' ORDER BY symbol, date"
    ).df()
    price["date"] = pd.to_datetime(price["date"])
    return valuation, price


def calc_forward_return(price: pd.DataFrame, n: int) -> pd.DataFrame:
    pivot = price.pivot(index="date", columns="symbol", values="close")
    fwd = pivot.shift(-n) / pivot - 1
    return fwd.stack().reset_index().rename(columns={0: "fwd_ret"})


def rolling_ic(factor_long: pd.DataFrame, fwd_df: pd.DataFrame) -> pd.Series:
    merged = factor_long.merge(fwd_df, on=["date", "symbol"])
    ic_list = []
    for date, grp in merged.groupby("date"):
        grp = grp.dropna(subset=["factor", "fwd_ret"])
        if len(grp) < 30:
            continue
        ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
        ic_list.append({"date": date, "ic": ic})
    return pd.DataFrame(ic_list).set_index("date")["ic"]


def run_window(valuation, price, price_dates, symbols, fwd_df, window):
    factor_matrix = valuation_momentum(valuation, price_dates, symbols, window=window)
    factor_long = factor_matrix.stack().reset_index().rename(columns={0: "factor"})
    ic_series = rolling_ic(factor_long, fwd_df)
    icir = ic_series.mean() / ic_series.std()
    print(
        f"window={window:2d}d  IC均值={ic_series.mean():.4f}  ICIR={icir:.3f}  IC>0占比={(ic_series > 0).mean():.1%}"
    )
    return ic_series, factor_long, icir


def layered_plot(factor_long, fwd_df, window, ax_bar, ax_ls):
    merged = factor_long.merge(fwd_df, on=["date", "symbol"]).dropna()
    records = []
    for date, grp in merged.groupby("date"):
        if len(grp) < 25:
            continue
        try:
            layer = layered_return(
                grp.set_index("symbol")["factor"],
                grp.set_index("symbol")["fwd_ret"],
                n_groups=5,
            )
            for g, ret in layer.items():
                records.append({"date": date, "group": f"G{g}", "ret": ret})
        except Exception:
            continue

    pivot = (
        pd.DataFrame(records)
        .pivot(index="date", columns="group", values="ret")
        .sort_index()
    )
    mean_ret = pivot.mean()

    mean_ret.plot(
        kind="bar",
        ax=ax_bar,
        color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"],
    )
    ax_bar.set_title(f"各层平均收益（window={window}d）")
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.tick_params(axis="x", rotation=0)

    (pivot["G5"] - pivot["G1"]).cumsum().plot(ax=ax_ls, color="#1f77b4")
    ax_ls.set_title(f"多空累计收益 G5-G1（window={window}d）")
    ax_ls.axhline(0, color="black", linewidth=0.8)

    spread = mean_ret["G5"] - mean_ret["G1"]
    print(f"  多空价差 G5-G1: {spread:+.4f}")


def main():
    con = duckdb.connect(DB_PATH)
    valuation, price = load_data(con)
    price_dates = pd.DatetimeIndex(sorted(price["date"].unique()))
    symbols = price["symbol"].unique().tolist()
    fwd_df = calc_forward_return(price, FORWARD_DAYS)

    print("=== 估值动量因子（PE 变化率）===\n")
    results = {}
    for w in WINDOWS:
        ic_series, factor_long, icir = run_window(
            valuation, price, price_dates, symbols, fwd_df, w
        )
        results[w] = (ic_series, factor_long, icir)

    best_w = max(results, key=lambda w: abs(results[w][2]))
    print(f"\n最佳窗口: {best_w}d（ICIR={results[best_w][2]:.3f}）")

    # 对最佳窗口画图
    ic_series, factor_long, _ = results[best_w]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    ic_series.plot(ax=axes[0], color="steelblue", alpha=0.7)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].axhline(
        ic_series.mean(),
        color="red",
        linewidth=1,
        linestyle="--",
        label=f"均值={ic_series.mean():.3f}",
    )
    axes[0].set_title(f"IC 时序（window={best_w}d）")
    axes[0].legend()

    layered_plot(factor_long, fwd_df, best_w, axes[1], axes[2])

    plt.tight_layout()
    out = OUT_DIR / f"factor_layered_valuation_momentum_w{best_w}.png"
    plt.savefig(out, dpi=150)
    print(f"图表已保存: {out}")


if __name__ == "__main__":
    main()
