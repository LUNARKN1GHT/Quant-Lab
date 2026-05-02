"""盈利质量因子 IC 验证"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.factor.earnings_quality import earnings_quality_pit
from quant.factor.layered import layered_return

DB_PATH = "data/quant.duckdb"
FORWARD_DAYS = 20
OUT_DIR = Path("output/factor_research")


def load_data(con):
    fundamental = con.execute("""
        SELECT symbol, report_date, disclose_date, cfo_to_profit
        FROM fundamental_quarterly
        ORDER BY symbol, disclose_date
    """).df()

    price = con.execute("""
        SELECT symbol, date, close
        FROM price_daily
        WHERE adjust = 'qfq'
        ORDER BY symbol, date
    """).df()
    price["date"] = pd.to_datetime(price["date"])
    return fundamental, price


def calc_forward_return(price: pd.DataFrame, n: int) -> pd.DataFrame:
    pivot = price.pivot(index="date", columns="symbol", values="close")
    fwd = pivot.shift(-n) / pivot - 1
    return fwd.stack().rename("fwd_ret").reset_index()


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


def main():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    con = duckdb.connect(DB_PATH)
    fundamental, price = load_data(con)

    price_dates = pd.DatetimeIndex(sorted(price["date"].unique()))
    symbols = price["symbol"].unique().tolist()

    print("计算 Point-in-Time 因子矩阵...")
    factor_matrix = earnings_quality_pit(fundamental, price_dates, symbols)

    factor_matrix.columns.name = "symbol"
    factor_long = (
        factor_matrix.stack()
        .reset_index()
        .rename(columns={0: "factor"})
    )

    fwd_df = calc_forward_return(price, FORWARD_DAYS)

    ic_series = rolling_ic(factor_long, fwd_df)
    icir = ic_series.mean() / ic_series.std()

    print("\n=== 盈利质量因子（CFO/净利润）===")
    print(f"  IC 均值  : {ic_series.mean():.4f}")
    print(f"  ICIR     : {icir:.3f}")
    print(f"  IC>0 占比: {(ic_series > 0).mean():.1%}")
    print(
        f"  样本日期 : {ic_series.index.min().date()} ~ {ic_series.index.max().date()}"
    )

    # 分层回测
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

    df = pd.DataFrame(records)
    pivot = df.pivot(index="date", columns="group", values="ret").sort_index()
    mean_ret = pivot.mean()

    print("\n=== 分层平均收益 ===")
    for g, r in mean_ret.items():
        bar = "█" * int(abs(r) * 500)
        sign = "+" if r >= 0 else ""
        print(f"  {g}: {sign}{r:.4f}  {bar}")
    print(f"  多空价差 G5-G1: {mean_ret['G5'] - mean_ret['G1']:+.4f}")

    # 画图
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
    axes[0].set_title("IC 时序")
    axes[0].legend()

    mean_ret.plot(
        kind="bar",
        ax=axes[1],
        color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"],
    )
    axes[1].set_title("各层平均收益（fwd=20d）")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=0)

    ls = (pivot["G5"] - pivot["G1"]).cumsum()
    ls.plot(ax=axes[2], color="#1f77b4")
    axes[2].set_title("多空累计收益（G5 - G1）")
    axes[2].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out = OUT_DIR / "factor_layered_earnings_quality.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
