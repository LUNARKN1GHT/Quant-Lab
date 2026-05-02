"""两个有效因子的深度分析：IC 衰减 + 分市场环境有效性"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.factor.fund_flow import fund_flow_momentum
from quant.factor.revenue_acceleration import revenue_acceleration_pit
from quant.regime.detector import detect_regime

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
OUT_DIR = Path("output/factor_research")
FORWARD_DAYS = [5, 10, 20, 40, 60]  # IC 衰减分析用多个预测窗口


def load_all(con):
    price = con.execute(
        "SELECT symbol, date, close FROM price_daily WHERE adjust='qfq' ORDER BY symbol, date"
    ).df()
    price["date"] = pd.to_datetime(price["date"])

    flow = con.execute(
        "SELECT symbol, date, main_net_pct FROM fund_flow_daily ORDER BY symbol, date"
    ).df()
    flow["date"] = pd.to_datetime(flow["date"])

    fundamental = con.execute(
        "SELECT symbol, report_date, disclose_date, revenue_yoy FROM fundamental_quarterly ORDER BY symbol, disclose_date"
    ).df()
    return price, flow, fundamental


def build_factors(price, flow, fundamental):
    price_dates = pd.DatetimeIndex(sorted(price["date"].unique()))
    symbols = price["symbol"].unique().tolist()

    # 主力资金因子（20d）
    flow_factor = flow.groupby("symbol")["main_net_pct"].transform(
        lambda s: fund_flow_momentum(s, 20)
    )
    flow["factor"] = flow_factor
    ff = flow[["date", "symbol", "factor"]].copy()

    # 财务加速度因子
    ra_matrix = revenue_acceleration_pit(fundamental, price_dates, symbols)
    ra = ra_matrix.stack().reset_index().rename(columns={0: "factor"})

    return ff, ra


def calc_fwd(price, n):
    pivot = price.pivot(index="date", columns="symbol", values="close")
    fwd = pivot.shift(-n) / pivot - 1
    return fwd.stack().reset_index().rename(columns={0: "fwd_ret"})


def ic_by_date(factor_df, fwd_df):
    merged = factor_df.merge(fwd_df, on=["date", "symbol"])
    records = []
    for date, grp in merged.groupby("date"):
        grp = grp.dropna(subset=["factor", "fwd_ret"])
        if len(grp) < 30:
            continue
        ic, _ = spearmanr(grp["factor"], grp["fwd_ret"])
        records.append({"date": date, "ic": ic})
    return pd.DataFrame(records).set_index("date")["ic"]


def ic_decay(factor_df, price, label):
    """IC 衰减：不同预测窗口下的平均 IC"""
    ics = {}
    for n in FORWARD_DAYS:
        fwd = calc_fwd(price, n)
        ic = ic_by_date(factor_df, fwd)
        ics[n] = {"IC均值": ic.mean(), "ICIR": ic.mean() / ic.std()}
    df = pd.DataFrame(ics, index=["IC均值", "ICIR"]).T
    print(f"\n=== {label} IC 衰减 ===")
    print(df.round(4).to_string())
    return df


def ic_by_regime(factor_df, price, regime, label):
    """分市场环境 IC 分析"""
    fwd = calc_fwd(price, 20)
    ic_series = ic_by_date(factor_df, fwd)
    ic_df = ic_series.rename("ic").to_frame()
    ic_df["regime"] = regime.reindex(ic_df.index).ffill()

    print(f"\n=== {label} 分市场环境 IC ===")
    result = {}
    for r, grp in ic_df.groupby("regime"):
        icir = grp["ic"].mean() / grp["ic"].std()
        result[r] = {"IC均值": grp["ic"].mean(), "ICIR": icir, "样本天数": len(grp)}
        print(
            f"  {r:5s}  IC均值={grp['ic'].mean():.4f}  ICIR={icir:.3f}  ({len(grp)}天)"
        )
    return ic_df, result


def plot_all(ff_decay, ra_decay, ff_ic_df, ra_ic_df, ff_regime, ra_regime):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- 行 1：主力资金因子 ---
    # IC 衰减柱状图
    ff_decay["IC均值"].plot(kind="bar", ax=axes[0][0], color="steelblue")
    axes[0][0].set_title("主力资金：IC 衰减（不同预测窗口）")
    axes[0][0].set_xlabel("预测窗口（交易日）")
    axes[0][0].axhline(0, color="black", linewidth=0.8)
    axes[0][0].tick_params(axis="x", rotation=0)

    # IC 时序（rolling 20 均值）
    ff_ic_df["ic"].rolling(20).mean().plot(ax=axes[0][1], color="steelblue")
    axes[0][1].axhline(0, color="black", linewidth=0.8)
    axes[0][1].set_title("主力资金：IC 滚动均值（20日）")

    # 分 regime IC 柱状图
    REGIME_COLOR = {"BULL": "#2ca02c", "RANGE": "#ff7f0e", "BEAR": "#d62728"}
    reg_ic = {r: v["IC均值"] for r, v in ff_regime.items()}
    s = pd.Series(reg_ic)
    s.plot(kind="bar", ax=axes[0][2], color=[REGIME_COLOR[r] for r in s.index])
    axes[0][2].set_title("主力资金：分市场环境 IC 均值")
    axes[0][2].axhline(0, color="black", linewidth=0.8)
    axes[0][2].tick_params(axis="x", rotation=0)

    # --- 行 2：财务加速度因子 ---
    ra_decay["IC均值"].plot(kind="bar", ax=axes[1][0], color="darkorange")
    axes[1][0].set_title("财务加速度：IC 衰减（不同预测窗口）")
    axes[1][0].set_xlabel("预测窗口（交易日）")
    axes[1][0].axhline(0, color="black", linewidth=0.8)
    axes[1][0].tick_params(axis="x", rotation=0)

    ra_ic_df["ic"].rolling(20).mean().plot(ax=axes[1][1], color="darkorange")
    axes[1][1].axhline(0, color="black", linewidth=0.8)
    axes[1][1].set_title("财务加速度：IC 滚动均值（20日）")

    reg_ic2 = {r: v["IC均值"] for r, v in ra_regime.items()}
    s2 = pd.Series(reg_ic2)
    s2.plot(kind="bar", ax=axes[1][2], color=[REGIME_COLOR[r] for r in s2.index])
    axes[1][2].set_title("财务加速度：分市场环境 IC 均值")
    axes[1][2].axhline(0, color="black", linewidth=0.8)
    axes[1][2].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    out = OUT_DIR / "factor_deep_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


def main():
    con = duckdb.connect(DB_PATH)
    price, flow, fundamental = load_all(con)

    close = price.pivot(index="date", columns="symbol", values="close")
    regime = detect_regime(close)

    ff, ra = build_factors(price, flow, fundamental)

    ff_decay = ic_decay(ff, price, "主力资金因子")
    ra_decay = ic_decay(ra, price, "财务加速度因子")

    ff_ic_df, ff_regime = ic_by_regime(ff, price, regime, "主力资金因子")
    ra_ic_df, ra_regime = ic_by_regime(ra, price, regime, "财务加速度因子")

    plot_all(ff_decay, ra_decay, ff_ic_df, ra_ic_df, ff_regime, ra_regime)


if __name__ == "__main__":
    main()
