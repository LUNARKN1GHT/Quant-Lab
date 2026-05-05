"""Kalman Filter vs 静态 OLS 对冲比率对比研究

对比：
1. 静态 OLS 对冲比率（全样本一次估计）
2. 滚动 OLS（60 日窗口）
3. Kalman Filter（动态跟踪）

评估维度：
- 对冲比率时序稳定性
- 价差平稳性（ADF p-value）
- 策略信号质量（z-score 分布）
"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.config import Config
from quant.strategy.kalman_hedge import fit_kalman, kalman_zscore
from quant.strategy.ou_process import fit_ou

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
SYMBOL_A = "601398"  # 工商银行
SYMBOL_B = "601939"  # 建设银行
_cfg = Config()
ROLLING_WINDOW = _cfg.stat_arb.pca_window


def load_prices(con, sym_a, sym_b):
    df = con.execute(f"""
        SELECT date, symbol, close
        FROM price_daily
        WHERE symbol IN ('{sym_a}', '{sym_b}') AND adjust = 'qfq'
        ORDER BY date
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="symbol", values="close").dropna()
    return pivot[sym_a], pivot[sym_b]


def rolling_ols_hedge(price_a, price_b, window=60):
    """滚动 OLS 对冲比率"""
    ratios = []
    for t in range(len(price_a)):
        if t < window:
            ratios.append(np.nan)
        else:
            x = price_b.values[t - window : t]
            y = price_a.values[t - window : t]
            beta = np.polyfit(x, y, 1)[0]
            ratios.append(beta)
    return pd.Series(ratios, index=price_a.index, name="rolling_hedge")


def static_ols_spread(price_a, price_b):
    beta = np.polyfit(price_b.values, price_a.values, 1)[0]
    raw = price_a - beta * price_b
    return (raw - raw.mean()) / raw.std(), beta


def adf_pvalue(series):
    clean = series.dropna()
    if len(clean) < 30:
        return np.nan
    return adfuller(clean)[1]


def main():
    con = duckdb.connect(DB_PATH)
    price_a, price_b = load_prices(con, SYMBOL_A, SYMBOL_B)

    print(f"数据区间: {price_a.index[0].date()} ~ {price_a.index[-1].date()}")
    print(f"样本数: {len(price_a)}")

    # 1. 静态 OLS
    static_z, static_beta = static_ols_spread(price_a, price_b)
    static_adf = adf_pvalue(static_z)

    # 2. 滚动 OLS
    roll_beta = rolling_ols_hedge(price_a, price_b, ROLLING_WINDOW)
    roll_spread = price_a - roll_beta * price_b
    roll_z = (
        roll_spread - roll_spread.rolling(ROLLING_WINDOW).mean()
    ) / roll_spread.rolling(ROLLING_WINDOW).std()
    roll_adf = adf_pvalue(roll_spread.dropna())

    # 3. Kalman Filter
    kf = fit_kalman(price_a, price_b, delta=1e-4)
    kf_z = kalman_zscore(kf, window=20)
    kf_adf = adf_pvalue(kf.spread)

    # OU 参数对比
    ou_static = fit_ou(static_z.dropna())
    ou_kf = fit_ou(kf.spread.dropna())

    print("\n=== 价差平稳性（ADF p-value，越小越平稳）===")
    print(f"  静态 OLS : {static_adf:.4f}  β={static_beta:.4f}")
    print(f"  滚动 OLS : {roll_adf:.4f}")
    print(f"  Kalman   : {kf_adf:.4f}")

    print("\n=== OU 参数对比 ===")
    print(
        f"  静态 OLS  半衰期={ou_static.half_life:.1f}d  θ={ou_static.theta:.4f}  σ={ou_static.sigma:.4f}"
    )
    print(
        f"  Kalman    半衰期={ou_kf.half_life:.1f}d   θ={ou_kf.theta:.4f}  σ={ou_kf.sigma:.4f}"
    )

    print("\n=== z-score 分布（绝对值 >2 的比例，越高信号越多）===")
    print(f"  静态 OLS : {(static_z.abs() > 2).mean():.1%}")
    print(f"  滚动 OLS : {(roll_z.abs() > 2).dropna().mean():.1%}")
    print(f"  Kalman   : {(kf_z.abs() > 2).dropna().mean():.1%}")

    # ------- 画图 -------
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 对冲比率对比
    axes[0][0].plot(
        roll_beta, label=f"滚动OLS({ROLLING_WINDOW}d)", color="steelblue", alpha=0.8
    )
    axes[0][0].plot(
        kf.hedge_ratio, label="Kalman Filter", color="darkorange", linewidth=1.5
    )
    axes[0][0].axhline(
        static_beta, color="gray", linestyle="--", label=f"静态OLS β={static_beta:.3f}"
    )
    axes[0][0].set_title("对冲比率时序对比")
    axes[0][0].legend(fontsize=8)

    # KF 截距
    axes[0][1].plot(kf.intercept, color="purple", linewidth=1)
    axes[0][1].set_title("Kalman Filter 截距 α 时序")
    axes[0][1].axhline(0, color="black", linewidth=0.8)

    # 价差对比
    axes[1][0].plot(static_z, label="静态OLS z-score", color="steelblue", alpha=0.6)
    axes[1][0].plot(kf_z, label="KF z-score", color="darkorange", alpha=0.8)
    axes[1][0].axhline(2, color="red", linestyle="--", linewidth=0.8)
    axes[1][0].axhline(-2, color="red", linestyle="--", linewidth=0.8)
    axes[1][0].axhline(0, color="black", linewidth=0.8)
    axes[1][0].set_title("价差 z-score 对比（±2σ 为入场阈值）")
    axes[1][0].legend(fontsize=8)

    # KF 原始价差 + 不确定度带
    spread_std = kf.spread.rolling(20).std()
    axes[1][1].plot(kf.spread, color="darkorange", linewidth=1, label="KF 价差")
    axes[1][1].fill_between(
        kf.spread.index,
        kf.spread - 2 * spread_std,
        kf.spread + 2 * spread_std,
        alpha=0.2,
        color="darkorange",
        label="±2σ 带",
    )
    axes[1][1].axhline(0, color="black", linewidth=0.8)
    axes[1][1].set_title("Kalman Filter 价差 + 滚动2σ带")
    axes[1][1].legend(fontsize=8)

    # z-score 分布直方图
    axes[2][0].hist(
        static_z.dropna(),
        bins=60,
        alpha=0.6,
        color="steelblue",
        label="静态OLS",
        density=True,
    )
    axes[2][0].hist(
        kf_z.dropna(), bins=60, alpha=0.6, color="darkorange", label="KF", density=True
    )
    axes[2][0].axvline(2, color="red", linestyle="--")
    axes[2][0].axvline(-2, color="red", linestyle="--")
    axes[2][0].set_title("z-score 分布对比")
    axes[2][0].legend(fontsize=8)

    # 滚动 ADF p-value（KF 价差，250日滚动窗口）
    roll_adf_kf = kf.spread.rolling(250).apply(
        lambda x: adfuller(x)[1] if len(x.dropna()) > 30 else np.nan, raw=False
    )
    roll_adf_static = static_z.rolling(250).apply(
        lambda x: adfuller(x)[1] if len(x.dropna()) > 30 else np.nan, raw=False
    )
    axes[2][1].plot(roll_adf_kf, label="KF 价差 ADF p", color="darkorange")
    axes[2][1].plot(
        roll_adf_static, label="静态OLS ADF p", color="steelblue", alpha=0.7
    )
    axes[2][1].axhline(0.05, color="red", linestyle="--", label="p=0.05")
    axes[2][1].set_title("滚动 ADF p-value（250日窗口）")
    axes[2][1].set_ylim(0, 0.5)
    axes[2][1].legend(fontsize=8)

    plt.suptitle(
        f"Kalman Filter vs 静态/滚动 OLS 对冲比率对比\n{SYMBOL_A}(工商银行) vs {SYMBOL_B}(建设银行)",
        y=1.01,
    )
    plt.tight_layout()

    out = Path("output/factor_research/kalman_hedge_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
