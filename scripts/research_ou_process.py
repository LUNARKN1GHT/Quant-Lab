# scripts/research_ou_process.py
"""OU 过程建模：参数估计、半衰期、阈值推导"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.strategy.cointegration import engle_granger
from quant.strategy.ou_process import entry_exit_thresholds, fit_ou, ou_zscore

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
OUT_DIR = Path("output/factor_research")
SYMBOL_A, SYMBOL_B = "601398", "601939"
NAME_A, NAME_B = "工商银行", "建设银行"


def load_prices(con) -> tuple[pd.Series, pd.Series]:
    price = con.execute(f"""
        SELECT symbol, date, close FROM price_daily
        WHERE adjust = 'qfq' AND symbol IN ('{SYMBOL_A}', '{SYMBOL_B}')
        ORDER BY date
    """).df()
    price["date"] = pd.to_datetime(price["date"])
    pivot = price.pivot(index="date", columns="symbol", values="close").dropna()
    return pivot[SYMBOL_A], pivot[SYMBOL_B]


def rolling_ou(spread: pd.Series, window: int = 252) -> pd.DataFrame:
    """滚动窗口拟合 OU，观察参数时变性"""
    records = []
    for i in range(window, len(spread) + 1):
        sub = spread.iloc[i - window : i]
        params = fit_ou(sub)
        records.append(
            {
                "date": spread.index[i - 1],
                "theta": params.theta,
                "half_life": params.half_life,
                "sigma": params.sigma,
            }
        )
    return pd.DataFrame(records).set_index("date")


def main():
    con = duckdb.connect(DB_PATH)
    price_a, price_b = load_prices(con)

    # 协整残差作为价差
    eg = engle_granger(price_a, price_b)
    spread = eg.residual

    # 全样本 OU 拟合
    params = fit_ou(spread)
    thresholds = entry_exit_thresholds(params)
    zscore = ou_zscore(spread, params, window=60)

    print("=" * 50)
    print(f"  OU 过程参数：{NAME_A} vs {NAME_B}")
    print("=" * 50)
    print(f"  θ（回归速度） : {params.theta:.4f}")
    print(f"  μ（长期均值） : {params.mu:.4f}")
    print(f"  σ（波动率）   : {params.sigma:.4f}")
    print(f"  半衰期        : {params.half_life:.1f} 交易日")
    print(f"  对数似然      : {params.log_likelihood:.1f}")
    print(f"\n  建议阈值（基于半衰期 {params.half_life:.0f}d）：")
    print(
        f"  开仓 z > ±{thresholds['entry']}  平仓 z < ±{thresholds['exit']}  止损 z > ±{thresholds['stop_loss']}"
    )

    # 滚动参数
    roll = rolling_ou(spread, window=252)
    valid_hl = roll["half_life"].clip(0, 100)

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    spread.plot(ax=axes[0][0], color="steelblue")
    axes[0][0].axhline(
        params.mu, color="red", linewidth=1, linestyle="--", label=f"μ={params.mu:.2f}"
    )
    axes[0][0].set_title("协整价差")
    axes[0][0].legend()

    zscore.plot(ax=axes[0][1], color="steelblue", alpha=0.8)
    axes[0][1].axhline(
        thresholds["entry"],
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"开仓 ±{thresholds['entry']}",
    )
    axes[0][1].axhline(-thresholds["entry"], color="red", linestyle="--", linewidth=1)
    axes[0][1].axhline(
        thresholds["exit"],
        color="green",
        linestyle="--",
        linewidth=1,
        label=f"平仓 ±{thresholds['exit']}",
    )
    axes[0][1].axhline(-thresholds["exit"], color="green", linestyle="--", linewidth=1)
    axes[0][1].axhline(0, color="black", linewidth=0.8)
    axes[0][1].set_title("价差 z-score（60日滚动标准化）")
    axes[0][1].legend(fontsize=8)

    valid_hl.plot(ax=axes[1][0], color="darkorange")
    axes[1][0].axhline(
        5, color="green", linestyle="--", linewidth=0.8, label="5日（下限）"
    )
    axes[1][0].axhline(
        30, color="red", linestyle="--", linewidth=0.8, label="30日（上限）"
    )
    axes[1][0].set_title("滚动半衰期（252日窗口，截断至100）")
    axes[1][0].legend(fontsize=8)

    roll["theta"].plot(ax=axes[1][1], color="purple")
    axes[1][1].axhline(0, color="black", linewidth=0.8)
    axes[1][1].set_title("滚动 θ（均值回归速度）")

    plt.suptitle(f"{NAME_A} vs {NAME_B}  OU 过程建模", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "ou_process_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
