"""协整理论深化：EG vs Johansen 对比 + 残差诊断 + 滚动稳定性"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.strategy.cointegration import (
    engle_granger,
    johansen,
    residual_diagnostics,
    rolling_hedge_ratio,
)

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
OUT_DIR = Path("output/factor_research")

# 示例：招商银行 vs 兴业银行（同为股份制银行，业务结构相近）
SYMBOL_A, SYMBOL_B = "600036", "601166"
NAME_A, NAME_B = "招商银行", "兴业银行"


def load_prices(con, symbol_a: str, symbol_b: str) -> tuple[pd.Series, pd.Series]:
    price = con.execute(f"""
        SELECT symbol, date, close FROM price_daily
        WHERE adjust = 'qfq' AND symbol IN ('{symbol_a}', '{symbol_b}')
        ORDER BY date
    """).df()
    price["date"] = pd.to_datetime(price["date"])
    pivot = price.pivot(index="date", columns="symbol", values="close").dropna()
    return pivot[symbol_a], pivot[symbol_b]


def compare_tests(price_a: pd.Series, price_b: pd.Series) -> None:
    """对比 EG 和 Johansen 两种检验的结论"""
    eg = engle_granger(price_a, price_b)
    jo = johansen(price_a, price_b)

    print("=" * 55)
    print(f"  协整检验对比：{NAME_A} vs {NAME_B}")
    print("=" * 55)
    print("\n【Engle-Granger 两步法】")
    print(
        f"  协整 p-value : {eg.pvalue:.4f}  {'✅ 协整' if eg.is_cointegrated else '❌ 不协整'}"
    )
    print(f"  对冲比率     : {eg.hedge_ratio:.4f}")
    print(f"  残差 ADF     : stat={eg.adf_stat:.3f}  p={eg.adf_pvalue:.4f}")

    print("\n【Johansen 检验】")
    print(
        f"  协整关系数   : {jo.n_cointegration}  {'✅ 协整' if jo.is_cointegrated else '❌ 不协整'}"
    )
    for i, (ts, tc, es, ec) in enumerate(
        zip(jo.trace_stats, jo.trace_crit, jo.eigen_stats, jo.eigen_crit)
    ):
        print(f"  H{i}: Trace={ts:.2f}(>{tc:.2f})  Eigen={es:.2f}(>{ec:.2f})")

    diag = residual_diagnostics(eg.residual)
    print("\n【残差诊断】")
    print(
        f"  ADF p-value  : {diag['adf_pvalue']:.4f}  {'平稳 ✅' if diag['is_stationary'] else '非平稳 ❌'}"
    )
    print(
        f"  Ljung-Box(20): p={diag['ljung_box_pvalue']:.4f}  {'有自相关 ⚠️' if diag['has_autocorr'] else '无自相关 ✅'}"
    )
    print(f"  Durbin-Watson: {diag['durbin_watson']:.3f}  (接近 2 为佳)")

    return eg


def plot_analysis(price_a, price_b, eg_result) -> None:
    roll = rolling_hedge_ratio(price_a, price_b, window=120)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 价格走势（归一化）
    (price_a / price_a.iloc[0]).plot(ax=axes[0][0], label=NAME_A)
    (price_b / price_b.iloc[0]).plot(ax=axes[0][0], label=NAME_B)
    axes[0][0].set_title("归一化价格走势")
    axes[0][0].legend()

    # 协整残差
    eg_result.residual.plot(ax=axes[0][1], color="steelblue")
    axes[0][1].axhline(0, color="black", linewidth=0.8)
    axes[0][1].axhline(
        eg_result.residual.std() * 2,
        color="red",
        linewidth=0.8,
        linestyle="--",
        label="±2σ",
    )
    axes[0][1].axhline(
        -eg_result.residual.std() * 2, color="red", linewidth=0.8, linestyle="--"
    )
    axes[0][1].set_title("协整残差（EG）")
    axes[0][1].legend()

    # 滚动对冲比率
    roll.plot(ax=axes[1][0], color="darkorange")
    axes[1][0].axhline(
        roll.mean(),
        color="red",
        linewidth=0.8,
        linestyle="--",
        label=f"均值={roll.mean():.3f}",
    )
    axes[1][0].set_title("滚动对冲比率（120日窗口）")
    axes[1][0].legend()

    # 残差 ACF（自相关图）
    from statsmodels.graphics.tsaplots import plot_acf

    plot_acf(eg_result.residual, lags=30, ax=axes[1][1], alpha=0.05)
    axes[1][1].set_title("残差自相关（ACF）")

    plt.suptitle(f"{NAME_A} vs {NAME_B} 协整分析", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "cointegration_analysis.png"
    plt.savefig(out, dpi=150)
    print(f"\n图表已保存: {out}")


def main():
    con = duckdb.connect(DB_PATH)
    price_a, price_b = load_prices(con, SYMBOL_A, SYMBOL_B)
    print(f"数据区间：{price_a.index.min().date()} ~ {price_a.index.max().date()}")

    eg_result = compare_tests(price_a, price_b)
    plot_analysis(price_a, price_b, eg_result)


if __name__ == "__main__":
    main()
