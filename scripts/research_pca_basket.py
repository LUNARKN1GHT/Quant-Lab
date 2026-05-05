"""PCA 篮子套利研究 — 银行板块

流程：
1. 读取银行板块所有股票日收益率
2. 滚动 PCA 提取 PC1（板块共同因子）残差
3. 对残差做 z-score，生成多空信号
4. 统计策略表现：年化收益、Sharpe、最大回撤
"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.config import Config
from quant.strategy.pca_basket import (
    basket_signal,
    pca_zscore,
    portfolio_return,
    rolling_pca_residuals,
)

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
_cfg = Config()
PCA_WINDOW = _cfg.stat_arb.pca_window
ZSCORE_WINDOW = _cfg.stat_arb.ou_zscore_window
ENTRY = _cfg.stat_arb.pca_entry
EXIT = _cfg.stat_arb.pca_exit
HOLDING = 5  # 持仓天数（策略专属，不纳入全局配置）

# 银行板块代表股（沪深300中银行股）
BANK_SYMBOLS = [
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "000568",  # 泸州老窖
    "002304",  # 洋河股份
    "000596",  # 古井贡酒
    "603369",  # 今世缘
    "000799",  # 酒鬼酒
    "600809",  # 山西汾酒
]


def load_returns(con, symbols):
    sym_list = "','".join(symbols)
    df = con.execute(f"""
        SELECT date, symbol, close
        FROM price_daily
        WHERE symbol IN ('{sym_list}') AND adjust = 'qfq'
        ORDER BY date
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    pivot = df.pivot(index="date", columns="symbol", values="close")
    # 只保留有足够数据的股票（至少 200 个交易日）
    pivot = pivot.loc[:, pivot.notna().sum() > 200]
    returns = pivot.pct_change()
    return returns


def static_pca_analysis(returns):
    """全样本 PCA 分析（展示用，不用于信号）"""
    clean = returns.dropna()
    pca = PCA(n_components=3)
    pca.fit(clean.values)
    return pca.explained_variance_ratio_, pca.components_


def sharpe(ret: pd.Series, annual: int = 252) -> float:
    r = ret.dropna()
    return r.mean() / r.std() * np.sqrt(annual) if r.std() > 1e-8 else 0.0


def max_drawdown(cum: pd.Series) -> float:
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max.replace(0, np.nan)
    return dd.min()


def main():
    con = duckdb.connect(DB_PATH)
    returns = load_returns(con, BANK_SYMBOLS)

    valid_symbols = returns.columns.tolist()
    print(f"有效股票数: {len(valid_symbols)}")
    print(f"数据区间: {returns.index[0].date()} ~ {returns.index[-1].date()}")

    # 静态 PCA 概览
    ev_ratio, components = static_pca_analysis(returns)
    print("\n=== 全样本 PCA 方差解释比 ===")
    for i, r in enumerate(ev_ratio):
        print(f"  PC{i + 1}: {r:.1%}")

    print("\nPC1 各股载荷:")
    loadings = pd.Series(components[0], index=valid_symbols).sort_values()
    for sym, v in loadings.items():
        print(f"  {sym}: {v:+.4f}")

    # 滚动 PCA 残差
    print(f"\n计算滚动 PCA 残差（window={PCA_WINDOW}）...")
    residuals = rolling_pca_residuals(returns, window=PCA_WINDOW, n_components=1)

    # 只保留特质残差确实平稳的股票（ADF p < 0.1）
    valid_cols = []
    for col in residuals.columns:
        s = residuals[col].dropna()
        if len(s) > 60:
            p = adfuller(s)[1]
            if p < 0.1:
                valid_cols.append(col)
    print(f"ADF 过滤后有效股票: {valid_cols}")
    residuals = residuals[valid_cols]
    returns_filtered = returns[valid_cols]

    # z-score 信号
    zs = pca_zscore(residuals, window=ZSCORE_WINDOW)
    sig = basket_signal(zs, entry_threshold=ENTRY, exit_threshold=EXIT)
    sig = sig * -1
    pnl = portfolio_return(sig, returns_filtered, holding_period=HOLDING)

    # 过滤热身期
    pnl = pnl.dropna()
    cum = (1 + pnl).cumprod()

    ann_ret = pnl.mean() * 252
    sr = sharpe(pnl)
    mdd = max_drawdown(cum)
    hit_rate = (pnl > 0).mean()
    avg_positions = sig.abs().sum(axis=1).mean()

    print("\n=== PCA 篮子套利策略表现 ===")
    print(f"  年化收益  : {ann_ret:.2%}")
    print(f"  Sharpe    : {sr:.3f}")
    print(f"  最大回撤  : {mdd:.2%}")
    print(f"  胜率      : {hit_rate:.1%}")
    print(f"  日均持仓数: {avg_positions:.1f} 只")

    # 按年统计
    print("\n=== 分年收益 ===")
    annual = pnl.groupby(pnl.index.year).apply(lambda x: (1 + x).prod() - 1)
    for yr, r in annual.items():
        bar = "█" * int(abs(r) * 100)
        sign = "+" if r >= 0 else ""
        print(f"  {yr}: {sign}{r:.2%}  {bar}")

    # -------- 画图 --------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # PC1 方差解释
    axes[0][0].bar(
        [f"PC{i + 1}" for i in range(len(ev_ratio))],
        ev_ratio,
        color=["steelblue", "darkorange", "green"],
    )
    axes[0][0].set_title("PCA 方差解释比（全样本）")
    axes[0][0].set_ylabel("解释比例")

    # PC1 载荷分布
    loadings.plot(kind="barh", ax=axes[0][1], color="steelblue")
    axes[0][1].axvline(0, color="black", linewidth=0.8)
    axes[0][1].set_title("PC1 股票载荷（β）")

    # 残差热力图（选最近 120 天）
    recent_resid = residuals.dropna(how="all").tail(120)
    im = axes[0][2].imshow(
        recent_resid.T.values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.03,
        vmax=0.03,
    )
    axes[0][2].set_yticks(range(len(recent_resid.columns)))
    axes[0][2].set_yticklabels(recent_resid.columns, fontsize=7)
    axes[0][2].set_title("近120日 PCA 残差热力图")
    plt.colorbar(im, ax=axes[0][2])

    # 累计收益
    cum.plot(ax=axes[1][0], color="steelblue")
    axes[1][0].axhline(1, color="black", linewidth=0.8)
    axes[1][0].set_title(f"PCA篮子套利累计收益（SR={sr:.2f}）")
    axes[1][0].set_ylabel("净值")

    # 日收益分布
    pnl.hist(bins=60, ax=axes[1][1], color="steelblue", edgecolor="white")
    axes[1][1].axvline(0, color="red", linewidth=1)
    axes[1][1].set_title(f"日收益分布（均值={pnl.mean() * 1e4:.1f}BP）")

    # 滚动 Sharpe（60日）
    roll_sr = pnl.rolling(60).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 1e-8 else 0,
        raw=True,
    )
    roll_sr.plot(ax=axes[1][2], color="darkorange")
    axes[1][2].axhline(0, color="black", linewidth=0.8)
    axes[1][2].axhline(1, color="red", linestyle="--", linewidth=0.8)
    axes[1][2].set_title("滚动 Sharpe（60日窗口）")

    plt.suptitle(
        f"PCA 篮子套利 — 银行板块 | entry±{ENTRY}σ exit±{EXIT}σ holding={HOLDING}d"
    )
    plt.tight_layout()

    out = Path("output/factor_research/pca_basket_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
