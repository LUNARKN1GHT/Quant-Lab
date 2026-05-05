"""市场中性策略回测 — 主力资金因子

对比三种组合构建方式：
  1. 原始多空（Top20 vs Bottom20，无中性化）
  2. Beta 中性（调整空头规模使 Σw*β=0）
  3. Beta + 行业中性（因子行业内去均值后再建仓）

市场基准：全部股票等权日收益均值（CSI300 近似）
"""

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.config import Config
from quant.factor.fund_flow import fund_flow_momentum
from quant.strategy.market_neutral import (
    backtest_weights,
    build_portfolio,
    compute_rolling_beta,
    sector_neutralize,
)

matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DB_PATH = "data/quant.duckdb"
_cfg = Config()
MOMENTUM_WINDOW = _cfg.market_neutral.momentum_window
BETA_WINDOW = _cfg.market_neutral.beta_window
N_LONG = _cfg.market_neutral.n_long
N_SHORT = _cfg.market_neutral.n_short


# 简化行业映射（按股票代码前缀粗分）
# 实盘应使用申万/中信行业分类
def simple_sector_map(symbols: list[str]) -> dict[str, str]:
    sector = {}
    for s in symbols:
        code = s[:3]
        if code in ("600", "601", "603", "605"):
            prefix = int(s[:6])
            if 600000 <= prefix <= 600099:
                sector[s] = "银行"
            elif 600519 <= prefix <= 600600:
                sector[s] = "消费"
            else:
                sector[s] = f"沪{code}"
        elif code in ("000", "001", "002", "003"):
            sector[s] = f"深{code}"
        else:
            sector[s] = "其他"
    return sector


def load_data(con):
    price = con.execute("""
        SELECT symbol, date, close
        FROM price_daily WHERE adjust='qfq'
        ORDER BY symbol, date
    """).df()
    price["date"] = pd.to_datetime(price["date"])

    flow = con.execute("""
        SELECT symbol, date, main_net_pct
        FROM fund_flow_daily ORDER BY symbol, date
    """).df()
    flow["date"] = pd.to_datetime(flow["date"])
    return price, flow


def sharpe(ret, annual=252):
    r = ret.dropna()
    return r.mean() / r.std() * np.sqrt(annual) if r.std() > 1e-8 else 0.0


def max_drawdown(cum):
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max.replace(0, np.nan)
    return dd.min()


def run_strategy(
    factor_matrix, betas, returns, sector_map=None, label="raw", beta_neutral=True
):
    """运行一种策略配置，返回日收益序列"""
    fm = factor_matrix
    if sector_map is not None:
        fm = sector_neutralize(factor_matrix, sector_map)

    weights_history = []
    dates = fm.index[BETA_WINDOW:]  # 跳过 Beta 热身期
    for date in dates:
        if date not in betas.index or date not in fm.index:
            continue
        scores = fm.loc[date].dropna()
        beta_today = betas.loc[date].dropna()
        w = build_portfolio(
            scores, beta_today, N_LONG, N_SHORT, beta_neutral=beta_neutral
        )
        weights_history.append((date, w))

    pnl = backtest_weights(weights_history, returns)
    return pnl


def print_stats(label, pnl):
    cum = (1 + pnl.dropna()).cumprod()
    sr = sharpe(pnl)
    mdd = max_drawdown(cum)
    ann = pnl.dropna().mean() * 252
    print(f"  {label:20s}  年化={ann:.2%}  SR={sr:.3f}  MDD={mdd:.2%}")


def main():
    con = duckdb.connect(DB_PATH)
    price, flow = load_data(con)

    # 收益率矩阵
    pivot_price = price.pivot(index="date", columns="symbol", values="close")
    returns = pivot_price.pct_change()

    # 市场基准（等权）
    market_ret = returns.mean(axis=1)

    # 主力资金因子矩阵
    print("构建主力资金因子...")
    factor_list = []
    for sym, grp in flow.groupby("symbol"):
        s = grp.set_index("date")["main_net_pct"]
        f = fund_flow_momentum(s, MOMENTUM_WINDOW).rename(sym)
        factor_list.append(f)
    factor_matrix = pd.concat(factor_list, axis=1)
    factor_matrix.index = pd.to_datetime(factor_matrix.index)
    factor_matrix = factor_matrix.reindex(pivot_price.index)

    # 滚动 Beta
    print(f"计算滚动 Beta（window={BETA_WINDOW}）...")
    betas = compute_rolling_beta(returns, market_ret, window=BETA_WINDOW)

    symbols = factor_matrix.columns.tolist()
    sector_map = simple_sector_map(symbols)

    # 三种策略对比
    print("\n运行策略回测...")
    pnl_raw = run_strategy(factor_matrix, betas, returns, beta_neutral=False)
    pnl_beta = run_strategy(factor_matrix, betas, returns, beta_neutral=True)
    pnl_neutral = run_strategy(
        factor_matrix, betas, returns, sector_map=sector_map, beta_neutral=True
    )

    # 注意：raw 和 beta 共用同一个 run_strategy，区别在 build_portfolio 里
    # 这里为演示简单，三者共享同一 build_portfolio（含 Beta 中性）
    # 若要对比纯 raw（无中性化），需在 build_portfolio 里去掉 scale 调整

    print("\n=== 策略表现对比 ===")
    print_stats("原始多空", pnl_raw)
    print_stats("Beta中性", pnl_beta)
    print_stats("Beta+行业中性", pnl_neutral)

    # 市场相关性
    for label, pnl in [
        ("原始多空", pnl_raw),
        ("Beta中性", pnl_beta),
        ("Beta+行业中性", pnl_neutral),
    ]:
        corr = pnl.dropna().corr(market_ret.reindex(pnl.dropna().index))
        print(f"  {label:20s}  与市场相关性={corr:.3f}")

    # 分年收益
    print("\n=== Beta中性 分年收益 ===")
    annual = pnl_beta.groupby(pnl_beta.index.year).apply(lambda x: (1 + x).prod() - 1)
    for yr, r in annual.items():
        bar = "█" * int(abs(r) * 100)
        sign = "+" if r >= 0 else ""
        print(f"  {yr}: {sign}{r:.2%}  {bar}")

    # ------- 画图 -------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 累计收益对比
    for label, pnl, color in [
        ("原始多空", pnl_raw, "steelblue"),
        ("Beta中性", pnl_beta, "darkorange"),
        ("Beta+行业中性", pnl_neutral, "green"),
    ]:
        cum = (1 + pnl.dropna()).cumprod()
        cum.plot(ax=axes[0][0], label=label, color=color)
    market_cum = (1 + market_ret.reindex(pnl_beta.index).dropna()).cumprod()
    market_cum.plot(
        ax=axes[0][0], label="市场基准", color="gray", linestyle="--", alpha=0.7
    )
    axes[0][0].set_title("三种组合累计收益对比")
    axes[0][0].legend(fontsize=8)
    axes[0][0].axhline(1, color="black", linewidth=0.5)

    # Beta 分布（最新截面）
    latest_beta = betas.iloc[-1].dropna()
    latest_beta.hist(bins=40, ax=axes[0][1], color="steelblue", edgecolor="white")
    axes[0][1].axvline(
        latest_beta.mean(),
        color="red",
        linestyle="--",
        label=f"均值={latest_beta.mean():.2f}",
    )
    axes[0][1].set_title("股票 Beta 分布（最新）")
    axes[0][1].legend()

    # 市场相关性滚动（60日）
    for label, pnl, color in [
        ("原始多空", pnl_raw, "steelblue"),
        ("Beta中性", pnl_beta, "darkorange"),
    ]:
        roll_corr = pnl.rolling(60).corr(market_ret)
        roll_corr.plot(ax=axes[0][2], label=label, color=color)
    axes[0][2].axhline(0, color="black", linewidth=0.8)
    axes[0][2].set_title("滚动市场相关性（60日）")
    axes[0][2].legend(fontsize=8)
    axes[0][2].set_ylim(-1, 1)

    # Beta 中性净值
    cum_beta = (1 + pnl_beta.dropna()).cumprod()
    cum_beta.plot(ax=axes[1][0], color="darkorange")
    axes[1][0].set_title(f"Beta中性组合净值（SR={sharpe(pnl_beta):.2f}）")
    axes[1][0].axhline(1, color="black", linewidth=0.5)

    # 多空腿 Beta 时序（验证中性化效果）
    # 用 pnl 滚动波动率替代
    roll_vol = pnl_beta.rolling(60).std() * np.sqrt(252)
    roll_vol.plot(ax=axes[1][1], color="darkorange")
    axes[1][1].set_title("Beta中性组合：滚动年化波动率（60日）")
    axes[1][1].set_ylabel("波动率")

    # 分年收益柱状图
    colors = ["#2ca02c" if r > 0 else "#d62728" for r in annual.values]
    annual.plot(kind="bar", ax=axes[1][2], color=colors)
    axes[1][2].axhline(0, color="black", linewidth=0.8)
    axes[1][2].set_title("Beta中性 分年收益")
    axes[1][2].tick_params(axis="x", rotation=45)

    plt.suptitle(f"市场中性策略 — 主力资金因子 | L{N_LONG}/S{N_SHORT} Beta中性化")
    plt.tight_layout()

    out = Path("output/factor_research/market_neutral_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {out}")


if __name__ == "__main__":
    main()
