"""动量因子回测：每月调仓，多头持仓 Top20，对比沪深 300 等权基准"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.backtest.engine import backtest, rebalance
from quant.config import Config
from quant.factor.momentum import momentum
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")

DATA_DIR = Path(__file__).parent.parent / cfg.data.data_dir


def load_close() -> pd.DataFrame:
    frames = {}
    for f in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f, usecols=["trade_date", "close"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        frames[f.stem] = df.set_index("trade_date").sort_index()["close"]
    return pd.DataFrame(frames)


def run():
    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

    daily_returns = close.pct_change()

    # 每日截面因子值
    factor_df = close.apply(lambda s: momentum(s, cfg.factor.momentum_windows[0]))

    # 每日构造持仓（因子选股 Top20，等权）
    print("构造持仓...")
    positions = factor_df.apply(
        lambda row: factor_select(row.dropna(), top_n=cfg.backtest.top_n).reindex(
            factor_df.columns, fill_value=0.0
        ),
        axis=1,
    )

    # 月度调仓
    positions_monthly = rebalance(positions, freq="ME")

    # 回测
    strategy_returns = backtest(
        positions_monthly, daily_returns, commission_rate=cfg.backtest.commission_rate
    )

    # 等权基准（沪深300成分股每日等权）
    benchmark_returns = daily_returns.mean(axis=1)

    # 对齐并去掉开头 NaN
    start = max(
        strategy_returns.first_valid_index(), benchmark_returns.first_valid_index()
    )
    strategy_returns = strategy_returns.loc[start:].fillna(0)
    benchmark_returns = benchmark_returns.loc[start:].fillna(0)

    # 净值曲线
    strategy_nav = (1 + strategy_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()

    # 绩效指标
    ann_ret = strategy_returns.mean() * 252
    ann_ret_bm = benchmark_returns.mean() * 252

    print("\n" + "=" * 50)
    print(f"{'指标':<18} {'策略':>10} {'基准':>10}")
    print("=" * 50)
    print(f"{'年化收益':<18} {ann_ret:>9.2%} {ann_ret_bm:>9.2%}")
    print(
        f"{'Sharpe':<18} {sharpe(strategy_returns):>10.3f}"
        + " {sharpe(benchmark_returns):>10.3f}"
    )
    print(
        f"{'Sortino':<18} {sortino(strategy_returns):>10.3f}"
        + " {sortino(benchmark_returns):>10.3f}"
    )
    print(
        f"{'最大回撤':<18} {max_drawdown(strategy_returns):>9.2%}"
        + " {max_drawdown(benchmark_returns):>9.2%}"
    )
    print(
        f"{'Calmar':<18} {calmar(strategy_returns):>10.3f}"
        + " {calmar(benchmark_returns):>10.3f}"
    )
    print("=" * 50)

    final_strategy = strategy_nav.iloc[-1]
    final_bm = benchmark_nav.iloc[-1]
    print(f"\n期末净值  策略: {final_strategy:.3f}  基准: {final_bm:.3f}")
    print(f"超额收益  累计: {final_strategy - final_bm:+.3f}")

    # 打印年度收益明细
    print("\n=== 年度收益 ===")
    yearly = strategy_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    yearly_bm = benchmark_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    for year in yearly.index:
        s = yearly[year]
        b = yearly_bm[year]
        print(f"  {year.year}  策略: {s:>7.2%}  基准: {b:>7.2%}  超额: {s - b:>+7.2%}")


if __name__ == "__main__":
    run()
