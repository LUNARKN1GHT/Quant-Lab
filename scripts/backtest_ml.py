"""LightGBM 截面预测回测：Walk-forward 月度调仓，对比等权基准"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.backtest.engine import backtest
from quant.factor.ma_bias import ma_bias
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.factor.volatility import volatility
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select

DATA_DIR = Path(__file__).parent.parent / "data/csi300"
TRAIN_DAYS = 240  # 训练窗口（约1年）
FORWARD_DAYS = 20  # 预测期（约1个月）
TOP_N = 20
COMMISSION = 0.0003


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


def build_features(close: pd.DataFrame) -> pd.DataFrame:
    """构建因子特征宽表，每列为一个因子，MultiIndex(date, stock)"""
    factor_dict = {
        "mom_20": close.apply(lambda s: momentum(s, 20)),
        "mom_60": close.apply(lambda s: momentum(s, 60)),
        "rsi_14": close.apply(lambda s: rsi(s, 14)),
        "vol_20": close.apply(lambda s: volatility(s, 20)),
        "ma_bias_20": close.apply(lambda s: ma_bias(s, 20)),
    }
    # 每个 DataFrame shape: (dates, stocks) → stack 成长格式
    panel = pd.concat({name: df.stack() for name, df in factor_dict.items()}, axis=1)
    panel.index.names = ["date", "stock"]
    return panel


def run():
    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

    print("构建特征...")
    features = build_features(close)

    # 前向收益（标签）
    fwd_ret = close.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS).stack()
    fwd_ret.index.names = ["date", "stock"]
    fwd_ret.name = "fwd_ret"

    # 合并特征与标签
    dataset = features.join(fwd_ret).dropna()
    dates = dataset.index.get_level_values("date").unique().sort_values()

    print(f"  有效样本: {len(dataset)} 条，覆盖 {len(dates)} 个交易日")

    # Walk-forward：每 FORWARD_DAYS 步重训练一次
    import lightgbm as lgb

    score_frames = []
    rebalance_dates = dates[TRAIN_DAYS::FORWARD_DAYS]

    print(f"Walk-forward 训练（共 {len(rebalance_dates)} 期）...")
    for i, t in enumerate(rebalance_dates, 1):
        # 训练集：t 之前 TRAIN_DAYS 个交易日
        train_dates = dates[dates < t][-TRAIN_DAYS:]
        if len(train_dates) < TRAIN_DAYS // 2:
            continue
        train_data = dataset.loc[train_dates]
        X_train = train_data.drop(columns="fwd_ret")
        y_train = train_data["fwd_ret"]

        # 预测集：当前截面（t 日所有股票）
        if t not in dataset.index.get_level_values("date"):
            continue
        X_pred = dataset.loc[t].drop(columns="fwd_ret")

        model = lgb.LGBMRegressor(n_estimators=100, verbosity=-1)
        model.fit(X_train, y_train)
        preds = pd.Series(model.predict(X_pred), index=X_pred.index, name=t)
        score_frames.append(preds)

        if i % 10 == 0:
            print(f"  [{i}/{len(rebalance_dates)}] {t.date()}")

    print("  训练完成")

    # 把预测分数整理成宽表 (date × stock)
    scores_wide = pd.DataFrame(score_frames)
    scores_wide.index = pd.DatetimeIndex(scores_wide.index)
    scores_wide = scores_wide.reindex(columns=close.columns)

    # 每期选 Top N
    print("构造持仓...")
    daily_returns = close.pct_change()
    positions = scores_wide.apply(
        lambda row: factor_select(row.dropna(), top_n=TOP_N).reindex(
            close.columns, fill_value=0.0
        ),
        axis=1,
    )
    # 对齐到全部交易日，月度调仓（ffill 非调仓日）
    positions = positions.reindex(close.index).ffill().fillna(0.0)

    strategy_returns = backtest(positions, daily_returns, commission_rate=COMMISSION)
    benchmark_returns = daily_returns.mean(axis=1)

    start = strategy_returns.first_valid_index()
    strategy_returns = strategy_returns.loc[start:].fillna(0)
    benchmark_returns = benchmark_returns.loc[start:].fillna(0)

    strategy_nav = (1 + strategy_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()

    ann_ret = strategy_returns.mean() * 252
    ann_ret_bm = benchmark_returns.mean() * 252

    print("\n" + "=" * 50)
    print(f"{'指标':<18} {'ML策略':>10} {'基准':>10}")
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
    print(f"\n期末净值  ML策略: {final_strategy:.3f}  基准: {final_bm:.3f}")
    print(f"超额收益  累计: {final_strategy - final_bm:+.3f}")

    print("\n=== 年度收益 ===")
    yearly = strategy_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    yearly_bm = benchmark_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    for year in yearly.index:
        s, b = yearly[year], yearly_bm[year]
        print(f"  {year.year}  策略: {s:>7.2%}  基准: {b:>7.2%}  超额: {s - b:>+7.2%}")


if __name__ == "__main__":
    run()
