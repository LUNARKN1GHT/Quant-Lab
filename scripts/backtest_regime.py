"""Regime-aware ML 回测：根据市场状态动态调整仓位比例"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.backtest.engine import backtest
from quant.config import Config
from quant.regime.detector import detect_regime
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select
from scripts.backtest_ml import build_features, load_close

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")


def run_regime(model: RegressorMixin | None = None) -> dict:
    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

    print("检测市场状态...")
    regime = detect_regime(
        close,
        ma_window=cfg.regime.ma_window,
        breadth_window=cfg.regime.breadth_window,
        vol_short=cfg.regime.vol_short,
        vol_long=cfg.regime.vol_long,
    )
    regime_counts = regime.value_counts()
    print(
        f"  BULL: {regime_counts.get('BULL', 0)}天 "
        + f" RANGE: {regime_counts.get('RANGE', 0)}天 "
        + f" BEAR: {regime_counts.get('BEAR', 0)}天"
    )

    print("构建特征...")
    features = build_features(close)

    fwd_ret = (
        close.pct_change(cfg.backtest.predict_window)
        .shift(-cfg.backtest.predict_window)
        .stack()
    )
    fwd_ret.index.names = ["date", "stock"]
    fwd_ret.name = "fwd_ret"

    dataset = features.join(fwd_ret).dropna()
    dates = dataset.index.get_level_values("date").unique().sort_values()
    print(f"  有效样本: {len(dataset)} 条，覆盖 {len(dates)} 个交易日")

    if model is None:
        model = RandomForestRegressor(
            n_estimators=cfg.ml.n_estimators, n_jobs=-1, random_state=42
        )

    score_frames = []
    rebalance_dates = dates[cfg.backtest.train_window :: cfg.backtest.predict_window]

    print(f"Walk-forward 训练（共 {len(rebalance_dates)} 期）...")
    for i, t in enumerate(rebalance_dates, 1):
        train_dates = dates[dates < t][-cfg.backtest.train_window :]
        if len(train_dates) < cfg.backtest.train_window // 2:
            continue
        train_data = dataset.loc[train_dates]
        X_train = train_data.drop(columns="fwd_ret")
        y_train = train_data["fwd_ret"]

        if t not in dataset.index.get_level_values("date"):
            continue
        X_pred = dataset.loc[t].drop(columns="fwd_ret")

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        preds = pd.Series(fold_model.predict(X_pred), index=X_pred.index, name=t)
        score_frames.append(preds)

        if i % 10 == 0:
            print(f"  [{i}/{len(rebalance_dates)}] {t.date()}")

    print("  训练完成")

    scores_wide = pd.DataFrame(score_frames)
    scores_wide.index = pd.DatetimeIndex(scores_wide.index)
    scores_wide = scores_wide.reindex(columns=close.columns)

    print("构造持仓（含 Regime 缩放）...")
    daily_returns = close.pct_change()
    positions = scores_wide.apply(
        lambda row: factor_select(row.dropna(), top_n=cfg.backtest.top_n).reindex(
            close.columns, fill_value=0.0
        ),
        axis=1,
    )
    positions = positions.reindex(close.index).ffill().fillna(0.0)

    # 按 regime 缩放仓位
    scale_map = {
        "BULL": cfg.regime.bull_scale,
        "RANGE": cfg.regime.range_scale,
        "BEAR": cfg.regime.bear_scale,
    }
    regime_scale = (
        regime.map(scale_map)
        .reindex(positions.index)
        .ffill()
        .fillna(cfg.regime.range_scale)
    )
    positions = positions.mul(regime_scale, axis=0)

    strategy_returns = backtest(
        positions, daily_returns, commission_rate=cfg.backtest.commission_rate
    )
    benchmark_returns = daily_returns.mean(axis=1)

    start = strategy_returns.first_valid_index()
    strategy_returns = strategy_returns.loc[start:].fillna(0)
    benchmark_returns = benchmark_returns.loc[start:].fillna(0)

    strategy_nav = (1 + strategy_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()

    ann_ret = strategy_returns.mean() * 252
    ann_ret_bm = benchmark_returns.mean() * 252

    print("\n" + "=" * 50)
    print(f"{'指标':<18} {'Regime策略':>10} {'基准':>10}")
    print("=" * 50)
    print(f"{'年化收益':<18} {ann_ret:>9.2%} {ann_ret_bm:>9.2%}")
    print(
        f"{'Sharpe':<18} {sharpe(strategy_returns):>10.3f}"
        + f" {sharpe(benchmark_returns):>10.3f}"
    )
    print(
        f"{'Sortino':<18} {sortino(strategy_returns):>10.3f}"
        + f" {sortino(benchmark_returns):>10.3f}"
    )
    print(
        f"{'最大回撤':<18} {max_drawdown(strategy_returns):>9.2%}"
        + f" {max_drawdown(benchmark_returns):>9.2%}"
    )
    print(
        f"{'Calmar':<18} {calmar(strategy_returns):>10.3f}"
        + f" {calmar(benchmark_returns):>10.3f}"
    )
    print("=" * 50)

    final_strategy = strategy_nav.iloc[-1]
    final_bm = benchmark_nav.iloc[-1]
    print(f"\n期末净值  Regime策略: {final_strategy:.3f}  基准: {final_bm:.3f}")
    print(f"超额收益  累计: {final_strategy - final_bm:+.3f}")

    print("\n=== 年度收益 ===")
    yearly = strategy_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    yearly_bm = benchmark_returns.resample("YE").apply(lambda r: (1 + r).prod() - 1)
    for year in yearly.index:
        s, b = yearly[year], yearly_bm[year]
        print(f"  {year.year}  策略: {s:>7.2%}  基准: {b:>7.2%}  超额: {s - b:>+7.2%}")

    return {
        "sharpe": sharpe(strategy_returns),
        "max_drawdown": max_drawdown(strategy_returns),
        "calmar": calmar(strategy_returns),
        "ann_ret": ann_ret,
        "final_nav": strategy_nav.iloc[-1],
    }


if __name__ == "__main__":
    run_regime()
