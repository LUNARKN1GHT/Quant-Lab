"""ML 截面预测回测：Walk-forward 月度调仓，对比等权基准

支持两种模式：
- run()：单模型（默认 LightGBM），直接预测下期收益排序选股
- run_stack()：Stacking 集成，多个基模型 → meta-model 组合，通常泛化性更好

特征工程由 build_features() 完成，目前使用 5 个技术因子构成截面特征矩阵。
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.backtest.engine import backtest
from quant.config import Config
from quant.factor.ma_bias import ma_bias
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.factor.volatility import volatility
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


def build_features(close: pd.DataFrame) -> pd.DataFrame:
    """构建截面因子特征矩阵，供 ML 模型训练和预测使用。

    每个因子先计算成 (date × stock) 宽表，再通过 stack() 转为长格式，
    最后 concat 成 MultiIndex(date, stock) 的特征矩阵，每列为一个因子。

    当前特征：短期动量(20日)、中期动量(60日)、RSI、波动率、均线偏离率
    """
    factor_dict = {
        "mom_20": close.apply(lambda s: momentum(s, cfg.factor.momentum_windows[0])),
        "mom_60": close.apply(lambda s: momentum(s, cfg.factor.momentum_windows[1])),
        "rsi_14": close.apply(lambda s: rsi(s, cfg.factor.rsi_window)),
        "vol_20": close.apply(lambda s: volatility(s, 20)),
        "ma_bias_20": close.apply(lambda s: ma_bias(s, 20)),
    }
    # 每个 DataFrame shape: (dates, stocks) → stack 转为长格式 (date, stock)
    panel = pd.concat({name: df.stack() for name, df in factor_dict.items()}, axis=1)
    panel.index.names = ["date", "stock"]
    return panel


def run(model: RegressorMixin | None = None) -> dict:
    """单模型 walk-forward 回测。

    Args:
        model: 回归模型，默认使用 LightGBM。传入其他 sklearn 兼容模型可直接替换。
    """
    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

    print("构建特征...")
    features = build_features(close)

    # 前向收益作为标签：t 日的标签 = t 到 t+predict_window 日的区间收益
    # shift(-predict_window) 将收益对齐到对应的特征日期
    fwd_ret = (
        close.pct_change(cfg.backtest.predict_window)
        .shift(-cfg.backtest.predict_window)
        .stack()
    )
    fwd_ret.index.names = ["date", "stock"]
    fwd_ret.name = "fwd_ret"

    # dropna 去掉特征或标签有缺失的样本
    dataset = features.join(fwd_ret).dropna()
    dates = dataset.index.get_level_values("date").unique().sort_values()

    print(f"  有效样本: {len(dataset)} 条，覆盖 {len(dates)} 个交易日")

    if model is None:
        model = lgb.LGBMRegressor(n_estimators=cfg.ml.n_estimators, verbosity=-1)

    score_frames = []
    # 每隔 predict_window 个交易日重训一次，模拟实盘月度再训练
    rebalance_dates = dates[cfg.backtest.train_window :: cfg.backtest.predict_window]

    print(f"Walk-forward 训练（共 {len(rebalance_dates)} 期）...")
    for i, t in enumerate(rebalance_dates, 1):
        # 训练集：t 之前最近 train_window 个交易日的历史数据
        train_dates = dates[dates < t][-cfg.backtest.train_window :]
        if len(train_dates) < cfg.backtest.train_window // 2:
            continue
        train_data = dataset.loc[train_dates]
        X_train = train_data.drop(columns="fwd_ret")
        y_train = train_data["fwd_ret"]

        # 预测集：t 日截面（所有股票当日因子值）
        if t not in dataset.index.get_level_values("date"):
            continue
        X_pred = dataset.loc[t].drop(columns="fwd_ret")

        # clone 确保每轮使用全新未训练的模型实例
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        preds = pd.Series(fold_model.predict(X_pred), index=X_pred.index, name=t)
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
        lambda row: factor_select(row.dropna(), top_n=cfg.backtest.top_n).reindex(
            close.columns, fill_value=0.0
        ),
        axis=1,
    )
    # 对齐到全部交易日，月度调仓（ffill 非调仓日）
    positions = positions.reindex(close.index).ffill().fillna(0.0)

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
    print(f"{'指标':<18} {'ML策略':>10} {'基准':>10}")
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
    print(f"\n期末净值  ML策略: {final_strategy:.3f}  基准: {final_bm:.3f}")
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


def run_stack(
    base_models: list[RegressorMixin],
    meta_model: RegressorMixin,
    holdout_ratio: float = cfg.ml.holdout_ratio,
) -> dict:
    """Stacking 集成 + walk-forward 回测。

    每个调仓期内：
    1. 训练集切分为 train1（前 1-holdout_ratio）和 holdout（后 holdout_ratio）
    2. 各基模型在 train1 上训练，预测 holdout → 生成元特征训练集
    3. 各基模型在全训练集重训，预测当期截面 → 生成元特征预测集
    4. meta_model 学习如何加权组合基模型，输出最终预测值

    Args:
        base_models: 基模型列表（如 LightGBM、Ridge、RandomForest）
        meta_model: 元模型，学习如何组合基模型（通常用简单线性模型）
        holdout_ratio: 训练集中留出生成元特征的比例，默认 0.2
    """

    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

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

    score_frames = []
    rebalance_dates = dates[cfg.backtest.train_window :: cfg.backtest.predict_window]

    print(f"Walk-forward Stacking（共 {len(rebalance_dates)} 期）...")
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

        split = int(len(X_train) * (1 - holdout_ratio))
        X_t1, y_t1 = X_train.iloc[:split], y_train.iloc[:split]
        X_ho, y_ho = X_train.iloc[split:], y_train.iloc[split:]

        meta_tr = np.zeros((len(X_ho), len(base_models)))
        meta_pr = np.zeros((len(X_pred), len(base_models)))

        for j, bm in enumerate(base_models):
            m1 = clone(bm)
            m1.fit(X_t1, y_t1)
            meta_tr[:, j] = m1.predict(X_ho)

            m2 = clone(bm)
            m2.fit(X_train, y_train)
            meta_pr[:, j] = m2.predict(X_pred)

        mm = clone(meta_model)
        mm.fit(meta_tr, y_ho)
        preds = pd.Series(mm.predict(meta_pr), index=X_pred.index, name=t)
        score_frames.append(preds)

        if i % 10 == 0:
            print(f"  [{i}/{len(rebalance_dates)}] {t.date()}")

    print("  训练完成")

    scores_wide = pd.DataFrame(score_frames)
    scores_wide.index = pd.DatetimeIndex(scores_wide.index)
    scores_wide = scores_wide.reindex(columns=close.columns)

    print("构造持仓...")
    daily_returns = close.pct_change()
    positions = scores_wide.apply(
        lambda row: factor_select(row.dropna(), top_n=cfg.backtest.top_n).reindex(
            close.columns, fill_value=0.0
        ),
        axis=1,
    )
    positions = positions.reindex(close.index).ffill().fillna(0.0)

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
    print(f"{'指标':<18} {'Stacking':>10} {'基准':>10}")
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
    print(f"\n期末净值  Stacking: {final_strategy:.3f}  基准: {final_bm:.3f}")
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
    run()
