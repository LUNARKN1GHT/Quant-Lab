import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from dashboard.shared import load_close, sidebar_config
from quant.backtest.engine import backtest
from quant.factor.bollinger import bollinger_position
from quant.factor.idiosyncratic_vol import idiosyncratic_vol
from quant.factor.ma_bias import ma_bias
from quant.factor.macd import macd
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.factor.skewness import kurtosis, skewness
from quant.factor.volatility import volatility
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select
from scripts.backtest_ml import build_features

st.set_page_config(page_title="回测对比", layout="wide")

cfg = sidebar_config()

st.title("📊 回测对比")

# ── 策略类型选择 ───────────────────────────────────────────────────────────
strategy_type = st.radio("策略类型", ["传统因子", "ML 模型"], horizontal=True)

WINDOWED_FACTORS = [
    "动量",
    "RSI",
    "波动率",
    "均线偏离",
    "布林带位置",
    "偏度",
    "峰度",
    "特质波动率",
]
DEFAULT_WINDOWS = {
    "动量": 20,
    "RSI": 14,
    "波动率": 20,
    "均线偏离": 20,
    "布林带位置": 20,
    "偏度": 20,
    "峰度": 20,
    "特质波动率": 20,
}

MODEL_MAP = {
    "LightGBM": LGBMRegressor(n_estimators=cfg.ml.n_estimators, verbosity=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=cfg.ml.n_estimators, n_jobs=-1, random_state=42
    ),
    "Ridge": Ridge(alpha=cfg.ml.ridge_alpha),
    "XGBoost": XGBRegressor(n_estimators=cfg.ml.n_estimators, verbosity=0),
}


def compute_factor_scores(
    name: str,
    close: pd.DataFrame,
    window: int,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """将因子名称分发到对应的计算函数，返回全股票因子值宽表。

    特质波动率需要市场等权收益率作为基准（CAPM 残差），
    其他因子只需单只股票收盘价序列。
    """
    market = close.mean(axis=1)  # 等权指数，用于特质波动率的 beta 估计
    if name == "动量":
        return close.apply(lambda s: momentum(s, window))
    if name == "RSI":
        return close.apply(lambda s: rsi(s, window))
    if name == "波动率":
        return close.apply(lambda s: volatility(s, window))
    if name == "均线偏离":
        return close.apply(lambda s: ma_bias(s, window))
    if name == "布林带位置":
        return close.apply(lambda s: bollinger_position(s, window))
    if name == "偏度":
        return close.apply(lambda s: skewness(s, window))
    if name == "峰度":
        return close.apply(lambda s: kurtosis(s, window))
    if name == "特质波动率":
        return close.apply(lambda s: idiosyncratic_vol(s, market, window))
    if name == "MACD":
        return close.apply(lambda s: macd(s, macd_fast, macd_slow, macd_signal))
    raise ValueError(f"未知因子: {name}")


# ── 策略参数面板 ───────────────────────────────────────────────────────────
if strategy_type == "传统因子":
    col1, col2 = st.columns(2)
    with col1:
        factor_type = st.selectbox("因子类型", WINDOWED_FACTORS + ["MACD"])
    with col2:
        if factor_type in WINDOWED_FACTORS:
            factor_window = st.slider(
                "因子窗口（天）", 5, 120, DEFAULT_WINDOWS.get(factor_type, 20), step=5
            )
            macd_fast, macd_slow, macd_signal = 12, 26, 9
        else:
            c1, c2, c3 = st.columns(3)
            macd_fast = c1.slider("Fast", 3, 30, 12, step=1)
            macd_slow = c2.slider("Slow", 10, 60, 26, step=1)
            macd_signal = c3.slider("Signal", 3, 20, 9, step=1)
            factor_window = macd_fast
    run_label = f"🚀 运行回测（{factor_type}）"
else:
    model_name = st.selectbox("选择模型", list(MODEL_MAP.keys()), index=1)
    factor_type = ""
    factor_window = 20
    macd_fast, macd_slow, macd_signal = 12, 26, 9
    run_label = f"🚀 运行回测（{model_name}）"

run_btn = st.button(run_label, type="primary")


# ── 传统因子回测 ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="因子回测运行中…")
def run_factor_backtest(
    factor_type: str,
    factor_window: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    top_n: int,
    commission: float,
    rebalance_days: int = 20,
) -> tuple[pd.Series, pd.Series, str]:
    """传统因子回测：每隔 rebalance_days 天重新选股，非调仓日持仓不变。

    Returns:
        (strategy_ret, benchmark_ret, label)
        benchmark 为成分股等权收益率，用于对比超额
    """
    close = load_close()
    scores = compute_factor_scores(
        factor_type, close, factor_window, macd_fast, macd_slow, macd_signal
    )
    daily_ret = close.pct_change()
    # 每隔 rebalance_days 取一个调仓日（等间隔调仓）
    rebalance_dates = scores.index[::rebalance_days]

    # 初始化全零持仓，调仓日写入新权重，其余日期通过 ffill 保持上期权重
    positions = pd.DataFrame(0.0, index=scores.index, columns=close.columns)
    for t in rebalance_dates:
        row = scores.loc[t].dropna()
        if len(row) < top_n:
            continue
        w = factor_select(row, top_n=top_n).reindex(close.columns, fill_value=0.0)
        positions.loc[t] = w

    positions = positions.ffill().fillna(0.0)
    strategy_ret = backtest(positions, daily_ret, commission_rate=commission)
    benchmark_ret = daily_ret.mean(axis=1)

    # 从第一个有效日期开始，过滤掉因子窗口预热期的 NaN
    start = strategy_ret.first_valid_index()
    label = f"{factor_type}({factor_window}日)"
    return (
        strategy_ret.loc[start:].fillna(0),  # type: ignore
        benchmark_ret.loc[start:].fillna(0),  # type: ignore
        label,
    )


# ── ML 回测 ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="ML 回测运行中，请稍候…")
def run_ml_backtest(
    model_name: str,
    top_n: int,
    train_window: int,
    commission: float,
) -> tuple[pd.Series, pd.Series, str]:
    """ML 模型回测：滚动训练预测下期收益，按预测值选股。

    流程：
    1. 用多因子合成特征矩阵（build_features）
    2. 滚动 walk-forward：每隔 predict_window 天重训一次模型
    3. 模型预测每只股票下期收益，按预测值排序选 Top-N
    4. 向量化回测计算策略收益
    """
    from sklearn.base import clone

    close = load_close()
    features = build_features(close)
    # 将收盘价转为前向收益（下期 predict_window 日的累积收益），stack 成长表
    fwd_ret = (
        close.pct_change(cfg.backtest.predict_window)
        .shift(-cfg.backtest.predict_window)
        .stack()
    )
    fwd_ret.index.names = ["date", "stock"]
    fwd_ret = fwd_ret.rename("fwd_ret")  # type: ignore
    dataset = features.join(fwd_ret).dropna()
    dates = dataset.index.get_level_values("date").unique().sort_values()

    model = clone(MODEL_MAP[model_name])
    score_frames = []
    # 每隔 predict_window 天重训一次，模拟实盘中的定期再训练
    rebalance_dates = dates[train_window :: cfg.backtest.predict_window]

    for t in rebalance_dates:
        # 取最近 train_window 个日期的历史数据作为训练集
        train_dates = dates[dates < t][-train_window:]
        if len(train_dates) < train_window // 2:
            continue
        if t not in dataset.index.get_level_values("date"):
            continue
        train_data = dataset.loc[train_dates]
        X_pred = dataset.loc[t].drop(columns="fwd_ret")
        fold_model = clone(model)
        fold_model.fit(train_data.drop(columns="fwd_ret"), train_data["fwd_ret"])
        preds = pd.Series(fold_model.predict(X_pred), index=X_pred.index, name=t)
        score_frames.append(preds)

    # 将每期预测结果拼成日期×股票的宽表
    scores_wide = pd.DataFrame(score_frames)
    scores_wide.index = pd.DatetimeIndex(scores_wide.index)
    scores_wide = scores_wide.reindex(columns=close.columns)

    daily_returns = close.pct_change()
    positions = scores_wide.apply(
        lambda row: factor_select(row.dropna(), top_n=top_n).reindex(
            close.columns, fill_value=0.0
        ),
        axis=1,
    )
    # reindex + ffill：调仓日之间持仓保持不变
    positions = positions.reindex(close.index).ffill().fillna(0.0)
    strategy_ret = backtest(positions, daily_returns, commission_rate=commission)
    benchmark_ret = daily_returns.mean(axis=1)

    start = strategy_ret.first_valid_index()
    return (
        strategy_ret.loc[start:].fillna(0),  # type: ignore
        benchmark_ret.loc[start:].fillna(0),  # type: ignore
        model_name,
    )


# ── 结果展示 ───────────────────────────────────────────────────────────────
if run_btn:
    if strategy_type == "传统因子":
        strategy_ret, benchmark_ret, label = run_factor_backtest(
            factor_type,
            factor_window,
            macd_fast,
            macd_slow,
            macd_signal,
            cfg.backtest.top_n,
            cfg.backtest.commission_rate,
        )
    else:
        strategy_ret, benchmark_ret, label = run_ml_backtest(
            model_name,
            cfg.backtest.top_n,
            cfg.backtest.train_window,
            cfg.backtest.commission_rate,
        )

    strategy_nav = (1 + strategy_ret).cumprod()
    benchmark_nav = (1 + benchmark_ret).cumprod()

    st.subheader("累计收益曲线")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_nav.index,
            y=strategy_nav.values,
            name=label,
            line=dict(color="#2196F3"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_nav.index,
            y=benchmark_nav.values,
            name="等权基准",
            line=dict(color="#9E9E9E", dash="dot"),
        )
    )
    fig.update_layout(xaxis_title="日期", yaxis_title="净值", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    st.subheader("回撤曲线")
    rolling_max = strategy_nav.cummax()
    drawdown = (strategy_nav - rolling_max) / rolling_max
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            fill="tozeroy",
            fillcolor="rgba(244,67,54,0.2)",
            line=dict(color="#F44336"),
            name="回撤",
        )
    )
    fig_dd.update_layout(
        yaxis_tickformat=".0%", xaxis_title="日期", yaxis_title="回撤幅度"
    )
    st.plotly_chart(fig_dd, width="stretch")

    st.subheader("风险收益指标")
    metrics = pd.DataFrame(
        {
            label: {
                "年化收益": f"{strategy_ret.mean() * 252:.2%}",
                "Sharpe": f"{sharpe(strategy_ret):.3f}",
                "Sortino": f"{sortino(strategy_ret):.3f}",
                "最大回撤": f"{max_drawdown(strategy_ret):.2%}",
                "Calmar": f"{calmar(strategy_ret):.3f}",
                "期末净值": f"{strategy_nav.iloc[-1]:.3f}",
            },
            "等权基准": {
                "年化收益": f"{benchmark_ret.mean() * 252:.2%}",
                "Sharpe": f"{sharpe(benchmark_ret):.3f}",
                "Sortino": f"{sortino(benchmark_ret):.3f}",
                "最大回撤": f"{max_drawdown(benchmark_ret):.2%}",
                "Calmar": f"{calmar(benchmark_ret):.3f}",
                "期末净值": f"{benchmark_nav.iloc[-1]:.3f}",
            },
        }
    )
    st.table(metrics)
else:
    st.info(
        "调整参数后点击「运行回测」查看结果。"
        "传统因子策略约 10–30 秒，ML 策略约 1–2 分钟。"
    )
