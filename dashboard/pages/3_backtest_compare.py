import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from dashboard.shared import load_close, sidebar_config
from quant.backtest.engine import backtest
from quant.risk.metrics import calmar, max_drawdown, sharpe, sortino
from quant.strategy.factor_strategy import factor_select
from scripts.backtest_ml import build_features

st.set_page_config(page_title="回测对比", layout="wide")

cfg = sidebar_config()
close = load_close()

st.title("📊 回测对比")

# 模型选择
model_name = st.selectbox(
    "选择模型",
    ["LightGBM", "RandomForest", "Ridge", "XGBoost"],
    index=1,
)

MODEL_MAP = {
    "LightGBM": LGBMRegressor(n_estimators=cfg.ml.n_estimators, verbosity=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=cfg.ml.n_estimators, n_jobs=-1, random_state=42
    ),
    "Ridge": Ridge(alpha=cfg.ml.ridge_alpha),
    "XGBoost": XGBRegressor(n_estimators=cfg.ml.n_estimators, verbosity=0),
}

run_btn = st.button("🚀 运行回测", type="primary")


@st.cache_data(show_spinner="回测运行中，请稍候…")
def run_backtest(model_name: str, top_n: int, train_window: int, commission: float):
    import pandas as pd
    from sklearn.base import clone

    close_ = load_close()
    features = build_features(close_)

    fwd_ret = (
        close_.pct_change(cfg.backtest.predict_window)
        .shift(-cfg.backtest.predict_window)
        .stack()
    )
    fwd_ret.index.names = ["date", "stock"]
    fwd_ret.name = "fwd_ret"
    dataset = features.join(fwd_ret).dropna()
    dates = dataset.index.get_level_values("date").unique().sort_values()

    model = clone(MODEL_MAP[model_name])
    score_frames = []
    rebalance_dates = dates[train_window :: cfg.backtest.predict_window]

    for t in rebalance_dates:
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

    scores_wide = pd.DataFrame(score_frames)
    scores_wide.index = pd.DatetimeIndex(scores_wide.index)
    scores_wide = scores_wide.reindex(columns=close_.columns)

    daily_returns = close_.pct_change()
    positions = scores_wide.apply(
        lambda row: factor_select(row.dropna(), top_n=top_n).reindex(
            close_.columns, fill_value=0.0
        ),
        axis=1,
    )
    positions = positions.reindex(close_.index).ffill().fillna(0.0)
    strategy_ret = backtest(positions, daily_returns, commission_rate=commission)
    benchmark_ret = daily_returns.mean(axis=1)

    start = strategy_ret.first_valid_index()
    strategy_ret = strategy_ret.loc[start:].fillna(0)
    benchmark_ret = benchmark_ret.loc[start:].fillna(0)
    return strategy_ret, benchmark_ret


if run_btn:
    strategy_ret, benchmark_ret = run_backtest(
        model_name,
        cfg.backtest.top_n,
        cfg.backtest.train_window,
        cfg.backtest.commission_rate,
    )

    strategy_nav = (1 + strategy_ret).cumprod()
    benchmark_nav = (1 + benchmark_ret).cumprod()

    # 累计收益曲线
    st.subheader("累计收益曲线")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=strategy_nav.index,
            y=strategy_nav.values,
            name=f"{model_name}",
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

    # 回撤曲线
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

    # 指标对比
    st.subheader("风险收益指标")
    import pandas as pd

    metrics = pd.DataFrame(
        {
            model_name: {
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
    st.info("调整左侧参数后点击「运行回测」查看结果。首次运行约需 1-2 分钟。")
