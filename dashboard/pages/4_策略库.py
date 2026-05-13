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

st.set_page_config(page_title="策略库", layout="wide")

cfg = sidebar_config()

st.title("📊 策略库")

# --- 常量 ----------

_WINDOWED_FACTORS = [
    "动量",
    "RSI",
    "波动率",
    "均线偏离",
    "布林带位置",
    "偏度",
    "峰度",
    "特质波动率",
]
_DEFAULT_WINDOWS = {
    "动量": 20,
    "RSI": 14,
    "波动率": 20,
    "均线偏离": 20,
    "布林带位置": 20,
    "偏度": 20,
    "峰度": 20,
    "特质波动率": 20,
}

_MODEL_MAP = {
    "LightGBM": LGBMRegressor(n_estimators=cfg.ml.n_estimators, verbosity=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=cfg.ml.n_estimators, n_jobs=-1, random_state=42
    ),
    "Ridge": Ridge(alpha=cfg.ml.ridge_alpha),
    "XGBoost": XGBRegressor(n_estimators=cfg.ml.n_estimators, verbosity=0),
}

_IMG_DIR = Path(__file__).parent.parent.parent / "output" / "factor_research"
_PAIR_CSV = _IMG_DIR / "cointegrated_pairs.csv"


# 共用函数
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


@st.cache_data(show_spinner="因子回测运行中…")
def _run_factor_backtest(
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


@st.cache_data(show_spinner="ML 回测运行中，请稍候…")
def _run_ml_backtest(
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

    model = clone(_MODEL_MAP[model_name])
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


def _show_backtest_result(
    strategy_ret: pd.Series, benchmark_ret: pd.Series, label: str
) -> None:
    strategy_nav = (1 + strategy_ret).cumprod()
    benchmark_nav = (1 + benchmark_ret).cumprod()

    st.subheader("累计收益曲线")
    fig_nav = go.Figure()
    fig_nav.add_trace(
        go.Scatter(
            x=strategy_nav.index,
            y=strategy_nav.values,
            name=label,
            line=dict(color="#2196F3"),
        )
    )
    fig_nav.add_trace(
        go.Scatter(
            x=benchmark_nav.index,
            y=benchmark_nav.values,
            name="等权基准",
            line=dict(color="#9E9E9E", dash="dot"),
        )
    )
    fig_nav.update_layout(xaxis_title="日期", yaxis_title="净值", hovermode="x unified")
    st.plotly_chart(fig_nav, width="stretch")

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


tab_backtest, tab_arb = st.tabs(["📈 回测对比", "📐 统计套利"])

# Tab 1：回测对比
with tab_backtest:
    strategy_type = st.radio("策略类型", ["传统因子", "ML 模型"], horizontal=True)

    model_name = ""
    if strategy_type == "传统因子":
        col1, col2 = st.columns(2)
        with col1:
            factor_type = st.selectbox("因子类型", _WINDOWED_FACTORS + ["MACD"])
        with col2:
            if factor_type in _WINDOWED_FACTORS:
                factor_window = st.slider(
                    "因子窗口（天）",
                    5,
                    120,
                    _DEFAULT_WINDOWS.get(factor_type, 20),
                    step=5,
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
        model_name = st.selectbox("选择模型", list(_MODEL_MAP.keys()), index=1)
        factor_type = ""
        factor_window = 20
        macd_fast, macd_slow, macd_signal = 12, 26, 9
        run_label = f"🚀 运行回测（{model_name}）"

    if st.button(run_label, type="primary", key="btn_backtest"):
        if strategy_type == "传统因子":
            s_ret, b_ret, label = _run_factor_backtest(
                factor_type,
                factor_window,
                macd_fast,
                macd_slow,
                macd_signal,
                cfg.backtest.top_n,
                cfg.backtest.commission_rate,
            )
        else:
            s_ret, b_ret, label = _run_ml_backtest(
                model_name,
                cfg.backtest.top_n,
                cfg.backtest.train_window,
                cfg.backtest.commission_rate,
            )
        st.session_state["backtest_result"] = (s_ret, b_ret, label)

    if "backtest_result" in st.session_state:
        from typing import cast

        s_ret, b_ret, label = cast(
            tuple[pd.Series, pd.Series, str], st.session_state["backtest_result"]
        )
        _show_backtest_result(s_ret, b_ret, label)
    else:
        st.info(
            "调整参数后点击「运行回测」查看结果。"
            "传统因子策略约 10~30 秒，ML 策略约 1~2 分钟。"
        )

# Tab 2：统计套利
with tab_arb:
    arb1, arb2, arb3, arb4, arb5 = st.tabs(
        ["🔗 协整分析", "〰️ OU 过程", "📡 Kalman Filter", "🧩 PCA 篮子", "⚖️ 市场中性"]
    )

    with arb1:
        st.subheader("批量协整对筛选结果")
        if _PAIR_CSV.exists():
            df_pairs = pd.read_csv(_PAIR_CSV)
            st.dataframe(
                df_pairs.style.format(precision=4).background_gradient(
                    subset=["eg_pvalue"] if "eg_pvalue" in df_pairs.columns else [],
                    cmap="RdYlGn_r",
                ),
                width="stretch",
            )
        else:
            st.warning("请先运行 scripts/find_cointegrated_pairs.py")

        img_coint = _IMG_DIR / "cointegration_analysis.png"
        if img_coint.exists():
            st.image(str(img_coint), width="stretch")

        with st.expander("方法论：EG vs Johansen"):
            st.markdown("""
            | 方法 | 原理 | 优点 | 缺点 |
            |------|------|------|------|
            | Engle-Granger | OLS 残差 ADF 检验 | 简单直观 | 只能检测一个协整关系 |
            | Johansen | VAR 模型特征根 | 可检测多个协整关系，方向对称 | 参数敏感 |

            **实践建议**：两者同时通过才选为候选对。
            """)

    with arb2:
        st.subheader("OU 过程参数与半衰期分析")
        ou1, ou2, ou3 = st.columns(3)
        ou1.metric("最佳配对", "工商银行 vs 建设银行")
        ou2.metric("历史半衰期", "5~25 天（2019-2025）")
        ou3.metric("当前半衰期", "75+ 天（2026，不宜交易）")

        img_ou = _IMG_DIR / "ou_process_analysis.png"
        if img_ou.exists():
            st.image(str(img_ou), width="stretch")

        st.markdown("""
        **OU 过程参数解读：**
        - **θ（均值回归速度）**：越大回归越快，建议 θ > 0.03（对应半衰期 < 23天）
        - **半衰期 = ln(2)/θ**：策略持仓周期的理论上限
        - **σ（波动率）**：越大信噪比越高，开仓机会越多

        **当前状态（2026）**：θ 降至约 0.02，半衰期超 75 天，该对暂不适合配对交易。
        """)

    with arb3:
        st.subheader("动态对冲比率：Kalman Filter vs 静态 OLS")
        kf1, kf2, kf3 = st.columns(3)
        kf1.metric("静态 OLS ADF", "p≈0.05（边界）")
        kf2.metric("Kalman Filter ADF", "p<0.05（大部分时间）✅")
        kf3.metric("KF 对冲比率范围", "0.5~0.75（平滑跟踪）")

        img_kf = _IMG_DIR / "kalman_hedge_analysis.png"
        if img_kf.exists():
            st.image(str(img_kf), width="stretch")

        with st.expander("Kalman Filter 状态空间模型"):
            st.markdown(r"""
            **观测方程：** $\text{price\_a}_t = \beta_t \cdot \text{price\_b}_t
                        + \alpha_t + \varepsilon_t$

            **状态方程：** $[\beta_t, \alpha_t] = [\beta_{t-1}, \alpha_{t-1}] + w_t$

            其中 $w_t \sim \mathcal{N}(0, Q)$ 为过程噪声，控制对冲比率的漂移速度。

            **参数 delta**：$Q = \frac{\delta}{1-\delta} I$，delta=1e-4时β每日变动约 1%
            """)

    with arb4:
        st.subheader("PCA 篮子套利 — 实证发现")
        pca1, pca2 = st.columns(2)
        with pca1:
            st.error("均值回归方向：SR = -6.38（失败）")
        with pca2:
            st.success("动量方向（信号翻转）：SR = +6.38（有效）")

        img_pca = _IMG_DIR / "pca_basket_analysis.png"
        if img_pca.exists():
            st.image(str(img_pca), width="stretch")

        st.markdown("""
        **核心发现：A 股行业内特质残差具有动量特性**

        | 板块 | PC1 解释度 | 均值回归 SR | 动量 SR |
        |------|-----------|------------|---------|
        | 银行 | 65% | -6.70 | +6.70 |
        | 白酒 | 75% | -6.38 | +6.38 |

        **原因**：A 股中散户资金驱动的趋势效应强于机构套利带来的均值回归，
        行业内跑赢股票短期内倾向继续跑赢（"强者恒强"）。
        """)

    with arb5:
        st.subheader("Beta 中性组合 — 主力资金因子")
        mn1, mn2, mn3 = st.columns(3)
        mn1.metric("Sharpe Ratio", "0.91")
        mn2.metric("组合波动率", "~8%（年化）")
        mn3.metric("市场相关性", "≈ 0.1（中性化有效）✅")

        img_mn = _IMG_DIR / "market_neutral_analysis.png"
        if img_mn.exists():
            st.image(str(img_mn), width="stretch")

        st.markdown("""
        **Beta 中性化效果验证：**
        - 原始多空组合与市场相关性约 -0.3~-0.5（空头暴露过大）
        - Beta 中性后相关性压缩至 ≈ 0.1 ✅
        - 当前数据窗口仅 5 个月（fund_flow 历史不足），需补充历史数据

        **行业中性化**：当前使用代码前缀粗分，实盘应接入申万一级行业分类。
        """)
