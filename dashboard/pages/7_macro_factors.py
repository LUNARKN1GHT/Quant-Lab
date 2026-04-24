import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import load_close, sidebar_config
from quant.macro.indicators import calc_lag_corr, composite_index
from quant.macro.loader import load_all_macro

st.set_page_config(page_title="宏观因子", layout="wide")
cfg = sidebar_config()
st.title("🌐 宏观因子")

load_btn = st.button("📥 加载宏观数据", type="primary")

LABEL_MAP = {
    "bond_yield": "十年期国债收益率",
    "pmi": "制造业 PMI",
    "m2_yoy": "M2 同比",
    "cpi_yoy": "CPI 同比",
}


@st.cache_data(show_spinner="加载宏观数据...")
def get_macro():
    return load_all_macro()


@st.cache_data(show_spinner="计算景气度指数…")
def get_composite(macro_df):
    return composite_index(macro_df)


@st.cache_data(show_spinner="计算滞后相关性…")
def get_lag_corrs(macro_df):
    close = load_close()
    market_ret = close.pct_change().mean(axis=1)
    return {col: calc_lag_corr(macro_df[col], market_ret) for col in macro_df.columns}


if load_btn:
    st.session_state["macro_df"] = get_macro()

if "macro_df" in st.session_state:
    macro_df: pd.DataFrame = st.session_state["macro_df"]
    score = get_composite(macro_df)
    lag_corrs = get_lag_corrs(macro_df=macro_df)

    # ----- 景气度指数 -----
    st.subheader("宏观景气度合成指数")
    st.caption("PMI + M2 + CPI 正向，国债收益率反向，z-score 标准化后等权合成")
    fig_score = go.Figure()
    fig_score.add_trace(
        go.Scatter(
            x=score.index,
            y=score.values,
            mode="lines",
            line=dict(color="#2196F3", width=2),
            name="景气度",
        )
    )
    fig_score.add_hline(y=0, line_color="gray", line_width=0.8, line_dash="dash")
    fig_score.update_layout(
        xaxis_title="日期", yaxis_title="景气度得分", hovermode="x unified"
    )
    st.plotly_chart(fig_score, width="stretch")

    # ----- 各指标走势 -----
    st.subheader("各宏观指标走势")
    cols = list(macro_df.columns)
    fig_sub = make_subplots(rows=2, cols=2, subplot_titles=[LABEL_MAP[c] for c in cols])
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for i, (col, color) in enumerate(zip(cols, colors)):
        row, col_idx = divmod(i, 2)
        fig_sub.add_trace(
            go.Scatter(
                x=macro_df.index,
                y=macro_df[col],
                mode="lines",
                line=dict(color=color, width=1.5),
                name=LABEL_MAP[col],
                showlegend=False,
            ),
            row=row + 1,
            col=col_idx + 1,
        )
    fig_sub.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig_sub, width="stretch")

    # ----- 滞后相关性 -----
    st.subheader("宏观指标与大盘的滞后相关性")
    st.caption("lag=N 表示该指标领先大盘 N 个月时的 Pearson 相关系数")
    fig_lag = go.Figure()
    for col, corr_series in lag_corrs.items():
        fig_lag.add_trace(
            go.Bar(
                x=corr_series.index,
                y=corr_series.values,
                name=LABEL_MAP.get(col, col),
            )
        )
    fig_lag.add_hline(y=0, line_color="gray", line_width=0.8)
    fig_lag.update_layout(
        barmode="group",
        xaxis_title="滞后月数",
        yaxis_title="相关系数",
        hovermode="x unified",
    )
    st.plotly_chart(fig_lag, width="stretch")

    # ----- 最新宏观快照 -----
    st.subheader("最新宏观快照")
    latest = macro_df.dropna(how="all").iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(LABEL_MAP["bond_yield"], f"{latest['bond_yield']:.2f}%")
    c2.metric(LABEL_MAP["pmi"], f"{latest['pmi']:.1f}")
    c3.metric(LABEL_MAP["m2_yoy"], f"{latest['m2_yoy']:.1f}%")
    c4.metric(LABEL_MAP["cpi_yoy"], f"{latest['cpi_yoy']:.1f}%")

else:
    st.info("点击「加载宏观数据」开始，首次约需 10~30 秒。")
