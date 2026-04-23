import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.regime.detector import detect_regime

st.set_page_config(page_title="市场状态", layout="wide")

cfg = sidebar_config()
close = load_close()

regime = detect_regime(
    close,
    ma_window=cfg.regime.ma_window,
    breadth_window=cfg.regime.breadth_window,
    vol_short=cfg.regime.vol_short,
    vol_long=cfg.regime.vol_long,
)

st.title("🌡️ 市场状态")

# 当前状态
latest_date = regime.index[-1]
latest = regime.iloc[-1]
color_map = {"BULL": "🟢", "RANGE": "🟡", "BEAR": "🔴"}
st.subheader(f"当前状态（{latest_date.date()}）：{color_map.get(latest, '')} {latest}")

# 状态分布 + 年度统计
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("历史状态分布")
    counts = regime.value_counts()
    fig_pie = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map={"BULL": "#4CAF50", "RANGE": "#FF9800", "BEAR": "#F44336"},
    )
    st.plotly_chart(fig_pie, width="stretch")

with col2:
    st.subheader("各年度状态占比")
    regime_df = regime.to_frame("regime")
    regime_df["year"] = regime_df.index.year
    yearly = regime_df.groupby(["year", "regime"]).size().unstack(fill_value=0)
    for s in ["BULL", "RANGE", "BEAR"]:
        if s not in yearly.columns:
            yearly[s] = 0
    yearly_pct = yearly[["BULL", "RANGE", "BEAR"]].div(yearly.sum(axis=1), axis=0)

    fig_bar = go.Figure()
    for state, color in {
        "BULL": "#4CAF50",
        "RANGE": "#FF9800",
        "BEAR": "#F44336",
    }.items():
        fig_bar.add_trace(
            go.Bar(
                x=yearly_pct.index,
                y=yearly_pct[state],
                name=state,
                marker_color=color,
            )
        )
    fig_bar.update_layout(
        barmode="stack",
        yaxis_tickformat=".0%",
        xaxis_title="年份",
        yaxis_title="占比",
    )
    st.plotly_chart(fig_bar, width="stretch")

# 近120日走势（指数均值 + 状态着色）
st.subheader("近120日市场状态")
index_close = close.mean(axis=1).tail(120)
recent_regime = regime.tail(120)

fig_line = go.Figure()
fig_line.add_trace(
    go.Scatter(
        x=index_close.index,
        y=index_close.values,
        name="指数均值",
        line=dict(color="#2196F3"),
    )
)
for state, color in {"BULL": "#4CAF50", "RANGE": "#FF9800", "BEAR": "#F44336"}.items():
    mask = recent_regime == state
    fig_line.add_trace(
        go.Scatter(
            x=index_close[mask].index,
            y=index_close[mask].values,
            mode="markers",
            name=state,
            marker=dict(color=color, size=5),
        )
    )
fig_line.update_layout(xaxis_title="日期", yaxis_title="价格", hovermode="x unified")
st.plotly_chart(fig_line, width="stretch")
