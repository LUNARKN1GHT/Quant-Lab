import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.advisor.position import compute_position

st.set_page_config(page_title="仓位建议", layout="wide")

cfg = sidebar_config()
close = load_close()
result = compute_position(close, cfg)

# 最新建议卡片
latest = result.iloc[-1]
st.subheader(f"最新建议 ({result.index[-1].date()})")
col1, col2, col3 = st.columns(3)
col1.metric("Regime 仓位", f"{latest['regime_scale']:.0%}")
col2.metric("波动率仓位", f"{latest['vol_scale']:.0%}")
col3.metric(
    "最终建议仓位",
    f"{latest['position']:.0%}",
    delta=f"{latest['position'] - result['position'].iloc[-2]:.0%}",
)


# 历史仓位走势
st.subheader("历史仓位走势")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["position"],
        name="建议仓位",
        line=dict(color="#2196F3"),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
    )
)
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["regime_scale"],
        name="Regime分量",
        line=dict(dash="dot", color="#FF9800"),
    )
)
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["vol_scale"],
        name="波动率分量",
        line=dict(dash="dot", color="#9C27B0"),
    )
)
fig.update_layout(
    yaxis_tickformat=".0%",
    hovermode="x unified",
    yaxis_title="仓位比例",
    xaxis_title="日期",
)
st.plotly_chart(fig, use_container_width=True)

# Regime 切换记录
st.subheader("Regime 切换记录（近60日）")
recent = result.tail(60)
changes = recent[recent["regime"] != recent["regime"].shift()][["regime", "position"]]
st.dataframe(changes.style.format({"position": "{:.0%}"}), use_container_width=True)
