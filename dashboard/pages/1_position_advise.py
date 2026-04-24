import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.advisor.position import compute_position
from quant.macro.indicators import composite_index
from quant.macro.loader import load_all_macro
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

st.set_page_config(page_title="仓位建议", layout="wide")

cfg = sidebar_config()
close = load_close()


@st.cache_data(show_spinner="加载宏观景气度...")
def get_macro_score() -> pd.Series:
    macro_df = load_all_macro()
    return composite_index(macro_df)


macro_score = get_macro_score()
result = compute_position(close, cfg, macro_score=macro_score)

# 最新建议卡片
latest = result.iloc[-1]
st.subheader(f"最新建议 ({result.index[-1].date()})")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Regime 仓位", f"{latest['regime_scale']:.0%}")
col2.metric("波动率仓位", f"{latest['vol_scale']:.0%}")
col3.metric("宏观乘数", f"{latest['macro_multiplier']:.2f}x")
col4.metric(
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
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["macro_multiplier"],
        name="宏观乘数",
        line=dict(dash="dot", color="#4CAF50"),
    )
)
fig.update_layout(
    yaxis_tickformat=".0%",
    hovermode="x unified",
    yaxis_title="仓位比例",
    xaxis_title="日期",
)
st.plotly_chart(fig, width="stretch")

# Regime 切换记录
st.subheader("Regime 切换记录（近60日）")
recent = result.tail(60)
changes = recent[recent["regime"] != recent["regime"].shift()][["regime", "position"]]
st.dataframe(changes.style.format({"position": "{:.0%}"}), width="stretch")

# 行业建议
st.subheader("当前行业超配建议")
st.caption("基于申万行业 RS + 动量，首次加载约需 1~2 分钟")

load_sector_btn = st.button("📥 加载行业信号", key="sector_btn")
if load_sector_btn:
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)
    rs = calc_rs(sector_close, benchmark, window=20)
    rs_momentum = calc_rs_momentum(rs, lookback=20)
    suggestions = get_suggestions(rs.iloc[-1], rs_momentum, top_n=3)
    st.session_state["sector_suggestions"] = suggestions

if "sector_suggestions" in st.session_state:
    sug = st.session_state["sector_suggestions"]
    overweight = sug[sug["建议"] == "超配 ▲"][["RS", "RS动量", "建议"]]
    underweight = sug[sug["建议"] == "低配 ▼"][["RS", "RS动量", "建议"]]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**超配行业**")
        st.dataframe(
            overweight.style.format({"RS": "{:.3f}", "RS动量": "{:.4f}"}),
            width="stretch",
        )
    with c2:
        st.markdown("**低配行业**")
        st.dataframe(
            underweight.style.format({"RS": "{:.3f}", "RS动量": "{:.4f}"}),
            width="stretch",
        )
