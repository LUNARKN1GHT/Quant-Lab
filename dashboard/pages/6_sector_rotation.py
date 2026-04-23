import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import sidebar_config
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

st.set_page_config(page_title="行业轮动", layout="wide")
sidebar_config()
st.title("🔄 行业轮动")

rs_window = st.slider("RS 计算窗口（天）", 5, 60, 20, step=5)
lookback = st.slider("RS 动量回看（天）", 10, 60, 20, step=5)
top_n = st.slider("超配/低配各 N 个", 1, 5, 3)

load_btn = st.button("📥 加载行业数据", type="primary")


@st.cache_data(show_spinner="加载申万行业数据…")
def get_data():
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)  # 等权行业指数作基准
    return sector_close, benchmark


if load_btn:
    sector_close, benchmark = get_data()
    rs = calc_rs(sector_close, benchmark, window=rs_window)
    rs_momentum = calc_rs_momentum(rs, lookback=lookback)
    rs_latest = rs.iloc[-1]
    suggestions = get_suggestions(rs_latest, rs_momentum, top_n=top_n)

    # RS 热力图（近 12 个月月末）
    st.subheader("行业 RS 热力图（近12个月）")
    rs_monthly = rs.resample("ME").last().tail(12).T
    fig_heat = px.imshow(
        rs_monthly,
        color_continuous_scale="RdYlGn",
        zmin=0.5,
        zmax=1.5,
        text_auto=".2f",  # type: ignore
        aspect="auto",
        labels={"color": "RS"},
    )
    st.plotly_chart(fig_heat, width="stretch")

    # 当前 RS 排名
    st.subheader("当前 RS 排名")
    rs_sorted = rs_latest.sort_values()
    fig_bar = go.Figure(
        go.Bar(
            x=rs_sorted.values,
            y=rs_sorted.index,
            orientation="h",
            marker_color=["#4CAF50" if v > 1 else "#F44336" for v in rs_sorted.values],
        )
    )
    fig_bar.add_vline(x=1, line_dash="dash", line_color="gray")
    fig_bar.update_layout(xaxis_title="RS（>1 强于基准）", height=500)
    st.plotly_chart(fig_bar, width="stretch")

    # 超配/低配建议表
    st.subheader("超配/低配建议")
    st.dataframe(
        suggestions.style.format(
            {
                "RS": "{:.3f}",
                "RS动量": "{:.4f}",
                "RS排名": "{:.0f}",
                "动量排名": "{:.0f}",
                "综合排名": "{:.1f}",
            }
        ),
        width="stretch",
    )
else:
    st.info(
        "点击「加载行业数据」开始，"
        + "首次需拉取所有行业历史数据约需 1-2 分钟，之后走本地缓存。"
    )
