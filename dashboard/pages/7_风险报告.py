"""风险报告页"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from quant.risk.report import risk_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from dashboard.shared import load_close, sidebar_config

st.set_page_config(page_title="风险报告", layout="wide")
cfg = sidebar_config()
st.title("📊 风险报告")

close = load_close()


@st.cache_data(show_spinner="计算风险指标...")
def _get_report(lookback: int) -> dict:
    eq_index = close.mean(axis=1)
    returns = eq_index.pct_change().dropna()
    if lookback > 0:
        returns = returns.iloc[-lookback:]
    return risk_report(returns)


# --- 控制栏 ----------
lookback = st.selectbox(
    "回看区间",
    options=[0, 252, 504, 756],
    format_func=lambda x: "全部" if x == 0 else f"近 {x // 252} 年",
    index=0,
)

report = _get_report(lookback=lookback)

# --- 顶部指标卡 ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("年化收益", f"{report['annual_return']:.1%}")
col2.metric("年化波动", f"{report['annual_vol']:.1%}")
col3.metric("Sharpe", f"{report['sharpe']:.2f}")
col4.metric("Sortino", f"{report['sortino']:.2f}")
col5.metric("最大回撤", f"{report['max_drawdown']:.1%}")
col6.metric("Calmar", f"{report['calmar']:.2f}")

st.divider()

# --- 回撤走势图 ----------
st.subheader("水下曲线（回撤走势）")
dd: pd.Series = report["drawdown_series"]
fig_dd = go.Figure(
    go.Scatter(
        x=dd.index,
        y=dd.values,
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.15)",
        line=dict(color="#EF4444", width=1),
        name="回撤",
    )
)
fig_dd.update_layout(
    yaxis_tickformat=".0%",
    height=280,
    margin=dict(t=10, b=20),
    hovermode="x unified",
    yaxis_title="回撤幅度",
    xaxis_title="日期",
)
st.plotly_chart(fig_dd, width="stretch")

# --- 水下统计 + 尾部风险 ---------
st.subheader("风险明细")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("最长连续水下天数", f"{report['max_underwater_days']} 天")
avg_dd = report.get("avg_drawdown")
col_b.metric(
    "平均回撤深度",
    f"{avg_dd:.1%}" if avg_dd is not None and not pd.isna(avg_dd) else "—",
)
col_c.metric("VaR（95%，单日）", f"{report['var_95']:.2%}")
col_d.metric("CVaR（95%，单日）", f"{report['cvar_95']:.2%}")

# --- 收益分布直方图 ----------
st.subheader("日收益分布")
eq_index = close.mean(axis=1)
returns_full = eq_index.pct_change().dropna()
returns_plot = returns_full.iloc[-lookback:] if lookback > 0 else returns_full

fig_hist = go.Figure(
    go.Histogram(
        x=returns_plot.values,
        nbinsx=80,
        marker_color="#2196F3",
        opacity=0.75,
        name="日收益",
    )
)
fig_hist.add_vline(
    x=report["var_95"],
    line_dash="dash",
    line_color="#EF4444",
    annotation_text=f"VaR {report['var_95']:.2%}",
    annotation_position="top right",
)
fig_hist.add_vline(
    x=report["cvar_95"],
    line_dash="dot",
    line_color="#B91C1C",
    annotation_text=f"CVaR {report['cvar_95']:.2%}",
    annotation_position="top left",
)
fig_hist.update_layout(
    xaxis_tickformat=".1%",
    height=300,
    margin=dict(t=20, b=20),
    hovermode="x",
    xaxis_title="日收益率",
    yaxis_title="频次",
)
st.plotly_chart(fig_hist, width="stretch")
