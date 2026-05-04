"""我的基金持仓看板"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from dashboard.shared import sidebar_config
from quant.fund.portfolio import (
    fund_returns,
    load_nav_matrix,
    portfolio_value,
    risk_metrics,
)

st.set_page_config(page_title="我的持仓", layout="wide")
sidebar_config()
st.title("💼 我的基金持仓")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "quant.duckdb"
HOLDINGS_PATH = Path(__file__).parent.parent.parent / "configs" / "my_holdings.yaml"

if not HOLDINGS_PATH.exists():
    st.error("请先创建 configs/my_holdings.yaml 填入持仓信息")
    st.stop()

with open(HOLDINGS_PATH) as f:
    cfg = yaml.safe_load(f)
holdings = cfg["holdings"]
symbols = [h["symbol"] for h in holdings]
names = {h["symbol"]: h.get("name", h["symbol"]) for h in holdings}


@st.cache_data(ttl=3600, show_spinner="加载净值数据...")
def load_data():
    con = duckdb.connect(str(DB_PATH))
    nav = load_nav_matrix(con, symbols)
    con.close()
    return nav


try:
    nav = load_data()
except Exception as e:
    st.error(f"数据加载失败：{e}，请先运行 scripts/download_fund_nav.py")
    st.stop()

returns = fund_returns(nav)

# ── 持仓总览 ──────────────────────────────────────────────────────────────────
st.subheader("持仓总览")
pv = portfolio_value(nav, holdings)
total_cost = pv["cost_value"].sum()
total_latest = pv["latest_value"].sum()
total_pnl = total_latest - total_cost
total_pnl_pct = total_pnl / total_cost

c1, c2, c3, c4 = st.columns(4)
c1.metric("持仓基金数", len(pv))
c2.metric("总成本", f"¥{total_cost:,.0f}")
c3.metric("当前市值", f"¥{total_latest:,.0f}")
c4.metric("总浮盈亏", f"¥{total_pnl:+,.0f}", f"{total_pnl_pct:+.2%}")

# 各基金明细
pv_display = pv[
    [
        "name",
        "shares",
        "cost_nav",
        "latest_nav",
        "cost_value",
        "latest_value",
        "pnl",
        "pnl_pct",
    ]
].copy()
pv_display.columns = [
    "名称",
    "份额",
    "成本净值",
    "最新净值",
    "成本市值",
    "当前市值",
    "浮盈亏",
    "收益率",
]
styled = pv_display.style.format(
    {
        "份额": "{:,.0f}",
        "成本净值": "{:.4f}",
        "最新净值": "{:.4f}",
        "成本市值": "¥{:,.0f}",
        "当前市值": "¥{:,.0f}",
        "浮盈亏": "¥{:+,.0f}",
        "收益率": "{:+.2%}",
    }
).map(lambda v: "color:green" if v > 0 else "color:red", subset=["浮盈亏", "收益率"])
st.dataframe(styled, use_container_width=True)

st.divider()
tab1, tab2, tab3 = st.tabs(["📈 净值走势", "📊 风险指标", "🔗 相关性"])

# ── Tab 1：净值走势 ────────────────────────────────────────────────────────────
with tab1:
    period = st.selectbox("时间范围", ["成立来", "近3年", "近1年", "近6月", "近3月"])
    n_map = {"近3月": 63, "近6月": 126, "近1年": 252, "近3年": 756, "成立来": 99999}
    n = n_map[period]

    # 归一化净值（以各自起点=1）
    fig = go.Figure()
    for sym in symbols:
        if sym not in nav.columns:
            continue
        s = nav[sym].dropna().tail(n)
        norm = s / s.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm.values,
                name=names.get(sym, sym),
                mode="lines",
            )
        )
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="归一化净值走势（各自起点=1）",
        yaxis_title="相对净值",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2：风险指标 ────────────────────────────────────────────────────────────
with tab2:
    rows = []
    for sym in symbols:
        if sym not in returns.columns:
            continue
        m = risk_metrics(returns[sym])
        m["基金"] = names.get(sym, sym)
        rows.append(m)
    df_risk = pd.DataFrame(rows).set_index("基金")
    st.dataframe(
        df_risk.style.format(
            {
                "年化收益": "{:.2%}",
                "年化波动": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "最大回撤": "{:.2%}",
                "Calmar": "{:.2f}",
                "近1月": "{:.2%}",
                "近3月": "{:.2%}",
                "近1年": "{:.2%}",
            }
        ).background_gradient(subset=["Sharpe"], cmap="RdYlGn"),
        use_container_width=True,
    )

# ── Tab 3：相关性 ──────────────────────────────────────────────────────────────
with tab3:
    window = st.slider("滚动相关性窗口（天）", 20, 120, 60, step=10)
    common_ret = returns[symbols].dropna()
    if common_ret.shape[1] >= 2:
        import plotly.express as px

        corr = common_ret.rename(columns=names).corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".3f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="基金间收益相关性矩阵",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if common_ret.shape[1] == 2:
            s1, s2 = symbols[0], symbols[1]
            roll_corr = common_ret[s1].rolling(window).corr(common_ret[s2])
            fig_rc = go.Figure()
            fig_rc.add_trace(
                go.Scatter(
                    x=roll_corr.index,
                    y=roll_corr.values,
                    fill="tozeroy",
                    line=dict(color="steelblue"),
                )
            )
            fig_rc.add_hline(y=0, line_color="black", line_width=0.8)
            fig_rc.update_layout(
                title=f"滚动{window}日相关性：{names.get(s1, s1)} vs {names.get(s2, s2)}",
                yaxis=dict(range=[-1, 1]),
            )
            st.plotly_chart(fig_rc, use_container_width=True)
    else:
        st.info("至少需要2只基金才能计算相关性")
