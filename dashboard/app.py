import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.advisor.position import compute_position
from quant.macro.indicators import composite_index
from quant.macro.loader import load_all_macro
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

st.set_page_config(page_title="Quant-Lab", page_icon="📈", layout="wide")

cfg = sidebar_config()

_REGIME_ICON = {"BULL": "🟢", "RANGE": "🟡", "BEAR": "🔴"}
_REGIME_LABEL = {"BULL": "上行趋势", "RANGE": "震荡区间", "BEAR": "下行趋势"}
_REGIME_COLOR = {
    "BULL": "rgba(76,175,80,0.12)",
    "RANGE": "rgba(255,152,0,0.12)",
    "BEAR": "rgba(244,67,54,0.12)",
}
_MACRO_LABEL = {
    "bond_yield": "国债收益率",
    "pmi": "PMI",
    "m2_yoy": "M2 同比",
    "cpi_yoy": "CPI 同比",
}


@st.cache_data(show_spinner="加载宏观数据…")
def _load_macro() -> tuple[pd.Series, pd.DataFrame]:
    macro_df = load_all_macro()
    return composite_index(macro_df), macro_df


@st.cache_data(show_spinner="加载行业数据…")
def _load_sector_top(top_n: int = 3) -> pd.DataFrame:
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)
    rs = calc_rs(sector_close, benchmark, window=20)
    rs_mom = calc_rs_momentum(rs, lookback=20)
    return get_suggestions(rs.iloc[-1], rs_mom, top_n=top_n)


# ── 数据加载 ─────────────────────────────────────────────────────────────────
close = load_close()
macro_score, macro_df = _load_macro()
result = compute_position(close=close, cfg=cfg, macro_score=macro_score)

latest = result.iloc[-1]
latest_date = result.index[-1]
regime = str(latest["regime"])
macro_latest = float(macro_score.dropna().iloc[-1])

# ── 标题 ─────────────────────────────────────────────────────────────────────
st.title("📈 Quant-Lab 驾驶舱")
st.caption(f"数据截至 {latest_date.date()}　｜　沪深300成分股 {close.shape[1]} 只")
st.divider()

# ── 第一行：核心指标 ──────────────────────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("市场状态", f"{_REGIME_ICON.get(regime, '')} {regime}")
m2.metric("建议仓位", f"{latest['position']:.0%}")
m3.metric("Regime 分量", f"{latest['regime_signal']:.0%}")
m4.metric("波动率信号", f"{latest['vol_signal']:.0%}")
m5.metric("宏观信号", f"{latest['macro_signal']:.0f}x")
m6.metric("宏观景气", f"{macro_latest:+.2f}")

st.divider()

# ── 第二行：图表 + 行业信号（等高） ──────────────────────────────────────────
col_chart, col_sector = st.columns([3, 2])

with col_chart:
    st.subheader("近 90 日仓位走势")
    recent = result.tail(90)

    fig = go.Figure()

    # Regime 背景色带（vrect 保证矩形）
    current_regime: str | None = None
    seg_start = recent.index[0]
    for date, row in recent.iterrows():
        r = str(row["regime"])
        if r != current_regime:
            if current_regime is not None:
                fig.add_vrect(
                    x0=seg_start,
                    x1=date,
                    fillcolor=_REGIME_COLOR.get(current_regime, "rgba(0,0,0,0.05)"),
                    layer="below",
                    line_width=0,
                )
            current_regime = r
            seg_start = date
    if current_regime is not None:
        fig.add_vrect(
            x0=seg_start,
            x1=recent.index[-1],
            fillcolor=_REGIME_COLOR.get(current_regime, "rgba(0,0,0,0.05)"),
            layer="below",
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=recent.index,
            y=recent["position"],
            name="建议仓位",
            fill="tozeroy",
            fillcolor="rgba(33,150,243,0.08)",
            line=dict(color="#2196F3", width=2),
        )
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis=dict(range=[0, 1.05]),
        hovermode="x unified",
        height=260,
        margin=dict(t=10, b=10, l=0, r=0),
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

with col_sector:
    st.subheader("行业信号")
    if st.button("📥 加载行业数据", key="btn_sector_home"):
        st.session_state["home_sector"] = _load_sector_top(top_n=3)

    if "home_sector" in st.session_state:
        sug: pd.DataFrame = st.session_state["home_sector"]
        overweight = sug[sug["建议"].str.contains("超配", na=False)]
        underweight = sug[sug["建议"].str.contains("低配", na=False)]

        if not overweight.empty:
            st.markdown("**超配 ▲**")
            for _, row in overweight.iterrows():
                st.success(f"{row.name}　RS = {row['RS']:.2f}", icon=None)
        if not underweight.empty:
            st.markdown("**低配 ▼**")
            for _, row in underweight.iterrows():
                st.error(f"{row.name}　RS = {row['RS']:.2f}", icon=None)
    else:
        st.info("点击加载行业信号，首次约需 1~2 分钟")

st.divider()

# ── 第三行：宏观快照（单行，不占竖向空间） ───────────────────────────────────
latest_macro = macro_df.dropna(how="all").iloc[-1]
n1, n2, n3, n4 = st.columns(4)
for col_widget, (col_key, label) in zip([n1, n2, n3, n4], _MACRO_LABEL.items()):
    val = latest_macro.get(col_key)
    if pd.notna(val):
        unit = "%" if col_key != "pmi" else ""
        col_widget.metric(label, f"{val:.1f}{unit}")  # type: ignore
