import sys
from pathlib import Path
from typing import cast

from quant.config import SignalWeightsConfig

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


@st.cache_data(show_spinner="加载行业数据...")
def _load_sector_signals() -> pd.DataFrame:
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)
    rs = calc_rs(sector_close=sector_close, benchmark=benchmark, window=20)
    rs_momentum = calc_rs_momentum(rs=rs, lookback=20)
    return get_suggestions(rs.iloc[-1], rs_momentum, top_n=3)


# --- 权重调节 ------------
st.title("⚖️ 仓位建议")
with st.expander("🎛️ 信号权重调整（拖动滑块实时生效）", expanded=False):
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    w_regime = col_w1.slider("Regime 权重", 0.0, 1.0, cfg.signal_weights.regime, 0.05)
    w_vol = col_w2.slider("波动率权重", 0.0, 1.0, cfg.signal_weights.vol, 0.05)
    w_macro = col_w3.slider("宏观权重", 0.0, 1.0, cfg.signal_weights.macro, 0.05)
    w_sector = col_w4.slider("行业权重", 0.0, 1.0, cfg.signal_weights.sector, 0.05)
    total = w_regime + w_vol + w_macro + w_sector
    if total == 0:
        st.error("权重之和不能为 0")
        st.stop()
    st.caption(f"权重之和: {total:.2f} (自动归一化，无需手动调整)")

weights = SignalWeightsConfig(
    regime=w_regime, vol=w_vol, macro=w_macro, sector=w_sector
)

# --- 数据加载 ------------
macro_score = get_macro_score()
result = compute_position(close, cfg, macro_score=macro_score, signal_weights=weights)

# --- 最新建议卡片 ----------
latest = result.iloc[-1]
st.subheader(f"最新建议 ({result.index[-1].date()})")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Regime 信号", f"{latest['regime_signal']:.0%}")
col2.metric("波动率信号", f"{latest['vol_signal']:.0%}")
col3.metric("宏观信号", f"{latest['macro_signal']:.2f}x")
col4.metric("行业信号", f"{latest['sector_signal']: .0%}")
col5.metric(
    "最终建议仓位",
    f"{latest['position']:.0%}",
    delta=f"{latest['position'] - result['position'].iloc[-2]:.0%}",
)


# --- 历史仓位走势 ----------
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
        y=result["regime_signal"],
        name="Regime信号",
        line=dict(dash="dot", color="#FF9800"),
    )
)
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["vol_signal"],
        name="波动率信号",
        line=dict(dash="dot", color="#9C27B0"),
    )
)
fig.add_trace(
    go.Scatter(
        x=result.index,
        y=result["macro_signal"],
        name="宏观信号",
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

# --- Regime 切换记录 ----------
st.subheader("Regime 切换记录（近60日）")
recent = result.tail(60)
changes = recent[recent["regime"] != recent["regime"].shift()][["regime", "position"]]
st.dataframe(changes.style.format({"position": "{:.0%}"}), width="stretch")

# --- 行业超配建议 -----------
st.subheader("当前行业超配建议")
st.caption("基于申万行业 RS + 动量，首次加载约需 1~2 分钟")

if st.button("📥 加载行业信号", key="sector_btn"):
    st.session_state["sector_suggestions"] = _load_sector_signals()

if "sector_suggestions" in st.session_state:
    sug = cast(pd.DataFrame, st.session_state["sector_suggestions"])
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
