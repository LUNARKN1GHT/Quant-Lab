import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.advisor.position import compute_position

st.set_page_config(page_title="Quant-Lab", page_icon="📈", layout="wide")

cfg = sidebar_config()

st.title("📈 Quant-Lab Dashboard")
st.caption("基于沪深300成分股数据")

close = load_close()
result = compute_position(close=close, cfg=cfg)
latest = result.iloc[-1]
latest_date = result.index[-1]

# 今日摘要
st.subheader(f"今日概况 ({latest_date.date()})")
col1, col2, col3, col4 = st.columns(4)

regime_color = {"BULL": "🟢", "RANGE": "🟡", "BEAR": "🔴"}
col1.metric(
    "市场状态", f"{regime_color.get(latest['regime'], '')}" + f"{latest['regime']}"
)
col2.metric("建议仓位", f"{latest['position']:.0%}")
col3.metric("Regime 分量", f"{latest['regime_scale']:.0%}")
col4.metric("波动率分量", f"{latest['vol_scale']:.0%}")

st.info(
    f"数据覆盖：{close.index[0].date()} ~ {close.index[-1].date()}"
    + f"股票数量：{close.shape[1]}"
)
