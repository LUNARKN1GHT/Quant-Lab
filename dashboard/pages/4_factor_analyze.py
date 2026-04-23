import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.factor.ic import calc_icir
from quant.factor.ma_bias import ma_bias
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.factor.volatility import volatility

st.set_page_config(page_title="因子分析", layout="wide")

cfg = sidebar_config()

st.title("🔬 因子分析")

FACTOR_MAP = {
    "动量(20日)": lambda c: c.apply(lambda s: momentum(s, 20)),
    "动量(60日)": lambda c: c.apply(lambda s: momentum(s, 60)),
    "RSI(14日)": lambda c: c.apply(lambda s: rsi(s, 14)),
    "波动率(20日)": lambda c: c.apply(lambda s: volatility(s, 20)),
    "均线偏离(20日)": lambda c: c.apply(lambda s: ma_bias(s, 20)),
}

factor_name = st.selectbox("选择因子", list(FACTOR_MAP.keys()))
fwd_window = st.slider(
    "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5
)


@st.cache_data(show_spinner="计算因子 IC…")
def compute_ic(factor_name: str, fwd_window: int):
    close = load_close()
    factor_fn = FACTOR_MAP[factor_name]
    factor_vals = factor_fn(close)
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

    # 月度 IC
    ic_list = []
    for date, row in factor_vals.resample("ME").last().iterrows():
        actual_date = fwd_ret.index.asof(date)
        if actual_date is None or str(actual_date) == "NaT":
            continue
        f = row.dropna()
        r = fwd_ret.loc[actual_date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        ic = f[common].corr(r[common])
        ic_list.append({"date": date, "ic": ic})

    import pandas as pd

    ic_series = pd.DataFrame(ic_list).set_index("date")["ic"].dropna()
    return ic_series, factor_vals, fwd_ret


ic_series, factor_vals, fwd_ret = compute_ic(factor_name, fwd_window)

# IC 统计摘要
icir = calc_icir(ic_series)
col1, col2, col3, col4 = st.columns(4)
col1.metric("IC 均值", f"{ic_series.mean():.4f}")
col2.metric("IC 标准差", f"{ic_series.std():.4f}")
col3.metric("ICIR", f"{icir:.3f}")
col4.metric("IC > 0 占比", f"{(ic_series > 0).mean():.1%}")

# IC 月度柱状图
st.subheader("月度 IC 序列")
colors = ["#4CAF50" if v > 0 else "#F44336" for v in ic_series.values]
fig_ic = go.Figure()
fig_ic.add_trace(
    go.Bar(x=ic_series.index, y=ic_series.values, marker_color=colors, name="月度IC")
)
fig_ic.add_hline(
    y=ic_series.mean(),
    line_dash="dash",
    line_color="#2196F3",
    annotation_text=f"均值 {ic_series.mean():.4f}",
)
fig_ic.add_hline(y=0, line_color="gray", line_width=0.5)
fig_ic.update_layout(xaxis_title="日期", yaxis_title="IC", hovermode="x unified")
st.plotly_chart(fig_ic, use_container_width=True)

# 因子分层回测
st.subheader(f"因子五分位分层收益（前向 {fwd_window} 日）")


@st.cache_data(show_spinner="计算分层收益…")
def compute_quantile_returns(factor_name: str, fwd_window: int, n_groups: int = 5):
    import pandas as pd

    close = load_close()
    factor_fn = FACTOR_MAP[factor_name]
    factor_vals = factor_fn(close)
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

    group_rets = {f"Q{i + 1}": [] for i in range(n_groups)}
    for date, row in factor_vals.resample("ME").last().iterrows():
        actual_date = fwd_ret.index.asof(date)
        if actual_date is None or str(actual_date) == "NaT":
            continue
        f = row.dropna()
        r = fwd_ret.loc[actual_date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups * 2:
            continue
        labels = pd.qcut(
            f[common], n_groups, labels=[f"Q{i + 1}" for i in range(n_groups)]
        )
        for g in group_rets:
            stocks = labels[labels == g].index
            group_rets[g].append(r[stocks].mean())

    return {g: np.mean(v) for g, v in group_rets.items() if v}


q_rets = compute_quantile_returns(factor_name, fwd_window)
fig_q = go.Figure()
fig_q.add_trace(
    go.Bar(
        x=list(q_rets.keys()),
        y=list(q_rets.values()),
        marker_color=["#F44336", "#FF9800", "#9E9E9E", "#8BC34A", "#4CAF50"],
        text=[f"{v:.2%}" for v in q_rets.values()],
        textposition="outside",
    )
)
fig_q.update_layout(
    yaxis_tickformat=".2%", xaxis_title="分位组", yaxis_title="平均前向收益"
)
st.plotly_chart(fig_q, use_container_width=True)
