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

st.set_page_config(page_title="参数调优", layout="wide")
sidebar_config()

st.title("🎛️ 参数调优")
st.caption("网格搜索因子参数，寻找稳健区间（而非最优点）")

FACTOR_BUILDERS = {
    "动量": lambda c, w: c.apply(lambda s: momentum(s, w)),
    "RSI": lambda c, w: c.apply(lambda s: rsi(s, w)),
    "波动率": lambda c, w: c.apply(lambda s: volatility(s, w)),
    "均线偏离": lambda c, w: c.apply(lambda s: ma_bias(s, w)),
}

PARAM_RANGES = {
    "动量": [5, 10, 20, 40, 60, 120],
    "RSI": [7, 10, 14, 21, 28],
    "波动率": [5, 10, 20, 40, 60],
    "均线偏离": [5, 10, 20, 40, 60],
}

col1, col2, col3 = st.columns(3)
factor_name = col1.selectbox("因子类型", list(FACTOR_BUILDERS.keys()))
fwd_window = col2.slider("前向收益窗口（天）", 5, 60, 20, step=5)
is_ratio = col3.slider("样本内比例（IS）", 0.4, 0.8, 0.6, step=0.05)

ALL_OPTIONS = [5, 7, 10, 14, 20, 21, 28, 40, 60, 120, 180]
windows = st.multiselect(
    "搜索参数窗口（天）",
    options=ALL_OPTIONS,
    default=PARAM_RANGES[factor_name],
)

run_btn = st.button("🔍 运行网格搜索", type="primary")


@st.cache_data(show_spinner="网格搜索中…")
def grid_search(factor_name: str, windows: tuple, fwd_window: int, is_ratio: float):
    import pandas as pd

    close = load_close()
    builder = FACTOR_BUILDERS[factor_name]
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

    n = len(close)
    split = int(n * is_ratio)
    close_is = close.iloc[:split]
    close_oos = close.iloc[split:]
    fwd_is = fwd_ret.iloc[:split]
    fwd_oos = fwd_ret.iloc[split:]

    def calc_monthly_ic(factor_vals, fwd):
        ic_list = []
        for date, row in factor_vals.resample("ME").last().iterrows():
            actual = fwd.index.asof(date)
            if actual is None or str(actual) == "NaT":
                continue
            f = row.dropna()
            r = fwd.loc[actual].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 10:
                continue
            ic_list.append(f[common].corr(r[common]))
        return pd.Series(ic_list).dropna()

    results = []
    for w in windows:
        fv_is = builder(close_is, w)
        fv_oos = builder(close_oos, w)
        ic_is = calc_monthly_ic(fv_is, fwd_is)
        ic_oos = calc_monthly_ic(fv_oos, fwd_oos)
        results.append(
            {
                "window": w,
                "IS_ICIR": calc_icir(ic_is) if len(ic_is) > 3 else np.nan,
                "OOS_ICIR": calc_icir(ic_oos) if len(ic_oos) > 3 else np.nan,
                "IS_IC均值": ic_is.mean() if len(ic_is) > 0 else np.nan,
                "OOS_IC均值": ic_oos.mean() if len(ic_oos) > 0 else np.nan,
            }
        )
    return pd.DataFrame(results).set_index("window")


if run_btn and windows:
    df = grid_search(factor_name, tuple(sorted(windows)), fwd_window, is_ratio)

    # ICIR 对比柱状图
    st.subheader("IS vs OOS ICIR 对比")
    st.caption("稳健参数特征：IS/OOS 差距小，且 OOS ICIR 仍为正")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["IS_ICIR"],
            name="样本内 ICIR",
            marker_color="#2196F3",
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["OOS_ICIR"],
            name="样本外 ICIR",
            marker_color="#FF9800",
            opacity=0.8,
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        barmode="group",
        xaxis_title="窗口（天）",
        yaxis_title="ICIR",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # IC 均值对比
    st.subheader("IS vs OOS IC 均值")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Bar(
            x=df.index,
            y=df["IS_IC均值"],
            name="样本内 IC",
            marker_color="#4CAF50",
            opacity=0.8,
        )
    )
    fig2.add_trace(
        go.Bar(
            x=df.index,
            y=df["OOS_IC均值"],
            name="样本外 IC",
            marker_color="#F44336",
            opacity=0.8,
        )
    )
    fig2.add_hline(y=0, line_color="gray", line_width=1)
    fig2.update_layout(
        barmode="group",
        xaxis_title="窗口（天）",
        yaxis_title="IC 均值",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 结果表格
    st.subheader("详细结果")
    styled = df.style.format("{:.4f}", na_rep="—")
    st.dataframe(styled, use_container_width=True)

    # 推荐参数
    best = df["OOS_ICIR"].idxmax()
    oos_val = float(df.loc[best, "OOS_ICIR"])
    is_val = float(df.loc[best, "IS_ICIR"])
    gap = abs(is_val - oos_val)
    st.info(
        f"**推荐参数：{best} 天**  |  "
        f"OOS ICIR = {oos_val:.3f}  |  "
        f"IS/OOS 差距 = {gap:.3f}"
        + ("（⚠️ IS/OOS 差距较大，可能过拟合）" if gap > 0.3 else "（✅ 较为稳健）")
    )
elif not windows:
    st.warning("请至少选择一个参数窗口")
else:
    st.info("选择因子和参数范围后，点击「运行网格搜索」。首次约需 30 秒。")
