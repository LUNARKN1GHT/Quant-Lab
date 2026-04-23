import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.factor.bollinger import bollinger_position
from quant.factor.ic import calc_icir
from quant.factor.idiosyncratic_vol import idiosyncratic_vol
from quant.factor.ma_bias import ma_bias
from quant.factor.macd import macd
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.factor.skewness import kurtosis, skewness
from quant.factor.volatility import volatility

st.set_page_config(page_title="因子分析", layout="wide")

cfg = sidebar_config()

st.title("🔬 因子分析")

FACTOR_NAMES = [
    "动量(20日)",
    "动量(60日)",
    "动量(120日)",
    "RSI(14日)",
    "波动率(20日)",
    "均线偏离(20日)",
    "MACD",
    "布林带位置(20日)",
    "偏度(20日)",
    "峰度(20日)",
    "特质波动率(20日)",
]


def compute_single_factor(name: str, close: pd.DataFrame) -> pd.DataFrame:
    market = close.mean(axis=1)
    dispatch = {
        "动量(20日)": lambda: close.apply(lambda s: momentum(s, 20)),
        "动量(60日)": lambda: close.apply(lambda s: momentum(s, 60)),
        "动量(120日)": lambda: close.apply(lambda s: momentum(s, 120)),
        "RSI(14日)": lambda: close.apply(lambda s: rsi(s, 14)),
        "波动率(20日)": lambda: close.apply(lambda s: volatility(s, 20)),
        "均线偏离(20日)": lambda: close.apply(lambda s: ma_bias(s, 20)),
        "MACD": lambda: close.apply(lambda s: macd(s)),
        "布林带位置(20日)": lambda: close.apply(lambda s: bollinger_position(s, 20)),
        "偏度(20日)": lambda: close.apply(lambda s: skewness(s, 20)),
        "峰度(20日)": lambda: close.apply(lambda s: kurtosis(s, 20)),
        "特质波动率(20日)": lambda: close.apply(
            lambda s: idiosyncratic_vol(s, market, 20)
        ),
    }
    return dispatch[name]()


tab1, tab2 = st.tabs(["单因子分析", "全因子筛选"])

# ── Tab 1：单因子详细分析 ──────────────────────────────────────────────────
with tab1:
    factor_name = st.selectbox("选择因子", FACTOR_NAMES)
    fwd_window = st.slider(
        "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5
    )

    @st.cache_data(show_spinner="计算因子 IC…")
    def compute_ic(factor_name: str, fwd_window: int) -> pd.Series:
        close = load_close()
        factor_vals = compute_single_factor(factor_name, close)
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

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
            ic_list.append({"date": date, "ic": f[common].corr(r[common])})

        return pd.DataFrame(ic_list).set_index("date")["ic"].dropna()

    @st.cache_data(show_spinner="计算分层收益…")
    def compute_quantile_returns(
        factor_name: str, fwd_window: int, n_groups: int = 5
    ) -> dict[str, float]:
        close = load_close()
        factor_vals = compute_single_factor(factor_name, close)
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

        group_rets: dict[str, list] = {f"Q{i + 1}": [] for i in range(n_groups)}
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

        return {g: float(np.mean(v)) for g, v in group_rets.items() if v}

    ic_series = compute_ic(factor_name, fwd_window)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("IC 均值", f"{ic_series.mean():.4f}")
    col2.metric("IC 标准差", f"{ic_series.std():.4f}")
    col3.metric("ICIR", f"{calc_icir(ic_series):.3f}")
    col4.metric("IC > 0 占比", f"{(ic_series > 0).mean():.1%}")

    st.subheader("月度 IC 序列")
    colors = ["#4CAF50" if v > 0 else "#F44336" for v in ic_series.values]
    fig_ic = go.Figure()
    fig_ic.add_trace(
        go.Bar(
            x=ic_series.index, y=ic_series.values, marker_color=colors, name="月度IC"
        )
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

    st.subheader(f"因子五分位分层收益（前向 {fwd_window} 日）")
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


# ── Tab 2：全因子筛选 ─────────────────────────────────────────────────────
with tab2:
    st.caption("对所有已实现因子批量计算 IC/ICIR，自动筛选有效因子并展示相关性矩阵")

    fwd_window_all = st.slider(
        "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5, key="fwd_all"
    )
    run_all = st.button("🔍 运行全因子筛选", type="primary")

    @st.cache_data(show_spinner="批量计算全因子 IC/ICIR…")
    def compute_all_factors(fwd_window: int):
        close = load_close()
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

        summary = []
        factor_last: dict[str, pd.Series] = {}

        for name in FACTOR_NAMES:
            factor_vals = compute_single_factor(name, close)
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
                ic_list.append(f[common].corr(r[common]))
            ic_s = pd.Series(ic_list).dropna()
            if ic_s.empty:
                continue
            summary.append(
                {
                    "因子": name,
                    "IC均值": ic_s.mean(),
                    "IC标准差": ic_s.std(),
                    "ICIR": calc_icir(ic_s),
                    "IC>0占比": (ic_s > 0).mean(),
                    "样本月数": len(ic_s),
                }
            )
            factor_last[name] = factor_vals.iloc[-1].dropna()

        summary_df = (
            pd.DataFrame(summary).set_index("因子").sort_values("ICIR", ascending=False)
        )
        corr_df = pd.DataFrame(factor_last).corr()
        return summary_df, corr_df

    if run_all:
        summary_df, corr_df = compute_all_factors(fwd_window_all)

        st.subheader("因子 IC/ICIR 汇总")
        icir_threshold = 0.3
        styled = summary_df.style.format(
            {
                "IC均值": "{:.4f}",
                "IC标准差": "{:.4f}",
                "ICIR": "{:.3f}",
                "IC>0占比": "{:.1%}",
                "样本月数": "{:.0f}",
            }
        ).background_gradient(subset=["ICIR"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        st.dataframe(styled, use_container_width=True)

        st.subheader("ICIR 排名")
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=summary_df.index,
                y=summary_df["ICIR"],
                marker_color=[
                    "#4CAF50"
                    if v > icir_threshold
                    else "#F44336"
                    if v < -icir_threshold
                    else "#FF9800"
                    for v in summary_df["ICIR"]
                ],
                text=[f"{v:.3f}" for v in summary_df["ICIR"]],
                textposition="outside",
            )
        )
        fig_bar.add_hline(
            y=icir_threshold,
            line_dash="dash",
            line_color="#4CAF50",
            annotation_text=f"有效阈值 {icir_threshold}",
        )
        fig_bar.add_hline(y=-icir_threshold, line_dash="dash", line_color="#F44336")
        fig_bar.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_bar.update_layout(
            xaxis_title="因子", yaxis_title="ICIR", xaxis_tickangle=-30
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("因子相关性矩阵（最新截面）")
        st.caption("相关性过高（>0.7）的因子存在冗余，可考虑合并或剔除")
        fig_heat = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
        )
        fig_heat.update_layout(coloraxis_colorbar_title="相关系数")
        st.plotly_chart(fig_heat, use_container_width=True)

        effective = summary_df[summary_df["ICIR"].abs() > icir_threshold]
        if not effective.empty:
            st.success(
                f"**有效因子（|ICIR| > {icir_threshold}）：** "
                + "、".join(effective.index.tolist())
            )
        else:
            st.warning("当前前向窗口下无因子 ICIR 超过阈值，可尝试调整窗口")
    else:
        st.info("点击「运行全因子筛选」开始批量计算，首次约需 1~2 分钟。")
