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

# 有单一窗口参数的因子
WINDOWED_FACTORS = [
    "动量",
    "RSI",
    "波动率",
    "均线偏离",
    "布林带位置",
    "偏度",
    "峰度",
    "特质波动率",
]
# 有独立参数的因子
OTHER_FACTORS = ["MACD"]
ALL_FACTOR_TYPES = WINDOWED_FACTORS + OTHER_FACTORS

# 全因子筛选时各因子的默认窗口
DEFAULT_WINDOWS: dict[str, int] = {
    "动量": 20,
    "RSI": 14,
    "波动率": 20,
    "均线偏离": 20,
    "布林带位置": 20,
    "偏度": 20,
    "峰度": 20,
    "特质波动率": 20,
}


def compute_factor(
    name: str,
    close: pd.DataFrame,
    window: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    market = close.mean(axis=1)
    if name == "动量":
        return close.apply(lambda s: momentum(s, window))
    if name == "RSI":
        return close.apply(lambda s: rsi(s, window))
    if name == "波动率":
        return close.apply(lambda s: volatility(s, window))
    if name == "均线偏离":
        return close.apply(lambda s: ma_bias(s, window))
    if name == "布林带位置":
        return close.apply(lambda s: bollinger_position(s, window))
    if name == "偏度":
        return close.apply(lambda s: skewness(s, window))
    if name == "峰度":
        return close.apply(lambda s: kurtosis(s, window))
    if name == "特质波动率":
        return close.apply(lambda s: idiosyncratic_vol(s, market, window))
    if name == "MACD":
        return close.apply(lambda s: macd(s, macd_fast, macd_slow, macd_signal))
    raise ValueError(f"未知因子: {name}")


def _calc_monthly_ic(factor_vals: pd.DataFrame, fwd_ret: pd.DataFrame) -> pd.Series:
    ic_list = []
    for date, row in factor_vals.resample("ME").last().iterrows():
        actual_date = fwd_ret.index.asof(date)  # type: ignore[arg-type]
        if actual_date is None or str(actual_date) == "NaT":
            continue
        f = row.dropna()
        r = fwd_ret.loc[actual_date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        ic_list.append({"date": date, "ic": f[common].corr(r[common])})  # type: ignore[arg-type]
    if not ic_list:
        return pd.Series(dtype=float)
    return pd.DataFrame(ic_list).set_index("date")["ic"].dropna()


tab1, tab2 = st.tabs(["单因子分析", "全因子筛选"])

# ── Tab 1：单因子详细分析 ──────────────────────────────────────────────────
with tab1:
    col_sel, col_fwd = st.columns([2, 1])
    with col_sel:
        factor_type = st.selectbox("因子类型", ALL_FACTOR_TYPES)
    with col_fwd:
        fwd_window = st.slider(
            "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5
        )

    # 根据因子类型动态显示参数控件
    if factor_type in WINDOWED_FACTORS:
        window = st.slider(
            "因子窗口（天）", 5, 120, DEFAULT_WINDOWS.get(factor_type, 20), step=5
        )
        macd_fast, macd_slow, macd_signal = 12, 26, 9
    else:  # MACD
        c1, c2, c3 = st.columns(3)
        macd_fast = c1.slider("Fast", 3, 30, 12, step=1)
        macd_slow = c2.slider("Slow", 10, 60, 26, step=1)
        macd_signal = c3.slider("Signal", 3, 20, 9, step=1)
        window = macd_fast

    @st.cache_data(show_spinner="计算因子 IC…")
    def compute_ic(
        factor_type: str,
        window: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        fwd_window: int,
    ) -> pd.Series:
        close = load_close()
        factor_vals = compute_factor(
            factor_type, close, window, macd_fast, macd_slow, macd_signal
        )
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)
        return _calc_monthly_ic(factor_vals, fwd_ret)

    @st.cache_data(show_spinner="计算分层收益…")
    def compute_quantile_returns(
        factor_type: str,
        window: int,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        fwd_window: int,
        n_groups: int = 5,
    ) -> dict[str, float]:
        close = load_close()
        factor_vals = compute_factor(
            factor_type, close, window, macd_fast, macd_slow, macd_signal
        )
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

        group_rets: dict[str, list[float]] = {f"Q{i + 1}": [] for i in range(n_groups)}
        for date, row in factor_vals.resample("ME").last().iterrows():
            actual_date = fwd_ret.index.asof(date)  # type: ignore[arg-type]
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
                group_rets[g].append(float(r[stocks].mean()))  # type: ignore[arg-type]

        return {g: float(np.mean(v)) for g, v in group_rets.items() if v}

    ic_series = compute_ic(
        factor_type, window, macd_fast, macd_slow, macd_signal, fwd_window
    )

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
    q_rets = compute_quantile_returns(
        factor_type, window, macd_fast, macd_slow, macd_signal, fwd_window
    )
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
    st.caption("各因子使用默认窗口批量计算 IC/ICIR，自动筛选有效因子并展示相关性矩阵")

    fwd_window_all = st.slider(
        "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5, key="fwd_all"
    )
    run_all = st.button("🔍 运行全因子筛选", type="primary")

    @st.cache_data(show_spinner="批量计算全因子 IC/ICIR…")
    def compute_all_factors(fwd_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        close = load_close()
        fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

        summary = []
        factor_last: dict[str, pd.Series] = {}

        for name in ALL_FACTOR_TYPES:
            w = DEFAULT_WINDOWS.get(name, 20)
            factor_vals = compute_factor(name, close, window=w)
            ic_s = _calc_monthly_ic(factor_vals, fwd_ret)
            if ic_s.empty:
                continue
            label = name if name == "MACD" else f"{name}({w}日)"
            summary.append(
                {
                    "因子": label,
                    "IC均值": ic_s.mean(),
                    "IC标准差": ic_s.std(),
                    "ICIR": calc_icir(ic_s),
                    "IC>0占比": (ic_s > 0).mean(),
                    "样本月数": len(ic_s),
                }
            )
            factor_last[label] = factor_vals.iloc[-1].dropna()

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
        ).bar(subset=["ICIR"], align="mid", color=["#F44336", "#4CAF50"])
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
            text_auto=".2f",  # type: ignore
            aspect="auto",  # type: ignore[arg-type]
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
