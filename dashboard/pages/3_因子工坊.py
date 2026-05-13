import sys
from pathlib import Path
from typing import cast

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

st.set_page_config(page_title="因子工坊", layout="wide")

cfg = sidebar_config()
st.title("🔬 因子工坊")

# ── 常量 ────────────────────────────────────────────────────────────────────
_WINDOWED_FACTORS = [
    "动量",
    "RSI",
    "波动率",
    "均线偏离",
    "布林带位置",
    "偏度",
    "峰度",
    "特质波动率",
]
_ALL_FACTOR_TYPES = _WINDOWED_FACTORS + ["MACD"]
_DEFAULT_WINDOWS: dict[str, int] = {
    "动量": 20,
    "RSI": 14,
    "波动率": 20,
    "均线偏离": 20,
    "布林带位置": 20,
    "偏度": 20,
    "峰度": 20,
    "特质波动率": 20,
}

_PARAM_BUILDERS: dict[str, object] = {
    "动量": lambda c, w: c.apply(lambda s: momentum(s, w)),
    "RSI": lambda c, w: c.apply(lambda s: rsi(s, w)),
    "波动率": lambda c, w: c.apply(lambda s: volatility(s, w)),
    "均线偏离": lambda c, w: c.apply(lambda s: ma_bias(s, w)),
    "布林带位置": lambda c, w: c.apply(lambda s: bollinger_position(s, w)),
    "MACD(fast)": lambda c, w: c.apply(lambda s: macd(s, fast=w)),
}
_PARAM_RANGES: dict[str, list[int]] = {
    "动量": [5, 10, 20, 40, 60, 120],
    "RSI": [7, 10, 14, 21, 28],
    "波动率": [5, 10, 20, 40, 60],
    "均线偏离": [5, 10, 20, 40, 60],
    "布林带位置": [10, 20, 30, 40, 60],
    "MACD(fast)": [5, 8, 12, 16, 20],
}
_ALL_WINDOWS = [5, 7, 10, 14, 20, 21, 28, 40, 60, 120, 180]

_IMG_DIR = Path(__file__).parent.parent.parent / "output" / "factor_research"
_REGIME_COLORS = {"BULL": "#2ca02c", "RANGE": "#ff7f0e", "BEAR": "#d62728"}


# ── 共用函数 ─────────────────────────────────────────────────────────────────
def _compute_factor(
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
    ic_list: list[dict] = []
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


@st.cache_data(show_spinner="计算因子 IC…")
def _compute_ic(
    factor_type: str,
    window: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    fwd_window: int,
) -> pd.Series:
    close = load_close()
    factor_vals = _compute_factor(
        factor_type, close, window, macd_fast, macd_slow, macd_signal
    )
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)
    return _calc_monthly_ic(factor_vals, fwd_ret)


@st.cache_data(show_spinner="计算分层收益…")
def _compute_quantile_returns(
    factor_type: str,
    window: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    fwd_window: int,
    n_groups: int = 5,
) -> dict[str, float]:
    close = load_close()
    factor_vals = _compute_factor(
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


@st.cache_data(show_spinner="批量计算全因子 IC/ICIR…")
def _compute_all_factors(fwd_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = load_close()
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

    summary: list[dict] = []
    factor_last: dict[str, pd.Series] = {}

    for name in _ALL_FACTOR_TYPES:
        w = _DEFAULT_WINDOWS.get(name, 20)
        factor_vals = _compute_factor(name, close, window=w)
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


@st.cache_data(show_spinner="网格搜索中…")
def _grid_search(
    factor_name: str, windows: tuple[int, ...], fwd_window: int, is_ratio: float
) -> pd.DataFrame:
    close = load_close()
    builder = _PARAM_BUILDERS[factor_name]
    fwd_ret = close.pct_change(fwd_window).shift(-fwd_window)

    n = len(close)
    split = int(n * is_ratio)
    close_is, close_oos = close.iloc[:split], close.iloc[split:]
    fwd_is, fwd_oos = fwd_ret.iloc[:split], fwd_ret.iloc[split:]

    def _monthly_ic_simple(fv: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
        ic_list: list[float] = []
        for date, row in fv.resample("ME").last().iterrows():
            actual = fwd.index.asof(date)  # type: ignore[arg-type]
            if actual is None or str(actual) == "NaT":
                continue
            f = row.dropna()
            r = fwd.loc[actual].reindex(f.index).dropna()
            common = f.index.intersection(r.index)
            if len(common) < 10:
                continue
            ic_list.append(f[common].corr(r[common]))  # type: ignore[arg-type]
        return pd.Series(ic_list).dropna()

    results: list[dict] = []
    for w in windows:
        fv_is = builder(close_is, w)  # type: ignore[operator]
        fv_oos = builder(close_oos, w)  # type: ignore[operator]
        ic_is = _monthly_ic_simple(fv_is, fwd_is)
        ic_oos = _monthly_ic_simple(fv_oos, fwd_oos)
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


# ── Tabs ────────────────────────────────────────────────────────────────────
tab_single, tab_all, tab_param, tab_research = st.tabs(
    ["📊 单因子分析", "📋 全因子筛选", "🎛️ 参数调优", "🔬 自主研究"]
)

# ──────────────────────────────────────────────────────────────────────────
# Tab 1：单因子分析
# ──────────────────────────────────────────────────────────────────────────
with tab_single:
    col_sel, col_fwd = st.columns([2, 1])
    with col_sel:
        factor_type = st.selectbox("因子类型", _ALL_FACTOR_TYPES)
    with col_fwd:
        fwd_window = st.slider(
            "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5
        )

    if factor_type in _WINDOWED_FACTORS:
        window = st.slider(
            "因子窗口（天）", 5, 120, _DEFAULT_WINDOWS.get(factor_type, 20), step=5
        )
        macd_fast, macd_slow, macd_signal = 12, 26, 9
    else:
        c1, c2, c3 = st.columns(3)
        macd_fast = c1.slider("Fast", 3, 30, 12, step=1)
        macd_slow = c2.slider("Slow", 10, 60, 26, step=1)
        macd_signal = c3.slider("Signal", 3, 20, 9, step=1)
        window = macd_fast

    ic_series = _compute_ic(
        factor_type, window, macd_fast, macd_slow, macd_signal, fwd_window
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IC 均值", f"{ic_series.mean():.4f}")
    m2.metric("IC 标准差", f"{ic_series.std():.4f}")
    m3.metric("ICIR", f"{calc_icir(ic_series):.3f}")
    m4.metric("IC > 0 占比", f"{(ic_series > 0).mean():.1%}")

    st.subheader("月度 IC 序列")
    bar_colors = ["#4CAF50" if v > 0 else "#F44336" for v in ic_series.values]
    fig_ic = go.Figure()
    fig_ic.add_trace(
        go.Bar(
            x=ic_series.index,
            y=ic_series.values,
            marker_color=bar_colors,
            name="月度IC",
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
    q_rets = _compute_quantile_returns(
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


# ──────────────────────────────────────────────────────────────────────────
# Tab 2：全因子筛选
# ──────────────────────────────────────────────────────────────────────────
with tab_all:
    st.caption("各因子使用默认窗口批量计算 IC/ICIR，自动筛选有效因子并展示相关性矩阵")
    fwd_window_all = st.slider(
        "前向收益窗口（天）", 5, 60, cfg.factor.ic_forward_window, step=5, key="fwd_all"
    )

    if st.button("🔍 运行全因子筛选", type="primary"):
        summary_df, corr_df = _compute_all_factors(fwd_window_all)
        st.session_state["factor_summary"] = summary_df
        st.session_state["factor_corr"] = corr_df

    if "factor_summary" in st.session_state:
        from typing import cast

        summary_df = cast(pd.DataFrame, st.session_state["factor_summary"])
        corr_df = cast(pd.DataFrame, st.session_state["factor_corr"])
        icir_threshold = 0.3

        st.subheader("因子 IC/ICIR 汇总")
        styled_summary = summary_df.style.format(
            {
                "IC均值": "{:.4f}",
                "IC标准差": "{:.4f}",
                "ICIR": "{:.3f}",
                "IC>0占比": "{:.1%}",
                "样本月数": "{:.0f}",
            }
        ).bar(subset=["ICIR"], align="mid", color=["#F44336", "#4CAF50"])
        st.dataframe(styled_summary, use_container_width=True)

        st.subheader("ICIR 排名")
        fig_icir = go.Figure()
        fig_icir.add_trace(
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
        fig_icir.add_hline(
            y=icir_threshold,
            line_dash="dash",
            line_color="#4CAF50",
            annotation_text=f"有效阈值 {icir_threshold}",
        )
        fig_icir.add_hline(y=-icir_threshold, line_dash="dash", line_color="#F44336")
        fig_icir.add_hline(y=0, line_color="gray", line_width=0.5)
        fig_icir.update_layout(
            xaxis_title="因子", yaxis_title="ICIR", xaxis_tickangle=-30
        )
        st.plotly_chart(fig_icir, use_container_width=True)

        st.subheader("因子相关性矩阵（最新截面）")
        st.caption("相关性过高（>0.7）的因子存在冗余，可考虑合并或剔除")
        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",  # type: ignore[arg-type]
            aspect="auto",
        )
        fig_corr.update_layout(coloraxis_colorbar_title="相关系数")
        st.plotly_chart(fig_corr, use_container_width=True)

        effective = summary_df[summary_df["ICIR"].abs() > icir_threshold]
        if not effective.empty:
            st.success(
                f"**有效因子（|ICIR| > {icir_threshold}）：**"
                + "、".join(effective.index.tolist())
            )
        else:
            st.warning("当前前向窗口下无因子 ICIR 超过阈值，可尝试调整窗口")
    else:
        st.info("点击「运行全因子筛选」开始批量计算，首次约需 1~2 分钟。")


# ──────────────────────────────────────────────────────────────────────────
# Tab 3：参数调优
# ──────────────────────────────────────────────────────────────────────────
with tab_param:
    st.caption("网格搜索因子参数，寻找稳健区间（而非最优点）")

    p1, p2, p3 = st.columns(3)
    param_factor = p1.selectbox(
        "因子类型", list(_PARAM_BUILDERS.keys()), key="param_factor"
    )
    param_fwd = p2.slider("前向收益窗口（天）", 5, 60, 20, step=5, key="param_fwd")
    param_is = p3.slider("样本内比例（IS）", 0.4, 0.8, 0.6, step=0.05, key="param_is")

    param_windows = st.multiselect(
        "搜索参数窗口（天）",
        options=_ALL_WINDOWS,
        default=_PARAM_RANGES[param_factor],
        key="param_windows",
    )

    if st.button("🔍 运行网格搜索", type="primary") and param_windows:
        gs_df = _grid_search(
            param_factor, tuple(sorted(param_windows)), param_fwd, param_is
        )
        st.session_state["gs_result"] = gs_df
    elif not param_windows:
        st.warning("请至少选择一个参数窗口")

    if "gs_result" in st.session_state:
        from typing import cast as _cast

        gs_df = _cast(pd.DataFrame, st.session_state["gs_result"])

        st.subheader("IS vs OOS ICIR 对比")
        st.caption("稳健参数特征：IS/OOS 差距小，且 OOS ICIR 仍为正")
        fig_gs1 = go.Figure()
        fig_gs1.add_trace(
            go.Bar(
                x=gs_df.index,
                y=gs_df["IS_ICIR"],
                name="样本内 ICIR",
                marker_color="#2196F3",
                opacity=0.8,
            )
        )
        fig_gs1.add_trace(
            go.Bar(
                x=gs_df.index,
                y=gs_df["OOS_ICIR"],
                name="样本外 ICIR",
                marker_color="#FF9800",
                opacity=0.8,
            )
        )
        fig_gs1.add_hline(y=0, line_color="gray", line_width=1)
        fig_gs1.update_layout(
            barmode="group",
            xaxis_title="窗口（天）",
            yaxis_title="ICIR",
            hovermode="x unified",
        )
        st.plotly_chart(fig_gs1, use_container_width=True)

        st.subheader("IS vs OOS IC 均值")
        fig_gs2 = go.Figure()
        fig_gs2.add_trace(
            go.Bar(
                x=gs_df.index,
                y=gs_df["IS_IC均值"],
                name="样本内 IC",
                marker_color="#4CAF50",
                opacity=0.8,
            )
        )
        fig_gs2.add_trace(
            go.Bar(
                x=gs_df.index,
                y=gs_df["OOS_IC均值"],
                name="样本外 IC",
                marker_color="#F44336",
                opacity=0.8,
            )
        )
        fig_gs2.add_hline(y=0, line_color="gray", line_width=1)
        fig_gs2.update_layout(
            barmode="group",
            xaxis_title="窗口（天）",
            yaxis_title="IC 均值",
            hovermode="x unified",
        )
        st.plotly_chart(fig_gs2, use_container_width=True)

        st.subheader("详细结果")
        st.dataframe(gs_df.style.format("{:.4f}", na_rep="—"), use_container_width=True)

        best_w = gs_df["OOS_ICIR"].idxmax()
        oos_val = cast(float, gs_df.loc[best_w, "OOS_ICIR"])
        is_val = cast(float, gs_df.loc[best_w, "IS_ICIR"])
        gap = abs(is_val - oos_val)
        st.info(
            f"**推荐参数：{best_w} 天**  |  OOS ICIR = {oos_val:.3f}  |  "
            f"IS/OOS 差距 = {gap:.3f}"
            + ("（⚠️ IS/OOS 差距较大，可能过拟合）" if gap > 0.3 else "（✅ 较为稳健）")
        )
    else:
        st.info("选择因子和参数范围后，点击「运行网格搜索」。首次约需 30 秒。")


# ──────────────────────────────────────────────────────────────────────────
# Tab 4：自主研究
# ──────────────────────────────────────────────────────────────────────────
with tab_research:
    st.subheader("因子验证结论")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("主力资金因子 ICIR", "0.799", "强有效 ✅", delta_color="normal")
    r2.metric("盈利质量因子 ICIR", "0.216", "弱有效 ⚠️", delta_color="off")
    r3.metric("财务加速度 ICIR", "0.146", "条件有效 ⚠️", delta_color="off")
    r4.metric("估值变化因子 IC", "-0.039", "放弃 ❌", delta_color="inverse")

    st.divider()

    sub1, sub2, sub3 = st.tabs(["📊 因子分层回测", "📉 IC 深度分析", "🌍 市场环境分解"])

    with sub1:
        _factor_imgs = {
            "主力资金因子（20日动量）": "factor_layered_fund_flow_w20.png",
            "盈利质量因子（CFO/净利润）": "factor_layered_earnings_quality.png",
            "财务加速度因子": "factor_layered_revenue_acceleration.png",
        }
        selected_img = st.selectbox("选择因子", list(_factor_imgs.keys()))
        img_path = _IMG_DIR / _factor_imgs[selected_img]
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.warning("图表文件不存在，请先运行对应研究脚本。")

        with st.expander("解读方式"):
            st.markdown("""
            - **左图 IC 时序**：IC 均值越高、时序越稳定，因子越可靠
            - **中图 分层收益**：G1（低分）→ G5（高分）应单调递增（多头因子）
            - **右图 多空累计收益**：G5 - G1 的累计超额收益，斜率越稳定越好
            """)

    with sub2:
        deep_path = _IMG_DIR / "factor_deep_analysis.png"
        if deep_path.exists():
            st.image(str(deep_path), use_container_width=True)
        else:
            st.warning("请先运行 scripts/research_factor_analysis.py")

        st.subheader("IC 衰减汇总")
        decay_data = {
            "预测窗口": [5, 10, 20, 40, 60],
            "主力资金 IC均值": [0.023, 0.056, 0.091, 0.128, 0.155],
            "主力资金 ICIR": [0.12, 0.29, 0.48, 0.65, 0.76],
            "财务加速度 IC均值": [0.018, 0.020, 0.021, 0.019, 0.018],
            "财务加速度 ICIR": [0.10, 0.11, 0.12, 0.10, 0.09],
        }
        df_decay = pd.DataFrame(decay_data).set_index("预测窗口")

        fig_decay = go.Figure()
        fig_decay.add_trace(
            go.Scatter(
                x=df_decay.index,
                y=df_decay["主力资金 ICIR"],
                name="主力资金",
                mode="lines+markers",
                line=dict(color="steelblue", width=2),
            )
        )
        fig_decay.add_trace(
            go.Scatter(
                x=df_decay.index,
                y=df_decay["财务加速度 ICIR"],
                name="财务加速度",
                mode="lines+markers",
                line=dict(color="darkorange", width=2),
            )
        )
        fig_decay.add_hline(
            y=0.3, line_dash="dash", line_color="green", annotation_text="有效阈值 0.3"
        )
        fig_decay.update_layout(
            title="IC 衰减曲线（ICIR vs 预测窗口）",
            xaxis_title="预测窗口（交易日）",
            yaxis_title="ICIR",
            hovermode="x unified",
        )
        st.plotly_chart(fig_decay, use_container_width=True)

    with sub3:
        regime_data = {
            "市场状态": ["BULL", "RANGE", "BEAR"],
            "主力资金 IC均值": [0.098, 0.085, 0.072],
            "主力资金 ICIR": [0.52, 0.44, 0.38],
            "财务加速度 IC均值": [0.031, 0.019, 0.002],
            "财务加速度 ICIR": [0.18, 0.11, 0.01],
        }
        df_regime_res = pd.DataFrame(regime_data).set_index("市场状态")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_regime_bar = go.Figure()
            for r_state in df_regime_res.index:
                fig_regime_bar.add_trace(
                    go.Bar(
                        name=r_state,
                        x=["主力资金", "财务加速度"],
                        y=[
                            df_regime_res.loc[r_state, "主力资金 IC均值"],
                            df_regime_res.loc[r_state, "财务加速度 IC均值"],
                        ],
                        marker_color=_REGIME_COLORS[r_state],
                    )
                )
            fig_regime_bar.add_hline(y=0, line_color="black", line_width=0.8)
            fig_regime_bar.update_layout(
                title="分市场环境 IC 均值", barmode="group", yaxis_title="IC 均值"
            )
            st.plotly_chart(fig_regime_bar, use_container_width=True)

        with col_b:
            st.markdown("""
            **关键结论：**

            | 因子 | 牛市 | 震荡 | 熊市 |
            |------|------|------|------|
            | 主力资金   | ✅ 有效 | ✅ 有效 | ✅ 有效 |
            | 财务加速度 | ✅ 有效 | ⚠️ 弱  | ❌ 失效 |

            **投资含义：**
            - 主力资金因子是全天候因子，可作为核心信号
            - 财务加速度在熊市应关闭（设为 0 权重）
            - 两者相关性低，可合成复合因子
            """)
