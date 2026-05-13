import sys
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.shared import load_close, sidebar_config
from quant.macro.indicators import calc_lag_corr, composite_index
from quant.macro.loader import load_all_macro
from quant.regime.detector import detect_regime
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

st.set_page_config(page_title="市场环境", layout="wide")

cfg = sidebar_config()
st.title("🌐 市场环境")

_REGIME_COLOR = {"BULL": "#4CAF50", "RANGE": "#FF9800", "BEAR": "#F44336"}
_REGIME_ICON = {"BULL": "🟢", "RANGE": "🟡", "BEAR": "🔴"}
_MACRO_LABEL = {
    "bond_yield": "十年期国债收益率",
    "pmi": "制造业 PMI",
    "m2_yoy": "M2 同比",
    "cpi_yoy": "CPI 同比",
}


@st.cache_data(show_spinner="加载申万行业数据…")
def _load_sector() -> tuple[pd.DataFrame, pd.Series]:
    sector_close = load_sector_close()
    return sector_close, sector_close.mean(axis=1)


@st.cache_data(show_spinner="加载宏观数据…")
def _load_macro() -> pd.DataFrame:
    return load_all_macro()


@st.cache_data(show_spinner="计算宏观景气度指数…")
def _get_composite(macro_df: pd.DataFrame) -> pd.Series:
    return composite_index(macro_df)


@st.cache_data(show_spinner="计算滞后相关性…")
def _get_lag_corrs(macro_df: pd.DataFrame) -> dict[str, pd.Series]:
    close = load_close()
    market_ret = close.pct_change().mean(axis=1)
    return {col: calc_lag_corr(macro_df[col], market_ret) for col in macro_df.columns}


tab_regime, tab_sector, tab_macro = st.tabs(
    ["🌡️ 市场状态", "🔄 行业轮动", "📊 宏观因子"]
)

# Tab 1：市场状态
with tab_regime:
    close = load_close()
    regime = detect_regime(
        close,
        ma_window=cfg.regime.ma_window,
        breadth_window=cfg.regime.breadth_window,
        vol_short=cfg.regime.vol_short,
        vol_long=cfg.regime.vol_long,
    )

    latest_date = regime.index[-1]
    latest_state = regime.iloc[-1]
    st.subheader(
        f"当前状态（{latest_date.date()}）："
        f"{_REGIME_ICON.get(latest_state, '')} {latest_state}"
    )

    col_pie, col_bar = st.columns([1, 2])

    with col_pie:
        st.subheader("历史状态分布")
        counts = regime.value_counts()
        fig_pie = px.pie(
            values=counts.values,
            names=counts.index,
            color=counts.index,
            color_discrete_map=_REGIME_COLOR,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        st.subheader("各年度状态占比")
        regime_df = regime.to_frame("regime").assign(
            year=pd.DatetimeIndex(regime.index).year
        )
        yearly = regime_df.groupby(["year", "regime"]).size().unstack(fill_value=0)
        for s in ["BULL", "RANGE", "BEAR"]:
            if s not in yearly.columns:
                yearly[s] = 0
        yearly_pct = yearly[["BULL", "RANGE", "BEAR"]].div(yearly.sum(axis=1), axis=0)

        fig_yearly = go.Figure()
        for state, clr in _REGIME_COLOR.items():
            fig_yearly.add_trace(
                go.Bar(
                    x=yearly_pct.index,
                    y=yearly_pct[state],
                    name=state,
                    marker_color=clr,
                )
            )
        fig_yearly.update_layout(
            barmode="stack",
            yaxis_tickformat=".0%",
            xaxis_title="年份",
            yaxis_title="占比",
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

    st.subheader("近 120 日市场状态")
    index_close = close.mean(axis=1).tail(120)
    recent_regime = regime.tail(120)

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=index_close.index,
            y=index_close.values,
            name="指数均值",
            line=dict(color="#2196F3"),
        )
    )
    for state, clr in _REGIME_COLOR.items():
        mask = recent_regime == state
        fig_line.add_trace(
            go.Scatter(
                x=index_close[mask].index,
                y=index_close[mask].values,
                mode="markers",
                name=state,
                marker=dict(color=clr, size=5),
            )
        )
    fig_line.update_layout(
        xaxis_title="日期", yaxis_title="价格", hovermode="x unified"
    )
    st.plotly_chart(fig_line, use_container_width=True)

# Tab 2：行业轮动
with tab_sector:
    rs_window = st.slider("RS 计算窗口（天）", 5, 60, 20, step=5)
    lookback = st.slider("RS 动量回看（天）", 10, 60, 20, step=5)
    top_n = st.slider("超配/低配各 N 个", 1, 5, 3)

    if st.button("📥 加载行业数据", type="primary", key="btn_sector"):
        sector_close, benchmark = _load_sector()
        st.session_state["sector_close"] = sector_close
        st.session_state["sector_benchmark"] = benchmark
        sector_regime = detect_regime(sector_close)
        st.session_state["sector_regime"] = sector_regime

    if "sector_close" in st.session_state:
        sector_close = cast(pd.DataFrame, st.session_state["sector_close"])
        benchmark = cast(pd.Series, st.session_state["sector_benchmark"])
        sector_regime = cast(pd.Series, st.session_state["sector_regime"])
        current_regime = sector_regime.iloc[-1]

        st.metric(
            "当前市场环境", f"{_REGIME_ICON.get(current_regime, '')} {current_regime}"
        )

        rs = calc_rs(sector_close, benchmark, window=rs_window)
        rs_momentum = calc_rs_momentum(rs, lookback=lookback)
        rs_latest = rs.iloc[-1]
        suggestions = get_suggestions(rs_latest, rs_momentum, top_n=top_n)

        # 市场环境历史
        st.subheader("市场环境历史")
        regime_num = sector_regime.map({"BULL": 1, "RANGE": 0, "BEAR": -1})
        fig_regime_sec = go.Figure()
        fig_regime_sec.add_trace(
            go.Scatter(
                x=sector_regime.index,
                y=regime_num,
                mode="markers",
                marker=dict(
                    color=[_REGIME_COLOR[r] for r in sector_regime],
                    size=8,
                    symbol="square",
                ),
                showlegend=False,
            )
        )
        for label, clr in _REGIME_COLOR.items():
            fig_regime_sec.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=clr, size=10, symbol="square"),
                    name=label,
                )
            )
        fig_regime_sec.update_layout(
            yaxis=dict(tickvals=[-1, 0, 1], ticktext=["BEAR", "RANGE", "BULL"]),
            height=250,
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_regime_sec, use_container_width=True)

        # RS 热力图
        st.subheader("行业 RS 热力图（近 12 个月）")
        rs_monthly = rs.resample("ME").last().tail(12).T
        fig_heat = px.imshow(
            rs_monthly,
            color_continuous_scale="RdYlGn",
            zmin=0.5,
            zmax=1.5,
            text_auto=".2f",  # type: ignore[arg-type]
            aspect="auto",
            labels={"color": "RS"},
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # 当前 RS 排名
        st.subheader("当前 RS 排名")
        rs_sorted = rs_latest.sort_values()
        fig_rs_bar = go.Figure(
            go.Bar(
                x=rs_sorted.values,
                y=rs_sorted.index,
                orientation="h",
                marker_color=[
                    "#4CAF50" if v > 1 else "#F44336" for v in rs_sorted.values
                ],
            )
        )
        fig_rs_bar.add_vline(x=1, line_dash="dash", line_color="gray")
        fig_rs_bar.update_layout(xaxis_title="RS（>1 强于基准）", height=500)
        st.plotly_chart(fig_rs_bar, use_container_width=True)

        # 超配/低配建议
        st.subheader("超配/低配建议")
        st.dataframe(
            suggestions.style.format(
                {
                    "RS": "{:.3f}",
                    "RS动量": "{:.4f}",
                    "RS排名": "{:.0f}",
                    "动量排名": "{:.0f}",
                    "综合排名": "{:.1f}",
                }
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "点击「加载行业数据」开始，"
            "首次需拉取所有行业历史数据约需 1~2 分钟，之后走本地缓存。"
        )


# Tab 3：宏观因子
with tab_macro:
    if st.button("📥 加载宏观数据", type="primary", key="btn_macro"):
        st.session_state["macro_df"] = _load_macro()

    if "macro_df" in st.session_state:
        macro_df: pd.DataFrame = st.session_state["macro_df"]
        score = _get_composite(macro_df)
        lag_corrs = _get_lag_corrs(macro_df)

        # 景气度合成指数
        st.subheader("宏观景气度合成指数")
        st.caption("PMI + M2 + CPI 正向，国债收益率反向，z-score 标准化后等权合成")
        fig_score = go.Figure()
        fig_score.add_trace(
            go.Scatter(
                x=score.index,
                y=score.values,
                mode="lines",
                line=dict(color="#2196F3", width=2),
                name="景气度",
            )
        )
        fig_score.add_hline(y=0, line_color="gray", line_width=0.8, line_dash="dash")
        fig_score.update_layout(
            xaxis_title="日期", yaxis_title="景气度得分", hovermode="x unified"
        )
        st.plotly_chart(fig_score, use_container_width=True)

        # 各指标走势（2×2 子图）
        st.subheader("各宏观指标走势")
        indicator_cols = list(macro_df.columns)
        fig_sub = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[_MACRO_LABEL[c] for c in indicator_cols],
        )
        subplot_colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
        for i, (ind_col, clr) in enumerate(zip(indicator_cols, subplot_colors)):
            row, col_idx = divmod(i, 2)
            fig_sub.add_trace(
                go.Scatter(
                    x=macro_df.index,
                    y=macro_df[ind_col],
                    mode="lines",
                    line=dict(color=clr, width=1.5),
                    name=_MACRO_LABEL[ind_col],
                    showlegend=False,
                ),
                row=row + 1,
                col=col_idx + 1,
            )
        fig_sub.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig_sub, use_container_width=True)

        # 滞后相关性
        st.subheader("宏观指标与大盘的滞后相关性")
        st.caption("lag=N 表示该指标领先大盘 N 个月时的 Pearson 相关系数")
        fig_lag = go.Figure()
        for ind_col, corr_series in lag_corrs.items():
            fig_lag.add_trace(
                go.Bar(
                    x=corr_series.index,
                    y=corr_series.values,
                    name=_MACRO_LABEL.get(ind_col, ind_col),
                )
            )
        fig_lag.add_hline(y=0, line_color="gray", line_width=0.8)
        fig_lag.update_layout(
            barmode="group",
            xaxis_title="滞后月数",
            yaxis_title="相关系数",
            hovermode="x unified",
        )
        st.plotly_chart(fig_lag, use_container_width=True)

        # 最新宏观快照
        st.subheader("最新宏观快照")
        latest_macro = macro_df.dropna(how="all").iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(_MACRO_LABEL["bond_yield"], f"{latest_macro['bond_yield']:.2f}%")
        c2.metric(_MACRO_LABEL["pmi"], f"{latest_macro['pmi']:.1f}")
        c3.metric(_MACRO_LABEL["m2_yoy"], f"{latest_macro['m2_yoy']:.1f}%")
        c4.metric(_MACRO_LABEL["cpi_yoy"], f"{latest_macro['cpi_yoy']:.1f}%")
    else:
        st.info("点击「加载宏观数据」开始，首次约需 10~30 秒。")
