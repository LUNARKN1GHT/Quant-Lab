"""今日报告 — 一键汇总当日仓位建议、宏观景气、行业轮动"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.shared import load_close, sidebar_config
from quant.advisor.position import compute_position
from quant.macro.indicators import composite_index
from quant.macro.loader import load_all_macro
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

st.set_page_config(page_title="今日报告", layout="wide")

cfg = sidebar_config()
st.title("📰 今日报告")
st.caption(f"报告日期：{date.today().strftime('%Y-%m-%d')}")

_REGIME_ZH = {
    "BULL": ("牛市", "🟢"),
    "RANGE": ("震荡", "🟡"),
    "BEAR": ("熊市", "🔴"),
}


# ── 数据加载（独立缓存，互不阻塞）─────────────────────────────────────────────
@st.cache_data(show_spinner="加载宏观数据…")
def _get_macro() -> tuple[pd.DataFrame, pd.Series]:
    macro_df = load_all_macro()
    return macro_df, composite_index(macro_df)


@st.cache_data(show_spinner="计算仓位建议…")
def _get_position(_cfg_hash: str) -> pd.DataFrame:
    close = load_close()
    _, macro_score = _get_macro()
    return compute_position(close, cfg, macro_score=macro_score)


@st.cache_data(show_spinner="加载行业数据…")
def _get_sector_suggestions() -> pd.DataFrame:
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)
    rs = calc_rs(sector_close, benchmark, window=20)
    rs_momentum = calc_rs_momentum(rs, lookback=20)
    return get_suggestions(rs.iloc[-1], rs_momentum, top_n=3)


# ── 触发 ───────────────────────────────────────────────────────────────────────
if st.button("🚀 生成今日报告", type="primary"):
    st.session_state["report_ready"] = True

if not st.session_state.get("report_ready"):
    st.info("点击「生成今日报告」开始。结果会缓存，重复点击秒级响应。")
    st.stop()

# ── 1. 仓位建议 ────────────────────────────────────────────────────────────────
st.subheader("📊 仓位建议")
result = _get_position(str(cfg.regime))
latest = result.iloc[-1]
prev_pos = result["position"].iloc[-2]
regime_name, regime_icon = _REGIME_ZH.get(latest["regime"], (latest["regime"], "⚪"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("市场状态", f"{regime_icon} {regime_name}")
c2.metric("Regime 信号", f"{latest['regime_signal']:.0%}")
c3.metric("波动率信号", f"{latest['vol_signal']:.0%}")
c4.metric("宏观信号", f"{latest['macro_signal']:.0%}")
c5.metric(
    "建议仓位",
    f"{latest['position']:.0%}",
    f"{latest['position'] - prev_pos:+.1%}",
)

st.divider()

# ── 2. 宏观景气 ────────────────────────────────────────────────────────────────
st.subheader("🌍 宏观景气")
macro_df, macro_score = _get_macro()
latest_macro = macro_df.dropna(how="all").iloc[-1]
latest_score = macro_score.dropna().iloc[-1]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("景气度得分", f"{latest_score:+.2f}")
m2.metric("十年期国债", f"{latest_macro['bond_yield']:.2f}%")
m3.metric("制造业 PMI", f"{latest_macro['pmi']:.1f}")
m4.metric("M2 同比", f"{latest_macro['m2_yoy']:.1f}%")
m5.metric("CPI 同比", f"{latest_macro['cpi_yoy']:.1f}%")

st.divider()

# ── 3. 行业轮动 ────────────────────────────────────────────────────────────────
st.subheader("🔄 行业轮动")
sector_sug = _get_sector_suggestions()
overweight = sector_sug[sector_sug["建议"] == "超配 ▲"][["RS", "RS动量"]]
underweight = sector_sug[sector_sug["建议"] == "低配 ▼"][["RS", "RS动量"]]

col_o, col_u = st.columns(2)
with col_o:
    st.markdown("**🟢 超配行业**")
    st.dataframe(
        overweight.style.format({"RS": "{:.3f}", "RS动量": "{:+.4f}"}),
        width="stretch",
    )
with col_u:
    st.markdown("**🔴 低配行业**")
    st.dataframe(
        underweight.style.format({"RS": "{:.3f}", "RS动量": "{:+.4f}"}),
        width="stretch",
    )

st.divider()

# ── 4. 近期 Regime 切换 ────────────────────────────────────────────────────────
st.subheader("🔁 近 30 日 Regime 切换")
recent = result.tail(30)
changes = recent[recent["regime"] != recent["regime"].shift()].copy()
if changes.empty:
    st.info("近 30 日无市场状态切换")
else:
    changes["状态"] = changes["regime"].map(
        lambda r: f"{_REGIME_ZH.get(r, (r, '⚪'))[1]} {_REGIME_ZH.get(r, (r, ''))[0]}"
    )
    changes["建议仓位"] = changes["position"].map(lambda v: f"{v:.0%}")
    changes_display = changes[["状态", "建议仓位"]].copy()
    changes_display.index = changes_display.index.strftime("%Y-%m-%d")  # type: ignore
    st.dataframe(changes_display, width="stretch")

st.divider()


# ── 5. 导出 Markdown ───────────────────────────────────────────────────────────
def _build_markdown() -> str:
    lines = [f"# 量化日报 {date.today().strftime('%Y-%m-%d')}", ""]
    lines += [
        "## 仓位建议",
        "",
        "| 信号 | 数值 |",
        "|------|------|",
        f"| 市场状态 | {regime_icon} {regime_name} |",
        f"| Regime 信号 | {latest['regime_signal']:.0%} |",
        f"| 波动率信号 | {latest['vol_signal']:.0%} |",
        f"| 宏观信号 | {latest['macro_signal']:.0%} |",
        f"| **建议仓位** | **{latest['position']:.0%}** |",
        f"| 较昨日变化 | {latest['position'] - prev_pos:+.1%} |",
        "",
        "## 宏观景气",
        "",
        f"- 景气度合成得分：{latest_score:+.2f}",
        f"- 十年期国债：{latest_macro['bond_yield']:.2f}%",
        f"- 制造业 PMI：{latest_macro['pmi']:.1f}",
        f"- M2 同比：{latest_macro['m2_yoy']:.1f}%",
        f"- CPI 同比：{latest_macro['cpi_yoy']:.1f}%",
        "",
        "## 行业轮动",
        "",
        "**超配行业**",
        "",
    ]
    for idx, row in overweight.iterrows():
        lines.append(f"- {idx}：RS={row['RS']:.3f}，动量={row['RS动量']:+.4f}")
    lines += ["", "**低配行业**", ""]
    for idx, row in underweight.iterrows():
        lines.append(f"- {idx}：RS={row['RS']:.3f}，动量={row['RS动量']:+.4f}")
    return "\n".join(lines)


st.download_button(
    "📥 下载 Markdown 报告",
    data=_build_markdown(),
    file_name=f"report_{date.today().strftime('%Y%m%d')}.md",
    mime="text/markdown",
)
