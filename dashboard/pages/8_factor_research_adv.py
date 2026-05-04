"""阶段二十一：自主因子研究成果展示"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import sidebar_config

st.set_page_config(page_title="自主因子研究", layout="wide")
sidebar_config()
st.title("🔬 自主因子研究（阶段二十一）")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "quant.duckdb"
IMG_DIR = Path(__file__).parent.parent.parent / "output" / "factor_research"

# ── 顶部结论卡片 ─────────────────────────────────────────────────────────────
st.subheader("因子验证结论")
col1, col2, col3, col4 = st.columns(4)
col1.metric("主力资金因子 ICIR", "0.799", "强有效 ✅", delta_color="normal")
col2.metric("盈利质量因子 ICIR", "0.216", "弱有效 ⚠️", delta_color="off")
col3.metric("财务加速度 ICIR", "0.146", "条件有效 ⚠️", delta_color="off")
col4.metric("估值变化因子 IC", "-0.039", "放弃 ❌", delta_color="inverse")

st.divider()

tab1, tab2, tab3 = st.tabs(["📊 因子分层回测", "📉 IC 深度分析", "🌍 市场环境分解"])

# ── Tab 1：因子分层图 ─────────────────────────────────────────────────────────
with tab1:
    factor_imgs = {
        "主力资金因子（20日动量）": "factor_layered_fund_flow_w20.png",
        "盈利质量因子（CFO/净利润）": "factor_layered_earnings_quality.png",
        "财务加速度因子": "factor_layered_revenue_acceleration.png",
    }
    selected = st.selectbox("选择因子", list(factor_imgs.keys()))
    img_path = IMG_DIR / factor_imgs[selected]
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

# ── Tab 2：IC 衰减 ─────────────────────────────────────────────────────────────
with tab2:
    img_path = IMG_DIR / "factor_deep_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning("请先运行 scripts/research_factor_analysis.py")

    st.subheader("IC 衰减汇总（手动录入结论）")
    decay_data = {
        "预测窗口": [5, 10, 20, 40, 60],
        "主力资金 IC均值": [0.023, 0.056, 0.091, 0.128, 0.155],
        "主力资金 ICIR": [0.12, 0.29, 0.48, 0.65, 0.76],
        "财务加速度 IC均值": [0.018, 0.020, 0.021, 0.019, 0.018],
        "财务加速度 ICIR": [0.10, 0.11, 0.12, 0.10, 0.09],
    }
    df_decay = pd.DataFrame(decay_data).set_index("预测窗口")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_decay.index,
            y=df_decay["主力资金 ICIR"],
            name="主力资金",
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_decay.index,
            y=df_decay["财务加速度 ICIR"],
            name="财务加速度",
            mode="lines+markers",
            line=dict(color="darkorange", width=2),
        )
    )
    fig.add_hline(
        y=0.3, line_dash="dash", line_color="green", annotation_text="有效阈值 0.3"
    )
    fig.update_layout(
        title="IC 衰减曲线（ICIR vs 预测窗口）",
        xaxis_title="预测窗口（交易日）",
        yaxis_title="ICIR",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 3：市场环境分解 ────────────────────────────────────────────────────────
with tab3:
    regime_data = {
        "市场状态": ["BULL", "RANGE", "BEAR"],
        "主力资金 IC均值": [0.098, 0.085, 0.072],
        "主力资金 ICIR": [0.52, 0.44, 0.38],
        "财务加速度 IC均值": [0.031, 0.019, 0.002],
        "财务加速度 ICIR": [0.18, 0.11, 0.01],
    }
    df_regime = pd.DataFrame(regime_data).set_index("市场状态")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = go.Figure()
        colors = {"BULL": "#2ca02c", "RANGE": "#ff7f0e", "BEAR": "#d62728"}
        for regime in df_regime.index:
            fig1.add_trace(
                go.Bar(
                    name=regime,
                    x=["主力资金", "财务加速度"],
                    y=[
                        df_regime.loc[regime, "主力资金 IC均值"],
                        df_regime.loc[regime, "财务加速度 IC均值"],
                    ],
                    marker_color=colors[regime],
                )
            )
        fig1.add_hline(y=0, line_color="black", line_width=0.8)
        fig1.update_layout(
            title="分市场环境 IC 均值", barmode="group", yaxis_title="IC 均值"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        st.markdown("""
        **关键结论：**

        | 因子 | 牛市 | 震荡 | 熊市 |
        |------|------|------|------|
        | 主力资金 | ✅ 有效 | ✅ 有效 | ✅ 有效 |
        | 财务加速度 | ✅ 有效 | ⚠️ 弱 | ❌ 失效 |

        **投资含义：**
        - 主力资金因子是全天候因子，可作为核心信号
        - 财务加速度在熊市应关闭（设为 0 权重）
        - 两者相关性低，可合成复合因子
        """)
