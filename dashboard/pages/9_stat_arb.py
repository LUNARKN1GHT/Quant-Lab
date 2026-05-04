"""阶段二十二：统计套利深化研究成果展示"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from dashboard.shared import sidebar_config

st.set_page_config(page_title="统计套利", layout="wide")
sidebar_config()
st.title("📐 统计套利深化（阶段二十二）")

IMG_DIR = Path(__file__).parent.parent.parent / "output" / "factor_research"
PAIR_CSV = IMG_DIR / "cointegrated_pairs.csv"

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔗 协整分析",
        "〰️ OU 过程",
        "📡 Kalman Filter",
        "🧩 PCA 篮子",
        "⚖️ 市场中性",
    ]
)

# ── Tab 1：协整分析 ────────────────────────────────────────────────────────────
with tab1:
    st.subheader("批量协整对筛选结果")

    if PAIR_CSV.exists():
        df_pairs = pd.read_csv(PAIR_CSV)
        st.dataframe(
            df_pairs.style.format(precision=4).background_gradient(
                subset=["eg_pvalue"] if "eg_pvalue" in df_pairs.columns else [],
                cmap="RdYlGn_r",
            ),
            use_container_width=True,
        )
    else:
        st.warning("请先运行 scripts/find_cointegrated_pairs.py")

    img_path = IMG_DIR / "cointegration_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    with st.expander("方法论：EG vs Johansen"):
        st.markdown("""
        | 方法 | 原理 | 优点 | 缺点 |
        |------|------|------|------|
        | Engle-Granger | OLS 残差 ADF 检验 | 简单直观 | 只能检测一个协整关系 |
        | Johansen | VAR 模型特征根 | 可检测多个协整关系，方向对称 | 参数敏感，小样本偏差 |

        **实践建议**：两者同时通过（EG p<0.05 且 Johansen n_cointegration≥1）才选为候选对。
        """)

# ── Tab 2：OU 过程 ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("OU 过程参数与半衰期分析")

    col1, col2, col3 = st.columns(3)
    col1.metric("最佳配对", "工商银行 vs 建设银行")
    col2.metric("历史半衰期", "5~25 天（2019-2025）")
    col3.metric("当前半衰期", "75+ 天（2026，不宜交易）")

    img_path = IMG_DIR / "ou_process_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.markdown("""
    **OU 过程参数解读：**
    - **θ（均值回归速度）**：越大回归越快，建议 θ > 0.03（对应半衰期 < 23天）
    - **半衰期 = ln(2)/θ**：策略持仓周期的理论上限
    - **σ（波动率）**：越大信噪比越高，开仓机会越多

    **当前状态（2026）**：θ 降至约 0.02，半衰期超 75 天，该对暂不适合配对交易。
    """)

# ── Tab 3：Kalman Filter ──────────────────────────────────────────────────────
with tab3:
    st.subheader("动态对冲比率：Kalman Filter vs 静态 OLS")

    col1, col2, col3 = st.columns(3)
    col1.metric("静态 OLS ADF", "p≈0.05（边界）")
    col2.metric("Kalman Filter ADF", "p<0.05（大部分时间）✅")
    col3.metric("KF 对冲比率范围", "0.5~0.75（平滑跟踪）")

    img_path = IMG_DIR / "kalman_hedge_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    with st.expander("Kalman Filter 状态空间模型"):
        st.markdown(r"""
        **观测方程：** $\text{price\_a}_t = \beta_t \cdot \text{price\_b}_t + \alpha_t + \varepsilon_t$

        **状态方程：** $[\beta_t, \alpha_t] = [\beta_{t-1}, \alpha_{t-1}] + w_t$

        其中 $w_t \sim \mathcal{N}(0, Q)$ 为过程噪声，控制对冲比率的漂移速度。

        **参数 delta**：$Q = \frac{\delta}{1-\delta} I$，delta=1e-4 时 β 每日变动约 1%。
        """)

# ── Tab 4：PCA 篮子 ───────────────────────────────────────────────────────────
with tab4:
    st.subheader("PCA 篮子套利 — 实证发现")

    col1, col2 = st.columns(2)
    with col1:
        st.error("均值回归方向：SR = -6.38（失败）")
    with col2:
        st.success("动量方向（信号翻转）：SR = +6.38（有效）")

    img_path = IMG_DIR / "pca_basket_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.markdown("""
    **核心发现：A 股行业内特质残差具有动量特性**

    | 板块 | PC1 解释度 | 均值回归 SR | 动量 SR |
    |------|-----------|------------|---------|
    | 银行 | 65% | -6.70 | +6.70 |
    | 白酒 | 75% | -6.38 | +6.38 |

    **原因**：A 股中散户资金驱动的趋势效应强于机构套利带来的均值回归，
    行业内跑赢股票短期内倾向继续跑赢（"强者恒强"）。
    """)

# ── Tab 5：市场中性 ───────────────────────────────────────────────────────────
with tab5:
    st.subheader("Beta 中性组合 — 主力资金因子")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharpe Ratio", "0.91")
    col2.metric("组合波动率", "~8%（年化）")
    col3.metric("市场相关性", "≈ 0.1（中性化有效）✅")

    img_path = IMG_DIR / "market_neutral_analysis.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

    st.markdown("""
    **Beta 中性化效果验证：**
    - 原始多空组合与市场相关性约 -0.3~-0.5（空头暴露过大）
    - Beta 中性后相关性压缩至 ≈ 0.1 ✅
    - 当前数据窗口仅 5 个月（fund_flow 历史不足），需补充历史数据

    **行业中性化**：当前使用代码前缀粗分，实盘应接入申万一级行业分类。
    """)
