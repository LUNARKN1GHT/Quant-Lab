import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.title("Quant-Lab Dashboard")

# 侧边栏导航
page = st.sidebar.selectbox("选择页面", ["回测结果", "因子分析", "风险报告"])

if page == "回测结果":
    st.header("回测结果")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # TODO: Demo 数据，模拟 252 天的收益率
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    daily_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    cum_returns = (1 + daily_returns).cumprod()

    # 绘制累积收益曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns.values, name="策略"))
    fig.update_layout(title="累积收益曲线", xaxis_title="日期", yaxis_title="精值")
    st.plotly_chart(fig, use_container_width=True)

    from quant.risk.metrics import calmar, max_drawdown, sharpe

    col1, col2, col3 = st.columns(3)
    col1.metric("Sharp Ratio", f"{sharpe(daily_returns):.2f}")
    col2.metric("Max Drawdown", f"{max_drawdown(daily_returns):.2f}")
    col3.metric("Calmar Ratio", f"{calmar(daily_returns):.2f}")

elif page == "因子分析":
    st.header("因子分析")
    # TODO: 这里放回测相关的内容

elif page == "风险报告":
    st.header("风险报告")
    # TODO: 这里放回测相关的内容
