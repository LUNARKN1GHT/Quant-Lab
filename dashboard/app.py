import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.title("Quant-Lab Dashboard")

# 侧边栏导航
page = st.sidebar.selectbox("选择页面", ["回测结果", "因子分析", "ML 模型", "风险报告"])

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

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # TODO：Demo 数据：模拟 24 个月的 IC 序列
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=24, freq="ME")
    ic_series = pd.Series(np.random.randn(24) * 0.05 + 0.03, index=dates)

    # IC bar 图
    fig = go.Figure()
    fig.add_bar(x=ic_series.index, y=ic_series.values, name="月度 IC")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="因子 IC 序列", xaxis_title="日期", yaxis_title="IC")
    st.plotly_chart(fig, use_container_width=True)

    # ICIR 摘要
    from quant.factor.ic import calc_icir

    st.metric("ICIR", f"{calc_icir(ic_series):.2f}")

elif page == "风险报告":
    st.header("风险报告")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from quant.risk.metrics import calmar, cvar, max_drawdown, sharpe, sortino, var

    # TODO：Demo 数据
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    daily_returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    cum_returns = (1 + daily_returns).cumprod()

    # 回撤曲线
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", name="回撤")
    )
    fig.update_layout(title="回撤曲线", xaxis_title="日期", yaxis_title="回撤幅度")
    st.plotly_chart(fig, use_container_width=True)

    # 风险指标表格
    metrics = {
        "Sharpe Ratio": f"{sharpe(daily_returns):.2f}",
        "Sortino Ratio": f"{sortino(daily_returns):.2f}",
        "最大回撤": f"{max_drawdown(daily_returns):.2%}",
        "Calmar Ratio": f"{calmar(daily_returns):.2f}",
        "VaR (95%)": f"{var(daily_returns):.2%}",
        "CVaR (95%)": f"{cvar(daily_returns):.2%}",
    }
    st.table(pd.DataFrame(metrics.items(), columns=["指标", "值"]))

elif page == "ML 模型":
    st.header("ML Alpha - Walk-forward 预测")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # TODO：Demo 数据
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    actual = pd.Series(np.random.randn(n) * 0.01, index=dates)
    # 模拟预测值（加一点噪声）
    predicted = actual + pd.Series(np.random.randn(n) * 0.005, index=dates)
    predicted.iloc[:20] = None  # 前 20 期无预测（训练期）

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, name="实际收益"))
    fig.add_trace(go.Scatter(x=dates, y=predicted, name="预测收益"))
    fig.update_layout(
        title="Walk-forward 预测 vs 实际", xaxis_title="日期", yaxis_title="收益率"
    )
    st.plotly_chart(fig, use_container_width=True)
