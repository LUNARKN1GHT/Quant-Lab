import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from quant.pipeline import run_pipeline

st.info(
    "数据来源：AKShare，股票池：600519 / 600036 / 601318 / 000333 / 000858"
    + " 区间：2022-2024"
)
st.title("Quant-Lab Dashboard")

# 侧边栏导航
page = st.sidebar.selectbox(
    "选择页面", ["回测结果", "因子分析", "ML 模型", "风险报告", "归因分析"]
)


@st.cache_data
def load_data():
    return run_pipeline(
        symbols=["600519", "600036", "601318", "000333", "000858"],
        start=datetime(2022, 1, 1),
        end=datetime(2024, 12, 31),
    )


if page == "回测结果":
    st.header("回测结果")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    result = load_data()
    daily_returns = result["returns"]
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

    result = load_data()
    daily_returns = result["returns"]
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

elif page == "归因分析":
    st.header("Brinson 归因分析")
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from quant.risk.attribution import brinson

    # 模拟行业数据
    sectors = ["科技", "金融", "消费", "医疗", "能源"]
    np.random.seed(42)
    portfolio_weights = pd.Series([0.35, 0.20, 0.25, 0.15, 0.05], index=sectors)
    benchmark_weights = pd.Series([0.20, 0.30, 0.25, 0.15, 0.10], index=sectors)
    portfolio_returns = pd.Series(np.random.randn(5) * 0.03 + 0.02, index=sectors)
    benchmark_returns = pd.Series(np.random.randn(5) * 0.02 + 0.015, index=sectors)

    result = brinson(
        portfolio_weights, benchmark_weights, portfolio_returns, benchmark_returns
    )

    # 柱状图
    fig = go.Figure()
    for col in ["allocation", "selection", "interaction"]:
        fig.add_bar(x=result.index, y=result[col], name=col)
    fig.update_layout(barmode="group", title="各行业 Brinson 归因分解")
    st.plotly_chart(fig, use_container_width=True)

    # 汇总表格
    st.subheader("超额收益汇总")
    summary = result.sum().rename("合计")
    st.table(pd.DataFrame(summary).T)
