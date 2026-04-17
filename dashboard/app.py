import streamlit as st

st.title("Quant-Lab Dashboard")

# 侧边栏导航
page = st.sidebar.selectbox("选择页面", ["回测结果", "因子分析", "风险报告"])

if page == "回测结果":
    st.header("回测结果")
    # TODO: 这里放回测相关的内容

if page == "因子分析":
    st.header("因子分析")
    # TODO: 这里放回测相关的内容

if page == "风险报告":
    st.header("风险报告")
    # TODO: 这里放回测相关的内容
