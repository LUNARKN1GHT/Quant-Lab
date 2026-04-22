"""侧边栏 Config 面板 + 数据加载"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st

from quant.config import (
    AdvisorConfig,
    BacktestConfig,
    Config,
    DataConfig,
    FactorConfig,
    MLConfig,
    RegimeConfig,
)

DATA_DIR = Path(__file__).parent.parent / "data/csi300"


def sidebar_config() -> Config:
    """侧边栏参数面板，返回当前 Config 对象"""
    st.sidebar.subheader("市场状态")
    ma_window = st.sidebar.slider("均线窗口", 20, 250, 120, step=10)
    bull_scale = st.sidebar.slider("BULL 仓位上限", 0.5, 1.0, 1.0, step=0.05)
    range_scale = st.sidebar.slider("RANGE 仓位上限", 0.2, 0.8, 0.6, step=0.05)
    bear_scale = st.sidebar.slider("BEAR 仓位上限", 0.0, 0.4, 0.2, step=0.05)

    st.sidebar.subheader("波动率目标")
    target_vol = st.sidebar.slider("目标年化波动率", 0.05, 0.30, 0.15, step=0.01)
    vol_window = st.sidebar.slider("波动率计算窗口（天）", 10, 60, 20, step=5)

    st.sidebar.subheader("回测")
    top_n = st.sidebar.slider("选股数量 Top-N", 5, 50, 20, step=5)
    train_window = st.sidebar.slider("训练窗口（天）", 120, 480, 240, step=20)
    commission = st.sidebar.number_input("手续费率", value=0.0003, format="%.4f")

    return Config(
        data=DataConfig(),
        factor=FactorConfig(),
        backtest=BacktestConfig(
            top_n=top_n,
            train_window=train_window,
            commission_rate=commission,
        ),
        ml=MLConfig(),
        regime=RegimeConfig(
            ma_window=ma_window,
            bull_scale=bull_scale,
            range_scale=range_scale,
            bear_scale=bear_scale,
        ),
        advisor=AdvisorConfig(
            target_vol=target_vol,
            vol_window=vol_window,
        ),
    )


@st.cache_data
def load_close() -> pd.DataFrame:
    """加载收盘价宽表（缓存，只加载一次）"""
    frames = {}
    for f in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f, usecols=["trade_date", "close"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        frames[f.stem] = df.set_index("trade_date").sort_index()["close"]
    return pd.DataFrame(frames)
