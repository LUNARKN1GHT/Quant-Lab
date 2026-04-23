import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import datetime
import time

import pandas as pd
import streamlit as st

from dashboard.shared import load_close, sidebar_config

st.set_page_config(page_title="数据管理", layout="wide")
sidebar_config()

st.title("🗄️ 数据管理")

DATA_DIR = Path(__file__).parent.parent.parent / "data/csi300"


def get_data_status() -> pd.DataFrame:
    rows = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(f, usecols=["trade_date"])
            rows.append(
                {
                    "股票": f.stem,
                    "最早日期": str(df["trade_date"].min()),
                    "最新日期": str(df["trade_date"].max()),
                    "记录数": len(df),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


# 数据概况
st.subheader("当前数据概况")
status = get_data_status()

if not status.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("股票数量", len(status))
    col2.metric("最早日期", status["最早日期"].min())
    col3.metric("最新日期", status["最新日期"].max())
    col4.metric("总记录数", f"{status['记录数'].sum():,}")

    with st.expander("查看各股票详情"):
        st.dataframe(status, use_container_width=True)
else:
    st.warning("data/csi300/ 目录下暂无数据")

st.divider()

# 增量更新
st.subheader("增量更新数据")
st.caption("只下载最新日期之后的数据，不重复拉取历史")

end_date = st.date_input(
    "更新至",
    value=datetime.date.today(),
    max_value=datetime.date.today(),
)
end_str = end_date.strftime("%Y%m%d")

update_btn = st.button("🔄 开始增量更新", type="primary")

if update_btn:
    from scripts.download_data import get_csi300_symbols, update_symbol

    try:
        with st.spinner("获取沪深300成分股列表…"):
            symbols = get_csi300_symbols()
    except Exception as e:
        st.error(f"获取成分股失败：{e}")
        st.stop()

    st.info(f"共 {len(symbols)} 只股票，更新至 {end_str}")

    progress = st.progress(0, text="准备中…")
    log = st.empty()

    counts = {"new": 0, "updated": 0, "skip": 0, "fail": 0}
    fail_list = []

    for i, symbol in enumerate(symbols, 1):
        result = update_symbol(symbol, end_str)
        counts[result] += 1
        if result == "fail":
            fail_list.append(symbol)

        progress.progress(
            i / len(symbols), text=f"[{i}/{len(symbols)}] {symbol} → {result}"
        )
        time.sleep(1.1)

    progress.empty()

    # 结果汇总
    st.success(
        f"更新完成！新增 **{counts['new']}** 只 · "
        f"更新 **{counts['updated']}** 只 · "
        f"已最新 **{counts['skip']}** 只 · "
        f"失败 **{counts['fail']}** 只"
    )
    if fail_list:
        st.warning(f"失败股票：{', '.join(fail_list)}")

    # 清除缓存，让其他页面重新加载数据
    load_close.clear()
    st.info("数据缓存已刷新，其他页面下次访问将加载最新数据。")
