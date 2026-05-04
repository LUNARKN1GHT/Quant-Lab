"""我的基金持仓看板 — 基于交易流水"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import sidebar_config
from quant.fund.ledger import (
    compute_holdings,
    load_transactions,
    next_id,
    save_transactions,
    transaction_returns,
)
from quant.fund.portfolio import load_nav_matrix

st.set_page_config(page_title="我的持仓", layout="wide")
sidebar_config()
st.title("💼 我的基金持仓")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "quant.duckdb"


# ── 交易流水（session state 做缓存，避免每次操作重读）────────────────────────
if "txns" not in st.session_state:
    st.session_state.txns = load_transactions()

# 每次脚本运行都强制恢复 date 列 dtype（防止 Arrow 序列化破坏）
txns = st.session_state.txns
if not txns.empty and "date" in txns.columns:
    txns["date"] = pd.to_datetime(txns["date"])
st.session_state.txns = txns


def reload():
    st.session_state.txns = load_transactions()


# ── 读取净值数据 ───────────────────────────────────────────────────────────────
symbols = txns["symbol"].unique().tolist() if not txns.empty else []
names = (
    txns.drop_duplicates("symbol").set_index("symbol")["name"].to_dict()
    if not txns.empty
    else {}
)


@st.cache_data(ttl=3600, show_spinner="加载净值...")
def get_nav(syms):
    if not syms:
        return pd.DataFrame()
    con = duckdb.connect(str(DB_PATH))
    nav = load_nav_matrix(con, syms)
    con.close()
    return nav


col_refresh, _ = st.columns([1, 4])
with col_refresh:
    if st.button("🔄 刷新净值数据", help="从东方财富下载最新净值并更新数据库"):
        with st.spinner("下载净值中..."):
            import akshare as ak
            import duckdb

            con = duckdb.connect(str(DB_PATH))
            con.execute("""
                CREATE TABLE IF NOT EXISTS fund_nav (
                    symbol VARCHAR, date DATE, nav DOUBLE, daily_pct DOUBLE,
                    PRIMARY KEY (symbol, date)
                )
            """)
            for sym in symbols:
                try:
                    df_nav = ak.fund_open_fund_info_em(
                        symbol=sym, indicator="单位净值走势", period="成立来"
                    ).rename(
                        columns={
                            "净值日期": "date",
                            "单位净值": "nav",
                            "日增长率": "daily_pct",
                        }
                    )
                    df_nav["date"] = pd.to_datetime(df_nav["date"])
                    df_nav["symbol"] = sym
                    df_nav["nav"] = pd.to_numeric(df_nav["nav"], errors="coerce")
                    df_nav["daily_pct"] = pd.to_numeric(
                        df_nav["daily_pct"], errors="coerce"
                    )
                    df_nav = df_nav[["symbol", "date", "nav", "daily_pct"]].dropna(
                        subset=["nav"]
                    )
                    con.execute(f"DELETE FROM fund_nav WHERE symbol = '{sym}'")
                    con.execute("INSERT INTO fund_nav SELECT * FROM df_nav")
                    st.toast(f"✅ {sym} 净值已更新")
                except Exception as e:
                    st.toast(f"❌ {sym} 下载失败: {e}")
            con.close()
            st.cache_data.clear()
            st.rerun()


nav = get_nav(tuple(sorted(symbols)))


# ════════════════════════════════════════════════════════════════════════════
tab_overview, tab_chart, tab_txn, tab_ret = st.tabs(
    ["📊 持仓总览", "📈 净值走势", "📝 交易记录", "💹 收益分析"]
)


# ── Tab 1：持仓总览 ────────────────────────────────────────────────────────────
with tab_overview:
    holdings = compute_holdings(txns)

    if holdings.empty:
        st.info("暂无持仓，请在「交易记录」页添加买入记录。")
    else:
        # 合并最新净值
        rows = []
        for _, h in holdings.iterrows():
            sym = h["symbol"]
            latest = (
                nav[sym].dropna().iloc[-1]
                if (not nav.empty and sym in nav.columns)
                else None
            )
            if latest is None:
                continue
            pnl = h["shares"] * (latest - h["avg_cost_nav"])
            rows.append(
                {
                    "名称": h["name"],
                    "代码": sym,
                    "份额": h["shares"],
                    "成本净值": h["avg_cost_nav"],
                    "最新净值": latest,
                    "成本市值": h["shares"] * h["avg_cost_nav"],
                    "当前市值": h["shares"] * latest,
                    "浮盈亏": pnl,
                    "收益率": latest / h["avg_cost_nav"] - 1,
                }
            )
        df_pos = pd.DataFrame(rows)

        total_cost = df_pos["成本市值"].sum()
        total_mkt = df_pos["当前市值"].sum()
        total_pnl = total_mkt - total_cost

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("持仓基金", len(df_pos))
        c2.metric("总成本", f"¥{total_cost:,.2f}")
        c3.metric("当前市值", f"¥{total_mkt:,.2f}")
        c4.metric(
            "总浮盈亏",
            f"¥{total_pnl:+,.2f}",
            f"{total_pnl / total_cost:+.2%}" if total_cost else "",
        )

        st.dataframe(
            df_pos.style.format(
                {
                    "份额": "{:,.2f}",
                    "成本净值": "{:.4f}",
                    "最新净值": "{:.4f}",
                    "成本市值": "¥{:,.2f}",
                    "当前市值": "¥{:,.2f}",
                    "浮盈亏": "¥{:+,.2f}",
                    "收益率": "{:+.2%}",
                }
            ).map(
                lambda v: (
                    "color:green"
                    if isinstance(v, float) and v > 0
                    else "color:red"
                    if isinstance(v, float) and v < 0
                    else ""
                ),
                subset=["浮盈亏", "收益率"],
            ),
            width="stretch",
            hide_index=True,
        )

        # 仓位饼图
        fig_pie = px.pie(
            df_pos, names="名称", values="当前市值", title="持仓比例（市值）"
        )
        st.plotly_chart(fig_pie, width="stretch")


# ── Tab 2：净值走势 ────────────────────────────────────────────────────────────
with tab_chart:
    if nav.empty:
        st.info("请先运行 scripts/download_fund_nav.py 下载净值数据。")
    else:
        col_sym, col_period, col_norm = st.columns([2, 2, 1])
        sel_sym = col_sym.multiselect(
            "选择基金", symbols, default=symbols, format_func=lambda s: names.get(s, s)
        )
        period = col_period.selectbox(
            "时间范围", ["成立来", "近3年", "近1年", "近6月", "近3月"]
        )
        normalize = col_norm.checkbox("归一化（各自起点=1）", value=True)

        n_map = {"近3月": 63, "近6月": 126, "近1年": 252, "近3年": 756, "成立来": 99999}
        n = n_map[period]

        fig = go.Figure()
        for sym in sel_sym:
            if sym not in nav.columns:
                continue
            s = nav[sym].dropna().tail(n)
            y = s / s.iloc[0] if normalize else s
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=y.values,
                    name=names.get(sym, sym),
                    mode="lines",
                    line=dict(width=2),
                )
            )

            # 买入标记
            buys = txns[(txns["symbol"] == sym) & (txns["type"] == "buy")]
            for _, b in buys.iterrows():
                buy_date = b["date"]
                if buy_date < s.index[0] or buy_date > s.index[-1]:
                    continue
                buy_nav = s.asof(buy_date)
                buy_y = buy_nav / s.iloc[0] if normalize else buy_nav
                fig.add_trace(
                    go.Scatter(
                        x=[buy_date],
                        y=[buy_y],
                        mode="markers+text",
                        marker=dict(symbol="triangle-up", size=14, color="green"),
                        text=[f"买入¥{b['nav']:.4f}"],
                        textposition="top center",
                        showlegend=False,
                        name=f"买入-{sym}",
                    )
                )

            # 卖出标记
            sells = txns[(txns["symbol"] == sym) & (txns["type"] == "sell")]
            for _, sv in sells.iterrows():
                sell_date = sv["date"]
                if sell_date < s.index[0] or sell_date > s.index[-1]:
                    continue
                sell_nav = s.asof(sell_date)
                sell_y = sell_nav / s.iloc[0] if normalize else sell_nav
                fig.add_trace(
                    go.Scatter(
                        x=[sell_date],
                        y=[sell_y],
                        mode="markers+text",
                        marker=dict(symbol="triangle-down", size=14, color="red"),
                        text=[f"卖出¥{sv['nav']:.4f}"],
                        textposition="bottom center",
                        showlegend=False,
                        name=f"卖出-{sym}",
                    )
                )

        fig.add_hline(y=1 if normalize else 0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="净值走势（▲买入 ▼卖出）",
            hovermode="x unified",
            yaxis_title="归一化净值" if normalize else "单位净值",
        )
        st.plotly_chart(fig, width="stretch")


# ── Tab 3：交易记录 ────────────────────────────────────────────────────────────
with tab_txn:
    st.subheader("添加交易")
    with st.form("add_txn", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        sym_input = c1.text_input("基金代码", placeholder="009610")
        name_input = c2.text_input("基金名称", placeholder="xxx基金")
        txn_type = c3.selectbox("类型", ["buy", "sell"])

        c4, c5, c6 = st.columns(3)
        txn_date = c4.date_input("交易日期")
        txn_shares = c5.number_input("份额", min_value=0.01, value=1000.0, step=100.0)
        txn_nav = c6.number_input(
            "成交净值", min_value=0.0001, value=1.0000, step=0.0001, format="%.4f"
        )
        txn_note = st.text_input("备注（可选）", placeholder="定投 / 补仓 / ...")

        submitted = st.form_submit_button("✅ 添加", type="primary")
        if submitted:
            if not sym_input or not name_input:
                st.error("代码和名称不能为空")
            else:
                new_row = pd.DataFrame(
                    [
                        {
                            "id": next_id(txns),
                            "symbol": sym_input.strip(),
                            "name": name_input.strip(),
                            "date": pd.Timestamp(txn_date),
                            "type": txn_type,
                            "shares": txn_shares,
                            "nav": txn_nav,
                            "note": txn_note,
                        }
                    ]
                )
                st.session_state.txns = (
                    pd.concat([txns, new_row], ignore_index=True)
                    .sort_values("date")
                    .reset_index(drop=True)
                )
                save_transactions(st.session_state.txns)
                st.success(
                    f"已添加：{name_input} {txn_type} {txn_shares}份 @ {txn_nav:.4f}"
                )
                st.cache_data.clear()
                st.rerun()

    st.divider()
    st.subheader("全部交易记录")
    if txns.empty:
        st.info("暂无交易记录。")
    else:
        display = txns.copy()
        display["date"] = [
            v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]
            for v in display["date"]
        ]
        display["type"] = display["type"].map({"buy": "🟢 买入", "sell": "🔴 卖出"})
        display = display.rename(
            columns={
                "id": "ID",
                "symbol": "代码",
                "name": "名称",
                "date": "日期",
                "type": "类型",
                "shares": "份额",
                "nav": "成交净值",
                "note": "备注",
            }
        )

        st.dataframe(display, width="stretch", hide_index=True)

        # 删除指定记录
        del_id = st.number_input("输入要删除的交易 ID", min_value=1, step=1, value=1)
        if st.button("🗑️ 删除该记录", type="secondary"):
            st.session_state.txns = txns[txns["id"] != del_id].reset_index(drop=True)
            save_transactions(st.session_state.txns)
            st.success(f"已删除 ID={del_id}")
            st.cache_data.clear()
            st.rerun()


# ── Tab 4：收益分析 ────────────────────────────────────────────────────────────
with tab_ret:
    if nav.empty or txns.empty:
        st.info("需要净值数据和交易记录才能分析收益。")
    else:
        df_ret = transaction_returns(txns, nav)
        if df_ret.empty:
            st.info("暂无买入记录。")
        else:
            st.subheader("每笔买入收益明细")
            st.dataframe(
                df_ret.style.format(
                    {
                        "买入净值": "{:.4f}",
                        "最新净值": "{:.4f}",
                        "收益率": "{:+.2%}",
                        "盈亏金额": "¥{:+,.2f}",
                    }
                ).map(
                    lambda v: (
                        "color:green"
                        if isinstance(v, float) and v > 0
                        else "color:red"
                        if isinstance(v, float) and v < 0
                        else ""
                    ),
                    subset=["收益率", "盈亏金额"],
                ),
                width="stretch",
                hide_index=True,
            )

            # 按基金汇总
            st.subheader("按基金汇总")
            summary = (
                df_ret.groupby("基金")
                .agg(
                    笔数=("交易ID", "count"),
                    总盈亏=("盈亏金额", "sum"),
                    平均收益率=("收益率", "mean"),
                    平均持有天数=("持有天数", "mean"),
                )
                .reset_index()
            )
            st.dataframe(
                summary.style.format(
                    {
                        "总盈亏": "¥{:+,.2f}",
                        "平均收益率": "{:+.2%}",
                        "平均持有天数": "{:.0f}天",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
