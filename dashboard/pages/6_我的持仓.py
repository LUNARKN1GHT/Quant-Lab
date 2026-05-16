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
from quant.factor.bollinger import bollinger_position
from quant.factor.ma_bias import ma_bias
from quant.factor.momentum import momentum
from quant.factor.rsi import rsi
from quant.fund.advisor_signal import fund_position_advice, latest_signal
from quant.fund.dca import (
    PERIOD_DAYS,
    PERIOD_LABELS,
    load_dca_plans,
    next_trading_day,
    save_dca_plans,
)
from quant.fund.ledger import (
    compute_holdings,
    load_transactions,
    next_id,
    save_transactions,
    transaction_returns,
)
from quant.fund.portfolio import load_nav_matrix
from quant.fund.portfolio_opt import (
    current_weights,
    estimate_mu_cov,
    reconcile,
    run_all_methods,
)
from quant.fund.watchlist import load_watchlist, save_watchlist

st.set_page_config(page_title="我的持仓", layout="wide")
sidebar_config()
st.title("💼 我的基金持仓")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "quant.duckdb"

# ── Session state 初始化 ──────────────────────────────────────────────────────
if "txns" not in st.session_state:
    st.session_state.txns = load_transactions()
txns = st.session_state.txns
if not txns.empty and "date" in txns.columns:
    txns["date"] = pd.to_datetime(txns["date"])
st.session_state.txns = txns

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()
wl = st.session_state.watchlist
if not wl.empty and "added_date" in wl.columns:
    wl["added_date"] = pd.to_datetime(wl["added_date"])
st.session_state.watchlist = wl

if "dca_plans" not in st.session_state:
    st.session_state.dca_plans = load_dca_plans()
dca_plans = st.session_state.dca_plans


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


@st.cache_data(ttl=3600, show_spinner="计算市场信号...")
def get_advisor_signal():
    from dashboard.shared import load_close
    from quant.config import Config

    cfg = Config()
    try:
        return latest_signal(cfg=cfg, close=load_close())
    except Exception as e:
        return {"error": str(e)}


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
            all_symbols = list(
                set(symbols) | set(wl["symbol"].tolist() if not wl.empty else [])
            )
            for sym in all_symbols:
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
holdings = compute_holdings(txns)


# ════════════════════════════════════════════════════════════════════════════
tab_overview, tab_chart, tab_txn, tab_ret, tab_watch, tab_dca, tab_advice = st.tabs(
    [
        "📊 持仓总览",
        "📈 净值走势",
        "📝 交易记录",
        "💹 收益分析",
        "📋 自选追踪",
        "🔁 定投计划",
        "📡 投资建议",
    ]
)


# ── Tab 1：持仓总览 ────────────────────────────────────────────────────────────
with tab_overview:
    if holdings.empty:
        st.info("暂无持仓，请在「交易记录」页添加买入记录。")
    else:
        # 合并最新净值
        rows = []
        for _, h in holdings.iterrows():  # type: ignore
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

        # --- 基金收益相关性热力图 ----------
        if len(sel_sym) >= 2:
            st.subheader("持仓基金收益相关性")
            ret_mat = nav[sel_sym].pct_change().dropna()
            corr = ret_mat.corr()
            labels = [names.get(s, s) for s in corr.columns]
            fig_corr = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=labels,
                    y=labels,
                    colorscale="RdBu",
                    zmin=-1,
                    zmax=1,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    colorbar=dict(title="相关系数"),
                )
            )
            fig_corr.update_layout(height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig_corr, width="stretch")


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

# ── Tab 5：自选追踪 ────────────────────────────────────────────────────────────
with tab_watch:
    st.subheader("添加自选基金")
    with st.form("add_watch", clear_on_submit=True):
        wc1, wc2 = st.columns(2)
        w_sym = wc1.text_input("基金代码", placeholder="009610")
        w_name = wc2.text_input("基金名称", placeholder="xxx基金")
        w_note = st.text_input("备注（可选）", placeholder="关注原因...")
        if st.form_submit_button("➕ 加入自选", type="primary"):
            if not w_sym or not w_name:
                st.error("代码和名称不能为空")
            elif not wl.empty and w_sym in wl["symbol"].values:
                st.warning(f"{w_sym} 已在自选列表中")
            else:
                new_wl = pd.DataFrame(
                    [
                        {
                            "symbol": w_sym.strip(),
                            "name": w_name.strip(),
                            "added_date": pd.Timestamp.today(),
                            "note": w_note,
                        }
                    ]
                )
                st.session_state.watchlist = pd.concat([wl, new_wl], ignore_index=True)
                save_watchlist(st.session_state.watchlist)
                st.success(f"已添加 {w_name} 到自选")
                st.cache_data.clear()
                st.rerun()

    st.divider()

    wl = st.session_state.watchlist
    if wl.empty:
        st.info("自选列表为空，请添加感兴趣的基金。")
    else:
        # ── 自选列表 + 删除 ──
        st.subheader("自选列表")
        wl_display = wl.copy()
        wl_display["added_date"] = [
            v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)[:10]
            for v in wl_display["added_date"]
        ]
        wl_display = wl_display.rename(
            columns={
                "symbol": "代码",
                "name": "名称",
                "added_date": "加入日期",
                "note": "备注",
            }
        )
        st.dataframe(wl_display, hide_index=True, width="stretch")

        del_sym = st.selectbox(
            "删除自选",
            options=wl["symbol"].tolist(),
            format_func=lambda s: f"{s} {wl.set_index('symbol').loc[s, 'name']}",
        )
        if st.button("🗑️ 移出自选", type="secondary"):
            st.session_state.watchlist = wl[wl["symbol"] != del_sym].reset_index(
                drop=True
            )
            save_watchlist(st.session_state.watchlist)
            st.rerun()

        st.divider()

        # ── 净值走势对比 ──
        st.subheader("净值走势")
        wl_syms = wl["symbol"].tolist()
        wl_names = wl.set_index("symbol")["name"].to_dict()
        wl_nav = get_nav(tuple(sorted(wl_syms)))

        if wl_nav.empty:
            st.info("请点击「刷新净值数据」按钮下载自选基金净值。")
        else:
            wl_period = st.selectbox(
                "时间范围",
                ["成立来", "近3年", "近1年", "近6月", "近3月"],
                key="wl_period",
            )
            wl_norm = st.checkbox("归一化", value=True, key="wl_norm")
            n_map = {
                "近3月": 63,
                "近6月": 126,
                "近1年": 252,
                "近3年": 756,
                "成立来": 99999,
            }

            fig_wl = go.Figure()
            for sym in wl_syms:
                if sym not in wl_nav.columns:
                    continue
                s = wl_nav[sym].dropna().tail(n_map[wl_period])
                y = s / s.iloc[0] if wl_norm else s
                fig_wl.add_trace(
                    go.Scatter(
                        x=s.index,
                        y=y.values,
                        name=wl_names.get(sym, sym),
                        mode="lines",
                        line=dict(width=2),
                    )
                )
            fig_wl.add_hline(y=1 if wl_norm else 0, line_dash="dash", line_color="gray")
            fig_wl.update_layout(
                hovermode="x unified",
                yaxis_title="归一化净值" if wl_norm else "单位净值",
            )
            st.plotly_chart(fig_wl, width="stretch")

            # ── 因子信号扫描 ──
            st.subheader("因子信号扫描")
            st.caption("基于 NAV 序列的技术面信号，仅供参考")

            signal_rows = []
            for sym in wl_syms:
                if sym not in wl_nav.columns:
                    continue
                s = wl_nav[sym].dropna()
                if len(s) < 30:
                    continue
                mom_val = momentum(s, 20).iloc[-1]
                rsi_val = rsi(s, 14).iloc[-1]
                bias_val = ma_bias(s, 20).iloc[-1]
                bb_val = bollinger_position(s, 20).iloc[-1]

                score = sum(
                    [
                        mom_val < 0,
                        rsi_val < 40,
                        bias_val < -0.03,
                        bb_val < 0.3,
                    ]
                )
                if score >= 3:
                    sig = "🟢 关注买入"
                elif score >= 2:
                    sig = "🟡 继续观察"
                else:
                    sig = "⚪ 暂无信号"

                signal_rows.append(
                    {
                        "名称": wl_names.get(sym, sym),
                        "代码": sym,
                        "动量(20日)": mom_val,
                        "RSI(14)": rsi_val,
                        "均线偏离": bias_val,
                        "布林位置": bb_val,
                        "综合评分": f"{score}/4",
                        "信号": sig,
                    }
                )

            if signal_rows:
                sig_df = pd.DataFrame(signal_rows)
                st.dataframe(
                    sig_df.style.format(
                        {
                            "动量(20日)": "{:+.2%}",
                            "RSI(14)": "{:.1f}",
                            "均线偏离": "{:+.2%}",
                            "布林位置": "{:.2f}",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info("净值数据不足（需 ≥30 个交易日）")

# ── Tab 6：定投计划 ────────────────────────────────────────────────────────────
with tab_dca:
    # ── Regime 定投指导 ───────────────────────────────────────────────────────
    _dca_signal = get_advisor_signal()
    if "error" not in _dca_signal:
        _regime_dca_hint = {
            "BULL": (
                "🟡 当前市场处于上行趋势（BULL）。定投性价比偏低，"
                "可维持计划金额，不建议额外加仓。"
            ),
            "RANGE": (
                "🟢 当前市场处于震荡区间（RANGE）。定投是最合适的入场方式，"
                "可按计划执行，波动中均摊成本。"
            ),
            "BEAR": (
                "🔵 当前市场处于下行阶段（BEAR）。若相信长期价值，"
                "此时是定投的高性价比窗口，可考虑适当增加定投金额或频率。"
            ),
        }
        st.info(_regime_dca_hint.get(_dca_signal["regime"], ""))

    dca_plans = st.session_state.dca_plans

    # ── 添加定投计划 ──
    st.subheader("管理定投计划")
    with st.form("add_dca", clear_on_submit=True):
        dc1, dc2, dc3, dc4 = st.columns(4)
        dca_sym = dc1.text_input("基金代码", placeholder="009610")
        dca_name = dc2.text_input("基金名称", placeholder="xxx基金")
        dca_amount = dc3.number_input(
            "每期金额（元）", min_value=1.0, value=500.0, step=100.0
        )
        dca_period = dc4.selectbox(
            "定投周期",
            options=list(PERIOD_LABELS.keys()),
            index=3,  # 默认 monthly
            format_func=lambda x: PERIOD_LABELS[x],
        )
        dca_note = st.text_input("备注（可选）", placeholder="月定投...")
        if st.form_submit_button("➕ 保存计划", type="primary"):
            if not dca_sym or not dca_name:
                st.error("代码和名称不能为空")
            elif not dca_plans.empty and dca_sym in dca_plans["symbol"].values:
                st.warning(f"{dca_sym} 已有定投计划，请先删除再新增")
            else:
                new_plan = pd.DataFrame(
                    [
                        {
                            "symbol": dca_sym.strip(),
                            "name": dca_name.strip(),
                            "amount": dca_amount,
                            "period": dca_period,
                            "note": dca_note,
                        }
                    ]
                )
                st.session_state.dca_plans = pd.concat(
                    [dca_plans, new_plan], ignore_index=True
                )
                save_dca_plans(st.session_state.dca_plans)
                st.success(
                    f"已保存：{dca_name}"
                    f" {PERIOD_LABELS[dca_period]}定投 ¥{dca_amount:.0f}"
                )
                st.rerun()

    # 展示 & 删除计划
    dca_plans = st.session_state.dca_plans
    if not dca_plans.empty:
        display_plans = dca_plans.copy()
        if "period" in display_plans.columns:
            display_plans["period"] = display_plans["period"].map(PERIOD_LABELS)
        st.dataframe(
            display_plans.rename(
                columns={
                    "symbol": "代码",
                    "name": "名称",
                    "amount": "每期金额(元)",
                    "period": "周期",
                    "note": "备注",
                }
            ),
            hide_index=True,
            width="stretch",
        )
        del_plan_sym = st.selectbox(
            "删除计划",
            options=dca_plans["symbol"].tolist(),
            format_func=lambda s: f"{s} {dca_plans.set_index('symbol').loc[s, 'name']}",
            key="del_dca_sym",
        )
        if st.button("🗑️ 删除该计划", type="secondary"):
            st.session_state.dca_plans = dca_plans[
                dca_plans["symbol"] != del_plan_sym
            ].reset_index(drop=True)
            save_dca_plans(st.session_state.dca_plans)
            st.rerun()

    st.divider()

    # ── 一键执行定投 ──
    st.subheader("执行定投")
    if dca_plans.empty:
        st.info("请先添加定投计划。")
    else:
        exec_plan_sym = st.selectbox(
            "选择定投计划",
            options=dca_plans["symbol"].tolist(),
            format_func=lambda s: (
                f"{s}  {dca_plans.set_index('symbol').loc[s, 'name']}"
                f"  ¥{dca_plans.set_index('symbol').loc[s, 'amount']:.0f}/期"
            ),
            key="exec_dca_sym",
        )
        plan_row = dca_plans[dca_plans["symbol"] == exec_plan_sym].iloc[0]
        plan_amount = float(plan_row["amount"])
        plan_name = str(plan_row["name"])
        plan_period = str(plan_row.get("period", "monthly"))

        # ── 上次/下次定投提示 ──
        sym_txns = st.session_state.txns[
            st.session_state.txns["symbol"] == exec_plan_sym
        ].sort_values("date")
        if not sym_txns.empty:
            last_date = sym_txns["date"].iloc[-1]
            period_days = PERIOD_DAYS.get(plan_period, 30)
            raw_next = last_date + pd.Timedelta(days=period_days)
            if not nav.empty and exec_plan_sym in nav.columns:
                adjusted = next_trading_day(raw_next, nav[exec_plan_sym])
                next_due = adjusted if adjusted is not None else raw_next
            else:
                next_due = raw_next
            today = pd.Timestamp.today().normalize()
            overdue = today >= next_due
            label = "🔴 已到期" if overdue else "🟢 未到期"
            st.caption(
                f"上次定投：{last_date.date()}　"
                f"周期：{PERIOD_LABELS.get(plan_period, '每月')}　"
                f"下次应投：{next_due.date()}（已跳过节假日）　{label}"
            )

        col_date, col_nav, col_amount = st.columns(3)
        exec_date = col_date.date_input("定投日期", key="exec_dca_date")
        exec_note = st.text_input(
            "备注", value=str(plan_row.get("note", "定投")), key="exec_dca_note"
        )

        # 自动从数据库读取当日净值（若当日无净值则取最近交易日）
        auto_nav: float | None = None
        if not nav.empty and exec_plan_sym in nav.columns:
            ts = pd.Timestamp(exec_date)
            series = nav[exec_plan_sym].dropna()
            if ts in series.index:
                auto_nav = float(series[ts])
            elif ts <= series.index[-1]:
                val = series.asof(ts)
                if pd.notna(val):  # type: ignore
                    auto_nav = float(val)  # type: ignore[arg-type]

        exec_nav = col_nav.number_input(
            "成交净值（自动填充，可修改）",
            min_value=0.0001,
            value=auto_nav if auto_nav else 1.0,
            step=0.0001,
            format="%.4f",
            key="exec_dca_nav",
        )
        exec_amount = col_amount.number_input(
            "本期金额（元）",
            min_value=1.0,
            value=plan_amount,
            step=100.0,
            key="exec_dca_amount",
        )

        exec_shares = exec_amount / exec_nav
        st.info(
            f"将买入 **{plan_name}**：¥{exec_amount:.2f} ÷ {exec_nav:.4f}"
            f" = **{exec_shares:.2f} 份**"
        )

        if st.button("✅ 确认执行定投", type="primary"):
            new_row = pd.DataFrame(
                [
                    {
                        "id": next_id(st.session_state.txns),
                        "symbol": exec_plan_sym,
                        "name": plan_name,
                        "date": pd.Timestamp(exec_date),
                        "type": "buy",
                        "shares": round(exec_shares, 2),
                        "nav": exec_nav,
                        "note": exec_note or "定投",
                    }
                ]
            )
            st.session_state.txns = (
                pd.concat([st.session_state.txns, new_row], ignore_index=True)
                .sort_values("date")
                .reset_index(drop=True)
            )
            save_transactions(st.session_state.txns)
            st.success(
                f"✅ 定投成功：{plan_name}  买入 {exec_shares:.2f} 份 @ {exec_nav:.4f}"
            )
            st.cache_data.clear()
            st.rerun()


# ── Tab 7：投资建议 ────────────────────────────────────────────────────────────
with tab_advice:
    if nav.empty or holdings.empty:
        st.info("需要持仓和净值数据才能生成建议。")
    else:
        # ── 全局 Advisor 信号 ─────────────────────────────────────────────────
        st.subheader("📡 市场信号与建议仓位")
        st.caption("基于沪深300趋势 × 波动率目标 × 宏观景气的三层仓位模型")

        signal = get_advisor_signal()

        if "error" in signal:
            st.warning(f"信号获取失败：{signal['error']}")
        else:
            sig_c1, sig_c2, sig_c3, sig_c4 = st.columns(4)
            sig_c1.metric(
                "市场状态",
                f"{signal['regime_emoji']} {signal['regime']}",
                help=signal["regime_label"],
            )
            sig_c2.metric(
                "建议权益仓位",
                f"{signal['position']:.1%}",
                help="= Regime × 波动率 × 宏观三层乘数综合结果",
            )
            sig_c3.metric(
                "信号日期",
                str(signal["date"].date()),  # type: ignore
            )
            sig_c4.metric(
                "宏观信号",
                f"{signal['macro_signal']:.0%}",
                help="宏观景气 z-score 归一化到 [0,1]；无宏观数据时为 0.5（中性）",
            )

            with st.expander("信号分解明细"):
                detail_df = pd.DataFrame(
                    [
                        {
                            "层级": "Regime（市场环境）",
                            "值": signal["regime_label"],
                            "系数": f"{signal['regime_signal']:.0%}",
                        },
                        {
                            "层级": "波动率目标法",
                            "值": "realized_vol → target_vol",
                            "系数": f"{signal['vol_signal']:.0%}",
                        },
                        {
                            "层级": "宏观景气信号",
                            "值": "PMI / 利率 / M2 综合",
                            "系数": f"{signal['macro_signal']:.0%}",
                        },
                        {
                            "层级": "最终建议仓位",
                            "值": "三层乘积（截断至[min, max]）",
                            "系数": f"{signal['position']:.2f}",
                        },
                    ]
                )
                st.dataframe(detail_df, hide_index=True, width="stretch")

            st.divider()

            # ── 持仓操作建议 ──────────────────────────────────────────────────
            st.subheader("持仓操作建议")
            st.caption(
                "当前仓位与 Advisor 建议仓位的偏差，超过总资金 2% 时触发操作提示"
            )

            # 构建含 mkt 的持仓 DataFrame
            advice_rows = []
            for _, h in holdings.iterrows():  # type: ignore
                sym = h["symbol"]
                if sym not in nav.columns:
                    continue
                latest_nav_val = nav[sym].dropna().iloc[-1]
                mkt = h["shares"] * latest_nav_val
                advice_rows.append(
                    {
                        "symbol": sym,
                        "name": h["name"],
                        "mkt": mkt,
                        "fund_type": "equity",  # 默认权益型
                    }
                )

            if advice_rows:
                advice_holdings = pd.DataFrame(advice_rows)
                total_mkt = advice_holdings["mkt"].sum()
                # 总资金 = 持仓市值（暂不含现金，后续可扩展）
                advice_df = fund_position_advice(signal, advice_holdings, total_mkt)
                st.dataframe(
                    advice_df.style.format(
                        {
                            "当前仓位": "{:.1%}",
                            "建议仓位": "{:.1%}",
                            "偏差金额": "{:+.2f}",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "⚠️ 总资金当前仅含持仓市值，未计入现金。"
                    "如需更精确的偏差计算，请在此处手动输入总资金。"
                )

            st.divider()

        # --- 持仓组合优化 ----------
        st.subheader("🎯 持仓组合优化")
        st.caption("基于持仓基金净值历史估计 μ/Σ，对比三种最优配置方法与当前实盘权重")

        if nav.empty or holdings.empty:
            st.info("需要持仓和净值数据才能进行组合优化。")
        else:
            held_symbols = holdings["symbol"].tolist()
            nav_held = nav[[s for s in held_symbols if s in nav.columns]].dropna(
                how="all"
            )

            if nav_held.shape[1] < 2:
                st.warning("至少需要 2 只基金才能进行组合优化。")
            else:
                opt_c1, opt_c2, opt_c3 = st.columns(3)
                opt_lookback = opt_c1.selectbox(
                    "估计窗口（交易日）", [60, 120, 250], index=1, key="opt_lookback"
                )
                opt_ra = opt_c2.slider(
                    "风险厌恶系数（MVO）", 0.5, 5.0, 2.0, 0.5, key="opt_ra"
                )
                opt_w_max = opt_c3.slider(
                    "单基金权重上限", 0.1, 1.0, 0.5, 0.05, key="opt_w_max"
                )

                if st.button("🚀 运行持仓优化", type="primary", key="btn_holdings_opt"):
                    try:
                        mu, cov, assets = estimate_mu_cov(nav_held, int(opt_lookback))
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        cur_w, total_mkt = current_weights(holdings, nav)
                        results = run_all_methods(
                            mu, cov, risk_aversion=opt_ra, w_max=opt_w_max
                        )
                        st.session_state["holdings_opt_result"] = {
                            "assets": assets,
                            "current_w": cur_w,
                            "total_mkt": float(total_mkt),
                            "results": results,
                            "names": names,
                        }

                if "holdings_opt_result" in st.session_state:
                    r = st.session_state["holdings_opt_result"]
                    assets = r["assets"]
                    current_w = r["current_w"]
                    total_mkt = r["total_mkt"]
                    results = r["results"]
                    names_map = r["names"]

                    # 统计指标对比
                    st.markdown("##### 组合统计对比")
                    stats_df = pd.DataFrame(
                        {
                            "方法": list(results.keys()),
                            "预期年化收益": [s["return"] for _, s in results.values()],
                            "年化波动率": [
                                s["volatility"] for _, s in results.values()
                            ],
                            "夏普比率": [s["sharpe"] for _, s in results.values()],
                        }
                    ).set_index("方法")
                    st.dataframe(
                        stats_df.style.format(
                            {
                                "预期年化收益": "{:.2%}",
                                "年化波动率": "{:.2%}",
                                "夏普比率": "{:.2f}",
                            }
                        ),
                        width="stretch",
                    )

                    # 权重对账表（默认按风险平价对账，可切换）
                    st.markdown("##### 偏差与调仓建议")
                    chosen = st.radio(
                        "对账参考方法",
                        list(results.keys()),
                        index=2,
                        horizontal=True,
                        key="opt_chosen_method",
                    )
                    target_w = results[chosen][0]
                    recon_df = reconcile(
                        assets, target_w, current_w, total_mkt, name_map=names_map
                    )
                    st.dataframe(
                        recon_df.style.format(
                            {
                                "当前权重": "{:.1%}",
                                "建议权重": "{:.1%}",
                                "权重偏差": "{:+.1%}",
                                "调仓金额": "¥{:+,.2f}",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    # 三方法权重并排对比图
                    st.markdown("##### 权重分布对比")
                    fig_compare = go.Figure()
                    asset_labels = [names_map.get(s, s) for s in assets]
                    fig_compare.add_trace(
                        go.Bar(
                            name="当前实盘",
                            x=asset_labels,
                            y=[current_w.get(s, 0) for s in assets],
                        )
                    )
                    for label, (w, _) in results.items():  # type: ignore
                        fig_compare.add_trace(
                            go.Bar(name=label, x=asset_labels, y=list(w))
                        )
                    fig_compare.update_layout(
                        barmode="group",
                        yaxis_tickformat=".0%",
                        height=380,
                        margin=dict(t=10, b=10, l=0, r=0),
                        xaxis_tickangle=-30,
                    )
                    st.plotly_chart(fig_compare, width="stretch")

                st.divider()

        # ── 因子信号（持仓基金）──────────────────────────────────────────────
        st.subheader("持仓基金因子信号")
        st.caption("基于持仓基金 NAV 走势的技术面扫描")

        holding_signals = []
        for _, h in holdings.iterrows():  # type: ignore
            sym = h["symbol"]
            if sym not in nav.columns:
                continue
            s = nav[sym].dropna()
            if len(s) < 30:
                continue
            mom_val = momentum(s, 20).iloc[-1]
            rsi_val = rsi(s, 14).iloc[-1]
            bias_val = ma_bias(s, 20).iloc[-1]
            bb_val = bollinger_position(s, 20).iloc[-1]

            score = sum(
                [
                    mom_val < 0,
                    rsi_val < 40,
                    bias_val < -0.03,
                    bb_val < 0.3,
                ]
            )
            if score >= 3:
                sig = "🟢 可考虑加仓"
            elif score == 2:
                sig = "🟡 持有观察"
            elif score <= 1:
                sig = "🔴 注意风险"
            else:
                sig = "⚪ 持有"

            holding_signals.append(
                {
                    "名称": h["name"],
                    "代码": sym,
                    "动量(20日)": mom_val,
                    "RSI(14)": rsi_val,
                    "均线偏离": bias_val,
                    "布林位置": bb_val,
                    "评分": f"{score}/4",
                    "建议": sig,
                }
            )

        if holding_signals:
            hs_df = pd.DataFrame(holding_signals)
            st.dataframe(
                hs_df.style.format(
                    {
                        "动量(20日)": "{:+.2%}",
                        "RSI(14)": "{:.1f}",
                        "均线偏离": "{:+.2%}",
                        "布林位置": "{:.2f}",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

        st.divider()

        # ── 风格归因 ──────────────────────────────────────────────────────────
        st.subheader("📐 持仓基金风格归因")
        st.caption(
            "将基金 NAV 收益回归到沪深300 / 中证500，分解 Beta 暴露与超额收益（Alpha）"
        )

        if st.button("📥 加载基准数据并计算风格归因", key="btn_style"):
            from quant.data.benchmark import load_benchmarks
            from quant.fund.style import fund_style_report

            try:
                benchmarks = load_benchmarks()
                style_rows = []
                for _, h in holdings.iterrows():  # type: ignore
                    sym = h["symbol"]
                    if sym not in nav.columns:
                        continue
                    row = fund_style_report(
                        nav[sym].dropna(), benchmarks, name=h["name"]
                    )
                    style_rows.append(row)
                st.session_state["style_report"] = pd.DataFrame(style_rows)
            except Exception as e:
                st.error(f"加载失败：{e}")

        if "style_report" in st.session_state:
            sr = st.session_state["style_report"]
            if "error" in sr.columns:
                st.warning("部分基金数据不足，已跳过")
                sr = sr[sr.get("error", pd.Series(dtype=str)).isna()]
            numeric_cols = ["CSI300 Beta", "CSI500 Beta", "Alpha（年化）", "R²"]
            existing = [c for c in numeric_cols if c in sr.columns]
            fmt = {
                "CSI300 Beta": "{:.3f}",
                "CSI500 Beta": "{:.3f}",
                "Alpha（年化）": "{:.2%}",
                "R²": "{:.3f}",
            }
            st.dataframe(
                sr[["name"] + existing].style.format(
                    {k: v for k, v in fmt.items() if k in existing}
                ),
                hide_index=True,
                width="stretch",
            )

        st.divider()

        # ── 风险预警 ──────────────────────────────────────────────────────────
        st.subheader("风险预警")
        warnings_found = False

        pos_rows = []
        for _, h in holdings.iterrows():  # type: ignore
            sym = h["symbol"]
            if sym not in nav.columns:
                continue
            latest = nav[sym].dropna().iloc[-1]
            cost_val = h["shares"] * h["avg_cost_nav"]
            mkt_val = h["shares"] * latest
            ret = latest / h["avg_cost_nav"] - 1
            pos_rows.append(
                {
                    "name": h["name"],
                    "symbol": sym,
                    "cost": cost_val,
                    "mkt": mkt_val,
                    "ret": ret,
                }
            )

        if pos_rows:
            pos_df = pd.DataFrame(pos_rows)
            total_mkt = pos_df["mkt"].sum()

            for _, p in pos_df.iterrows():  # type: ignore
                weight = p["mkt"] / total_mkt if total_mkt > 0 else 0

                if weight > 0.5:
                    st.warning(
                        f"⚠️ **{p['name']}** 仓位占比 {weight:.1%}，集中度过高，建议分散"
                    )
                    warnings_found = True

                if p["ret"] < -0.10:
                    st.error(
                        f"🔴 **{p['name']}** 浮亏 {p['ret']:.1%}，"
                        + "已超 -10%，建议复盘止损策略"
                    )
                    warnings_found = True
                elif p["ret"] < -0.05:
                    st.warning(f"⚠️ **{p['name']}** 浮亏 {p['ret']:.1%}，请持续关注")
                    warnings_found = True

            if not txns.empty:
                buys = txns[txns["type"] == "buy"]
                today = pd.Timestamp.today()
                for _, b in buys.iterrows():
                    hold_days = (today - b["date"]).days
                    sym = b["symbol"]
                    if sym not in nav.columns:
                        continue
                    latest_nav_val = nav[sym].dropna().iloc[-1]
                    ret_b = latest_nav_val / b["nav"] - 1
                    if hold_days > 180 and ret_b < 0:
                        st.warning(
                            f"⚠️ **{b['name']}** 已持有 {hold_days} 天，"
                            f"仍亏损 {ret_b:.1%}，建议重新评估"
                        )
                        warnings_found = True

        if not warnings_found:
            st.success("✅ 当前持仓无明显风险信号")
