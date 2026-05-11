"""将系统 Advisor 输出转换为基金层面的持仓建议

优先使用本地 DuckDB price_daily 表中的成分股数据，
避免网络依赖，同时享有完整的市场宽度信号（280+ 只股票）。
"""

from pathlib import Path

import duckdb
import pandas as pd

from quant.advisor.position import compute_position
from quant.config import Config
from quant.regime.detector import Regime

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "quant.duckdb"

REGIME_LABEL = {
    Regime.BULL.value: ("BULL — 趋势上行", "🟢"),
    Regime.RANGE.value: ("RANGE — 震荡整理", "🟡"),
    Regime.BEAR.value: ("BEAR — 趋势下行", "🔴"),
}

FUND_TYPE_SCALE = {
    "equity": {"BULL": 1.0, "RANGE": 1.0, "BEAR": 1.0},
    "balanced": {"BULL": 0.85, "RANGE": 0.85, "BEAR": 0.85},
    "bond": {"BULL": 0.5, "RANGE": 0.7, "BEAR": 1.0},
    "money": {"BULL": 0.3, "RANGE": 0.4, "BEAR": 0.6},
}


def load_local_close() -> pd.DataFrame:
    """从本地 DuckDB price_daily 表读取收盘价宽表。

    使用本地已缓存的 280+ 只股票数据，无网络依赖，
    市场宽度信号（breadth）也能正常计算。
    """
    con = duckdb.connect(str(_DB_PATH), read_only=True)
    df = con.execute(
        "SELECT symbol, date, close FROM price_daily WHERE adjust = 'qfq' ORDER BY date"
    ).df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values="close")


def latest_signal(
    cfg: Config,
    macro_score: pd.Series | None = None,
    close: pd.DataFrame | None = None,
) -> dict:
    """返回今日最新持仓建议及各层信号分解。

    Args:
        cfg (Config): 全局配置
        macro_score (pd.Series | None, optional): 宏观景气指数.
        close (pd.DataFrame | None, optional): 外部传入的成分股收盘宽表.

    Returns:
        dict: 最新信号建议字典
    """
    if close is None:
        close = load_local_close()

    result_df = compute_position(close, cfg, macro_score=macro_score)
    latest = result_df.dropna().iloc[-1]
    regime_val = latest["regime"]
    label, emoji = REGIME_LABEL.get(regime_val, (regime_val, "⚪️"))

    return {
        "regime": regime_val,
        "regime_label": label,
        "regime_emoji": emoji,
        "regime_scale": float(latest["regime_scale"]),
        "vol_scale": float(latest["vol_scale"]),
        "macro_multiplier": float(latest["macro_multiplier"]),
        "position": float(latest["position"]),
        "date": result_df.dropna().index[-1],
    }


def fund_position_advice(
    signal: dict, holdings_df: pd.DataFrame, total_capital: float
) -> pd.DataFrame:
    """对妹纸持仓基金给出建议仓位和操作方向

    Args:
        signal (dict): `latest_signal()` 的返回值
        holdings_df (pd.DataFrame): 持仓明细
        total_capital (float): 总资金（含现金

    Returns:
        pd.DataFrame: 每行一只基金，含建议仓位比例 / 操作方向 / 偏差金额
    """
    base_position = signal["position"]
    regime = signal["regime"]
    rows: list = []

    for _, h in holdings_df.iterrows():
        ftype = h.get("fund_type", "equity")
        scale = FUND_TYPE_SCALE.get(ftype, FUND_TYPE_SCALE["equity"]).get(regime, 1.0)

        suggested_weight = base_position * scale

        current_mkt = h["mkt"]
        current_weight = current_mkt / total_capital if total_capital > 0 else 0
        suggested_mkt = total_capital * suggested_weight
        diff = suggested_mkt - current_mkt

        if diff > total_capital * 0.02:
            action = "🟢 建议加仓"
        elif diff < -total_capital * 0.02:
            action = "🔴 建议减仓"
        else:
            action = "⚪ 持有不动"

        rows.append(
            {
                "名称": h["name"],
                "代码": h["symbol"],
                "类型": ftype,
                "当前仓位": current_weight,
                "建议仓位": suggested_weight,
                "偏差金额": diff,
                "操作建议": action,
            }
        )

    return pd.DataFrame(rows)
