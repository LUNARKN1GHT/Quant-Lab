"""多基金组合分析：收益、风险、相关性、风格归因"""

import numpy as np
import pandas as pd


def load_nav_matrix(con, symbols: list[str]) -> pd.DataFrame:
    """从 DuckDB 读取净值矩阵，index=date, columns=symbol"""
    sym_list = "','".join(symbols)
    df = con.execute(f"""
        SELECT date, symbol, nav FROM fund_nav
        WHERE symbol IN ('{sym_list}')
        ORDER BY date
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="symbol", values="nav")


def fund_returns(nav: pd.DataFrame) -> pd.DataFrame:
    return nav.pct_change().dropna(how="all")


def risk_metrics(returns: pd.Series, annual: int = 252) -> dict:
    r = returns.dropna()
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    sharpe = r.mean() / r.std() * np.sqrt(annual) if r.std() > 1e-8 else 0.0
    sortino_std = r[r < 0].std()
    sortino = r.mean() / sortino_std * np.sqrt(annual) if sortino_std > 1e-8 else 0.0
    ann_ret = (cum.iloc[-1] ** (annual / len(r))) - 1 if len(r) > 0 else 0.0
    return {
        "年化收益": ann_ret,
        "年化波动": r.std() * np.sqrt(annual),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "最大回撤": dd.min(),
        "Calmar": ann_ret / abs(dd.min()) if dd.min() != 0 else 0.0,
        "近1月": (1 + r.tail(21)).prod() - 1,
        "近3月": (1 + r.tail(63)).prod() - 1,
        "近1年": (1 + r.tail(252)).prod() - 1,
    }


def portfolio_value(nav: pd.DataFrame, holdings: list[dict]) -> pd.DataFrame:
    """计算每只基金和组合总市值时序

    holdings: [{"symbol": "009610", "shares": 10000, "cost_nav": 1.25}, ...]
    """
    rows = []
    for h in holdings:
        sym = h["symbol"]
        if sym not in nav.columns:
            continue
        s = nav[sym].dropna()
        rows.append(
            {
                "symbol": sym,
                "name": h.get("name", sym),
                "shares": h["shares"],
                "cost_nav": h["cost_nav"],
                "latest_nav": s.iloc[-1],
                "cost_value": h["shares"] * h["cost_nav"],
                "latest_value": h["shares"] * s.iloc[-1],
                "pnl": h["shares"] * (s.iloc[-1] - h["cost_nav"]),
                "pnl_pct": (s.iloc[-1] / h["cost_nav"] - 1),
            }
        )
    return pd.DataFrame(rows)


def style_attribution(returns: pd.Series, benchmarks: pd.DataFrame) -> dict:
    """将基金收益回归到基准因子，输出 Beta 暴露和 Alpha

    benchmarks: DataFrame，columns 为各基准收益率（如 CSI300、CSI500、债券指数）
    """
    common = returns.index.intersection(benchmarks.index)
    if len(common) < 60:
        return {}
    y = returns.loc[common].values
    X = benchmarks.loc[common].values
    X = np.column_stack([X, np.ones(len(X))])  # 加截距
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - np.var(y - X @ coef) / np.var(y)
    result = {benchmarks.columns[i]: coef[i] for i in range(len(benchmarks.columns))}
    result["Alpha（日）"] = coef[-1]
    result["R²"] = r2
    return result
