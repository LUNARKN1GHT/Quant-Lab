# scripts/find_cointegrated_pairs.py
"""在沪深300中批量筛选高质量协整股票对"""

import sys
from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from quant.strategy.cointegration import (
    engle_granger,
    residual_diagnostics,
    rolling_hedge_ratio,
)

DB_PATH = "data/quant.duckdb"

# 只在同行业内搜索，减少伪协整
# 银行：600036 601166 601398 601288 601939
# 白酒：600519 000858 000568 600809
# 家电：000333 000651
SECTOR_POOLS = {
    "银行": ["600036", "601166", "601398", "601288", "601939"],
    "白酒": ["600519", "000858", "000568", "600809"],
    "家电": ["000333", "000651"],
}


def load_prices(con, symbols: list[str]) -> pd.DataFrame:
    syms = "','".join(symbols)
    price = con.execute(f"""
        SELECT symbol, date, close FROM price_daily
        WHERE adjust = 'qfq' AND symbol IN ('{syms}')
        ORDER BY date
    """).df()
    price["date"] = pd.to_datetime(price["date"])
    return price.pivot(index="date", columns="symbol", values="close").dropna()


def score_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    roll_window: int = 120,
) -> dict | None:
    """对一个股票对打分，返回 None 表示不通过筛选"""
    eg = engle_granger(price_a, price_b)
    if not eg.is_cointegrated:
        return None

    diag = residual_diagnostics(eg.residual)
    if not diag["is_stationary"]:
        return None

    roll = rolling_hedge_ratio(price_a, price_b, window=roll_window)
    # 变异系数 = 标准差 / |均值|，衡量对冲比率的稳定性
    cv = roll.std() / abs(roll.mean()) if roll.mean() != 0 else float("inf")

    return {
        "eg_pvalue": eg.pvalue,
        "adf_pvalue": diag["adf_pvalue"],
        "has_autocorr": diag["has_autocorr"],
        "durbin_watson": diag["durbin_watson"],
        "hedge_ratio_mean": roll.mean(),
        "hedge_ratio_cv": cv,
        "hedge_ratio": eg.hedge_ratio,
    }


def main():
    con = duckdb.connect(DB_PATH)
    results = []

    for sector, symbols in SECTOR_POOLS.items():
        print(f"\n=== {sector} ===")
        prices = load_prices(con, symbols)
        available = [s for s in symbols if s in prices.columns]

        for sym_a, sym_b in combinations(available, 2):
            score = score_pair(prices[sym_a], prices[sym_b])
            if score:
                score.update({"sector": sector, "symbol_a": sym_a, "symbol_b": sym_b})
                results.append(score)
                flag = "⚠️ 有自相关" if score["has_autocorr"] else "✅"
                print(
                    f"  {sym_a} vs {sym_b}  "
                    f"EG={score['eg_pvalue']:.3f}  "
                    f"ADF={score['adf_pvalue']:.3f}  "
                    f"CV={score['hedge_ratio_cv']:.2f}  {flag}"
                )

    if not results:
        print("\n未找到通过筛选的股票对")
        return

    df = pd.DataFrame(results).sort_values("hedge_ratio_cv")
    print("\n=== 通过筛选的股票对（按对冲比率稳定性排序）===")
    cols = [
        "sector",
        "symbol_a",
        "symbol_b",
        "eg_pvalue",
        "adf_pvalue",
        "hedge_ratio_cv",
        "has_autocorr",
    ]
    print(df[cols].to_string(index=False))

    out = Path("output/factor_research/cointegrated_pairs.csv")
    df.to_csv(out, index=False)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
