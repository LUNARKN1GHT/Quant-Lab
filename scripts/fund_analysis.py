"""分析持仓基金的历史表现，输出方向性建议"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from quant.fund.loader import load_funds
from quant.risk.metrics import calmar, max_drawdown, sharpe

FUNDS = ["011103", "004512", "024307", "009610"]


def analyse(nav: pd.DataFrame) -> pd.DataFrame:
    returns = nav.pct_change().dropna(how="all")

    rows = []
    for code in nav.columns:
        r = returns[code].dropna()
        if len(r) < 60:
            continue

        ann_ret = r.mean() * 252
        mom_3m = (1 + r.tail(63)).prod() - 1  # 近3月动量
        mom_1m = (1 + r.tail(21)).prod() - 1  # 近1月动量
        dd = max_drawdown(r)
        sp = sharpe(r)
        cl = calmar(r)

        # 简单方向建议：基于近期动量 + 回撤
        if mom_1m > 0.02 and mom_3m > 0.05 and dd > -0.15:
            direction = "加仓 ▲"
        elif mom_1m < -0.03 or dd < -0.25:
            direction = "减仓 ▼"
        else:
            direction = "持有 →"

        rows.append(
            {
                "基金": code,
                "年化收益": ann_ret,
                "近3月": mom_3m,
                "近1月": mom_1m,
                "Sharpe": sp,
                "最大回撤": dd,
                "Calmar": cl,
                "建议": direction,
            }
        )
    return pd.DataFrame(rows).set_index("基金")


def main():
    print("拉取基金净值...")
    nav = load_funds(FUNDS)
    print(f"\n数据范围: {nav.index[0].date()} ~ {nav.index[-1].date()}\n")

    result = analyse(nav)

    print(
        f"{'基金':<8} {'年化收益':>8} {'近3月':>7} {'近1月':>7} "
        f"{'Sharpe':>7} {'最大回撤':>8} {'Calmar':>7} {'建议':>8}"
    )
    print("=" * 70)
    for code, row in result.iterrows():
        print(
            f"{code:<8} {row['年化收益']:>7.2%} {row['近3月']:>7.2%} "
            f"{row['近1月']:>7.2%} {row['Sharpe']:>7.3f} "
            f"{row['最大回撤']:>7.2%} {row['Calmar']:>7.3f}  {row['建议']}"
        )


if __name__ == "__main__":
    main()
