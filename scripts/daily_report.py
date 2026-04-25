"""一键生成当日量化日报（Markdown 格式）"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.shared import load_close
from quant.advisor.position import compute_position
from quant.config import Config
from quant.macro.indicators import composite_index
from quant.macro.loader import load_all_macro
from quant.sector.loader import load_sector_close
from quant.sector.rotation import calc_rs, calc_rs_momentum, get_suggestions

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")


def _regime_zh(regime: str) -> str:
    return {"BULL": "牛市 🟢", "RANGE": "震荡 🟡", "BEAR": "熊市 🔴"}.get(
        regime, regime
    )


def build_report() -> str:
    today = date.today().strftime("%Y-%m-%d")
    lines: list[str] = []

    lines += [f"# 量化日报 {today}", ""]

    # ── 1. 仓位建议 ────────────────────────────────────────────────
    print("计算仓位建议…")
    close = load_close()
    macro_df = load_all_macro()
    macro_score = composite_index(macro_df)
    result = compute_position(close, cfg, macro_score=macro_score)

    latest = result.iloc[-1]
    prev_position = result["position"].iloc[-2]

    lines += [
        "## 仓位建议",
        "",
        "| 信号 | 数值 |",
        "|------|------|",
        f"| 市场状态 | {_regime_zh(latest['regime'])} |",
        f"| Regime 仓位 | {latest['regime_scale']:.0%} |",
        f"| 波动率仓位 | {latest['vol_scale']:.0%} |",
        f"| 宏观乘数 | {latest['macro_multiplier']:.2f}x |",
        f"| **建议仓位** | **{latest['position']:.0%}** |",
        f"| 较昨日变化 | {latest['position'] - prev_position:+.0%} |",
        "",
    ]

    # ── 2. 宏观景气 ────────────────────────────────────────────────
    print("加载宏观数据…")
    latest_macro = macro_df.dropna(how="all").iloc[-1]
    latest_score = macro_score.dropna().iloc[-1]

    lines += [
        "## 宏观景气",
        "",
        "| 指标 | 最新值 |",
        "|------|--------|",
        f"| 景气度合成得分 | {latest_score:.2f} |",
        f"| 十年期国债收益率 | {latest_macro['bond_yield']:.2f}% |",
        f"| 制造业 PMI | {latest_macro['pmi']:.1f} |",
        f"| M2 同比 | {latest_macro['m2_yoy']:.1f}% |",
        f"| CPI 同比 | {latest_macro['cpi_yoy']:.1f}% |",
        "",
    ]

    # ── 3. 行业轮动 ────────────────────────────────────────────────
    print("加载行业数据…")
    sector_close = load_sector_close()
    benchmark = sector_close.mean(axis=1)
    rs = calc_rs(sector_close, benchmark, window=20)
    rs_momentum = calc_rs_momentum(rs, lookback=20)
    suggestions = get_suggestions(rs.iloc[-1], rs_momentum, top_n=3)

    overweight = suggestions[suggestions["建议"] == "超配 ▲"]
    underweight = suggestions[suggestions["建议"] == "低配 ▼"]

    lines += ["## 行业轮动", ""]
    lines += ["**超配行业**", ""]
    lines += ["| 行业 | RS | RS动量 |", "|------|-----|--------|"]
    for idx, row in overweight.iterrows():
        lines.append(f"| {idx} | {row['RS']:.3f} | {row['RS动量']:.4f} |")
    lines += [""]
    lines += ["**低配行业**", ""]
    lines += ["| 行业 | RS | RS动量 |", "|------|-----|--------|"]
    for idx, row in underweight.iterrows():
        lines.append(f"| {idx} | {row['RS']:.3f} | {row['RS动量']:.4f} |")
    lines += [""]

    # ── 4. 近期 Regime 切换 ────────────────────────────────────────
    recent = result.tail(30)
    changes = recent[recent["regime"] != recent["regime"].shift()].copy()
    changes["regime"] = changes["regime"].map(_regime_zh)

    lines += ["## 近30日 Regime 切换", ""]
    if changes.empty:
        lines += ["近30日无切换", ""]
    else:
        lines += ["| 日期 | 状态 | 建议仓位 |", "|------|------|---------|"]
        for dt, row in changes.iterrows():
            lines.append(
                f"| {str(dt)[:10]} | {row['regime']} | {row['position']:.0%} |"
            )
        lines += [""]

    return "\n".join(lines)


def main():
    report = build_report()
    out_dir = Path(__file__).parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"report_{date.today().strftime('%Y%m%d')}.md"
    out_file.write_text(report, encoding="utf-8")
    print(f"\n报告已生成：{out_file}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    main()
