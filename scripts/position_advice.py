"""输出当前仓位建议及历史仓位变化"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.advisor.position import compute_position
from quant.config import Config
from scripts.backtest_ml import load_close

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")


def main():
    print("加载数据...")
    close = load_close()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日\n")

    result = compute_position(close, cfg)

    # 最新建议
    latest = result.iloc[-1]
    print("=" * 40)
    print(f"最新建议（{result.index[-1].date()}）")
    print("=" * 40)
    print(f"  市场状态  : {latest['regime']}")
    print(f"  Regime仓位: {latest['regime_scale']:.0%}")
    print(f"  波动率仓位: {latest['vol_scale']:.0%}")
    print(f"  建议仓位  : {latest['position']:.0%}")

    # 近30个交易日仓位变化
    print("\n=== 近30日仓位变化 ===")
    recent = result.tail(30)[["regime", "position"]]
    changes = recent[recent["position"] != recent["position"].shift()]
    print(changes.to_string(float_format="{:.0%}".format))

    # 各状态历史统计
    print("\n=== 历史仓位分布 ===")
    print(result["position"].describe().apply("{:.2%}".format))


if __name__ == "__main__":
    main()
