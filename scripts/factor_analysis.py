"""单因子验证脚本：动量因子 IC / ICIR / 分层收益"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.config import Config
from quant.factor.ic import calc_ic, calc_icir
from quant.factor.layered import layered_return
from quant.factor.momentum import momentum

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")

DATA_DIR = Path(__file__).parent.parent / cfg.data.data_dir


def load_all() -> pd.DataFrame:
    """加载全部 CSV，返回宽表 close，index=trade_date，columns=ts_code"""
    frames = {}
    for f in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(f, usecols=["trade_date", "close"])
        except (pd.errors.EmptyDataError, ValueError):
            continue
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        df = df.set_index("trade_date").sort_index()
        frames[f.stem] = df["close"]
    return pd.DataFrame(frames)


def run():
    print("加载数据...")
    close = load_all()
    print(f"  {close.shape[1]} 只股票，{len(close)} 个交易日")

    # 计算动量因子（截面：每日每只股票的因子值）
    factor_df = close.apply(lambda s: momentum(s, cfg.factor.momentum_windows[0]))

    # 计算前向收益
    fwd_return_df = close.pct_change(cfg.factor.ic_forward_window).shift(
        -cfg.factor.ic_forward_window
    )

    # 逐日计算截面 IC
    ic_list = []
    for date in factor_df.index:
        f = factor_df.loc[date].dropna()
        r = fwd_return_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 10:
            continue
        ic = calc_ic(f[common], r[common])
        ic_list.append({"date": date, "ic": ic})

    ic_series = pd.DataFrame(ic_list).set_index("date")["ic"].dropna()

    icir = calc_icir(ic_series)
    ic_mean = ic_series.mean()
    ic_positive_ratio = (ic_series > 0).mean()

    print(
        f"\n=== 动量因子（回看 {cfg.factor.momentum_windows[0]} 日，"
        + f"预测 {cfg.factor.ic_forward_window} 日）==="
    )
    print(f"  样本期数   : {len(ic_series)}")
    print(f"  IC 均值    : {ic_mean:.4f}")
    print(f"  IC 标准差  : {ic_series.std():.4f}")
    print(f"  ICIR       : {icir:.4f}")
    print(f"  IC>0 占比  : {ic_positive_ratio:.1%}")

    # 分层收益（用最后一个有效截面演示）
    last_date = ic_series.index[-1]
    f_last = factor_df.loc[last_date].dropna()
    r_last = fwd_return_df.loc[last_date].dropna()
    common = f_last.index.intersection(r_last.index)

    layers = layered_return(
        f_last[common], r_last[common], n_groups=cfg.factor.n_groups
    )
    print(f"\n=== 最后截面（{last_date.date()}）分层收益 ===")
    for g, ret in layers.items():
        bar = "█" * int(abs(ret) * 500)
        sign = "+" if ret >= 0 else "-"
        print(f"  Q{g}: {sign}{abs(ret):.2%}  {bar}")

    # 全样本分层平均收益
    print(f"\n=== 全样本平均分层收益（{cfg.factor.n_groups} 分位）===")
    all_layers = []
    for date in ic_series.index:
        f = factor_df.loc[date].dropna()
        r = fwd_return_df.loc[date].dropna()
        common = f.index.intersection(r.index)
        if len(common) < cfg.factor.n_groups * 2:
            continue
        try:
            layer = layered_return(f[common], r[common], n_groups=cfg.factor.n_groups)
            all_layers.append(layer)
        except Exception:
            continue

    avg_layers = pd.DataFrame(all_layers).mean()
    for g, ret in avg_layers.items():
        bar = "█" * int(abs(ret) * 2000)
        sign = "+" if ret >= 0 else "-"
        print(f"  Q{g}: {sign}{abs(ret):.2%}  {bar}")

    spread = avg_layers.iloc[-1] - avg_layers.iloc[0]
    print(f"\n  Q5-Q1 价差: {spread:+.2%}")

    ROOT = Path(__file__).parent.parent
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.to_yaml(result_dir / f"config_{timestamp}.yaml")


if __name__ == "__main__":
    run()
