"""主力资金趋势因子"""

import pandas as pd


def fund_flow_momentum(main_net_pct: pd.Series, window: int) -> pd.Series:
    """计算主力资金趋势因子：主力净流入占比的 N 日滚动均值。

    Args:
        main_net_pct: 主力净流入占比时间序列（单只股票），单位为百分比
        window: 滚动窗口天数，窗口越长信号越平滑但滞后越大

    Returns:
        滚动均值序列，正值表示持续净流入（主力看多），负值表示持续净流出
    """
    return main_net_pct.rolling(window=window).mean()
