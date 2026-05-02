"""主力资金趋势因子"""

import pandas as pd


def fund_flow_momentum(main_net_pct: pd.Series, window: int) -> pd.Series:
    """主力净流入占比的滚动均值

    Args:
        main_net_pct (pd.Series): 主力净流入占比时间序列（单只股票）
        window (int): 滚动窗口天数

    Returns:
        pd.Series: _description_
    """
    return main_net_pct.rolling(window=window).mean()
