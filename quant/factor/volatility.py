"""波动率因子：衡量过去 N 天日收益率的标准差

低波动异象（Low Volatility Anomaly）：低波动股票长期风险调整收益往往优于高波动股票。
波动率因子常作为风险因子使用，也可作为反转信号（高波动后均值回归）。
"""

import pandas as pd


def volatility(close: pd.Series, window: int) -> pd.Series:
    """计算 N 日滚动波动率（日收益率的滚动标准差）。

    Args:
        close: 收盘价时间序列
        window: 滚动窗口大小（交易日），常用值：20（月度）、60（季度）

    Returns:
        波动率因子序列，前 window 个值为 NaN
    """
    # 日收益率：r_t = close_t / close_{t-1} - 1
    daily_return = close / close.shift(1) - 1
    # 滚动标准差即为历史波动率
    return daily_return.rolling(window=window).std()
