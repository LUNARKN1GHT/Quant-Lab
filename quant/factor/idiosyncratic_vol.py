"""特质波动率因子（Idiosyncratic Volatility）

将个股收益率对市场收益率做滚动回归，残差的标准差即为特质波动率，
代表无法被市场因子解释的个股特有风险。

CAPM：r_i = alpha + beta * r_m + epsilon
特质波动率 = std(epsilon)，越高说明个股走势越独立于大盘。
"""

import pandas as pd


def idiosyncratic_vol(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """计算 N 日滚动特质波动率。

    Args:
        close: 个股收盘价时间序列
        market_close: 市场指数收盘价（如沪深300），与 close 日期对齐
        window: 滚动窗口（交易日），默认 20

    Returns:
        特质波动率序列
    """
    r = close.pct_change()        # 个股日收益率
    m = market_close.pct_change() # 市场日收益率

    # 用滚动协方差和市场方差估计滚动 beta
    roll_cov = r.rolling(window).cov(m)
    roll_var = m.rolling(window).var()
    beta = roll_cov / roll_var  # beta = Cov(r, m) / Var(m)

    # 残差 = 个股收益 - beta * 市场收益（去除系统性部分）
    residuals = r - beta * m
    return residuals.rolling(window).std()
