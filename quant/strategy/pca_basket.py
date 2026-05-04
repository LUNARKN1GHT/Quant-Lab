"""PCA 篮子套利

将板块内所有股票的收益率分解为：
  r_i = β_i * F1 + ε_i
其中 F1 为第一主成分（板块共同因子），ε_i 为个股特质残差。
当 ε_i 偏离历史均值过大时，做空高估股、做多低估股。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    loadings: pd.Series  # 每只股票在 PC1 上的载荷 β_i
    factor_series: pd.Series  # PC1 因子时序 F1_t
    residuals: pd.DataFrame  # 特质残差矩阵 ε_{i,t}，index=date, columns=symbol
    explained_ratio: float  # PC1 解释方差比例


def rolling_pca_residuals(
    returns: pd.DataFrame,
    window: int = 60,
    n_components: int = 1,
) -> pd.DataFrame:
    """滚动 PCA：每日用过去 window 期数据估计主成分，输出当日残差。

    避免 look-ahead bias：第 t 日仅使用 [t-window, t-1] 的数据拟合 PCA，
    再将第 t 日收益率投影到主成分空间计算残差。

    Args:
        returns:      日收益率矩阵，index=date, columns=symbol
        window:       估计窗口（交易日数）
        n_components: 保留的主成分数量

    Returns:
        残差矩阵，shape 与 returns 相同，前 window 行为 NaN
    """
    n_dates, n_stocks = returns.shape
    residuals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for t in range(window, n_dates):
        train = returns.iloc[t - window : t].dropna(axis=1)
        if train.shape[1] < 5:
            continue

        pca = PCA(n_components=n_components)
        pca.fit(train.values)

        # 当日收益率（只取 train 中有效的列）
        today = returns.iloc[t][train.columns]
        if today.isna().all():
            continue

        # 投影到主成分空间再重建
        today_arr = today.fillna(0).values.reshape(1, -1)
        projected = pca.transform(today_arr)  # (1, n_components)
        reconstructed = pca.inverse_transform(projected).flatten()

        resid = today.values - reconstructed
        residuals.loc[returns.index[t], train.columns] = resid

    return residuals


def pca_zscore(residuals: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """对残差做截面标准化 z-score（每日横截面均值/std）"""
    roll_mean = residuals.rolling(window).mean()
    roll_std = residuals.rolling(window).std()
    return (residuals - roll_mean) / roll_std.clip(lower=1e-6)


def basket_signal(
    zscore: pd.DataFrame,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5,
) -> pd.DataFrame:
    """根据 z-score 生成多空信号矩阵。

    Returns:
        信号矩阵：+1=做多（低估），-1=做空（高估），0=空仓
    """
    signal = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)
    signal[zscore < -entry_threshold] = 1  # 残差过低：做多
    signal[zscore > entry_threshold] = -1  # 残差过高：做空
    signal[(zscore.abs() < exit_threshold)] = 0
    return signal


def portfolio_return(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    holding_period: int = 1,
) -> pd.Series:
    """截面等权多空组合收益。

    做多低 z-score 股票、做空高 z-score 股票，各腿等权。
    """
    pnl = []
    for date, row in signal.iterrows():
        longs = row[row == 1].index.tolist()
        shorts = row[row == -1].index.tolist()
        if not longs and not shorts:
            pnl.append({"date": date, "ret": 0.0})
            continue

        idx = returns.index.get_loc(date)
        if idx + holding_period >= len(returns):
            pnl.append({"date": date, "ret": np.nan})
            continue

        fwd = returns.iloc[idx : idx + holding_period]
        long_ret = fwd[longs].mean(axis=1).mean() if longs else 0.0
        short_ret = fwd[shorts].mean(axis=1).mean() if shorts else 0.0
        pnl.append({"date": date, "ret": long_ret - short_ret})

    df = pd.DataFrame(pnl).set_index("date")["ret"]
    return df
