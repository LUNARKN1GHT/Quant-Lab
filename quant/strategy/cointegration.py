"""协整分析工具：Engle-Granger / Johansen 检验、残差分析、滚动稳定性"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass
class EGResult:
    """Engle-Granger 两步法检验结果"""

    pvalue: float
    hedge_ratio: float
    residual: pd.Series
    adf_stat: float
    adf_pvalue: float
    is_cointegrated: bool


@dataclass
class JohansenResult:
    """Johansen 检验结果"""

    n_cointegration: int  # 协整关系数量
    trace_stats: np.ndarray  # Trace 统计量
    trace_crit: np.ndarray  # 临界值（95%）
    eigen_stats: np.ndarray  # 最大特征值统计量
    eigen_crit: np.ndarray
    is_cointegrated: bool


def engle_granger(
    price_a: pd.Series,
    price_b: pd.Series,
    pvalue_threshold: float = 0.05,
) -> EGResult:
    """Engle-Granger 两步法协整检验

    步骤：
      1. OLS 回归 price_a ~ price_b，得到对冲比率和残差
      2. 对残差做 ADF 检验，残差平稳则两序列协整

    Args:
        price_a: 股票 A 价格序列
        price_b: 股票 B 价格序列
        pvalue_threshold: 显著性水平

    Returns:
        EGResult 含检验统计量、对冲比率、残差序列
    """
    _, pvalue, _ = coint(price_a, price_b)

    # OLS 估计对冲比率
    x = pd.DataFrame({"b": price_b, "const": 1.0})
    model = OLS(price_a, x).fit()
    hedge = model.params["b"]
    residual = pd.Series(model.resid, index=price_a.index)

    adf_stat, adf_pvalue, *_ = adfuller(residual, autolag="AIC")

    return EGResult(
        pvalue=pvalue,
        hedge_ratio=hedge,
        residual=residual,
        adf_stat=adf_stat,
        adf_pvalue=adf_pvalue,
        is_cointegrated=pvalue < pvalue_threshold,
    )


def johansen(
    price_a: pd.Series,
    price_b: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen 协整检验

    基于 VAR 模型，同时检验协整关系数量和估计协整向量。
    适合多变量场景，比 Engle-Granger 更通用。

    Args:
        price_a: 股票 A 价格序列
        price_b: 股票 B 价格序列
        det_order: 确定性项（-1=无常数, 0=有常数, 1=有趋势）
        k_ar_diff: VAR 差分阶数

    Returns:
        JohansenResult 含 Trace/最大特征值统计量与临界值
    """
    data = pd.concat([price_a, price_b], axis=1).dropna()
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

    # cvt: 临界值矩阵，列为 90%/95%/99%，取第 1 列（95%）
    trace_stats = result.lr1  # Trace 统计量
    trace_crit = result.cvt[:, 1]  # 95% 临界值
    eigen_stats = result.lr2  # 最大特征值统计量
    eigen_crit = result.cvm[:, 1]

    # 统计显著的协整关系数
    n_coint = int((trace_stats > trace_crit).sum())

    return JohansenResult(
        n_cointegration=n_coint,
        trace_stats=trace_stats,
        trace_crit=trace_crit,
        eigen_stats=eigen_stats,
        eigen_crit=eigen_crit,
        is_cointegrated=n_coint > 0,
    )


def residual_diagnostics(residual: pd.Series, lags: int = 20) -> dict:
    """协整残差诊断：ADF 平稳性 + Ljung-Box 自相关 + DW 统计量

    Args:
        residual: 协整残差序列
        lags: Ljung-Box 检验的滞后阶数

    Returns:
        包含各诊断指标的字典
    """
    adf_stat, adf_pvalue, *_ = adfuller(residual, autolag="AIC")
    lb = acorr_ljungbox(residual, lags=[lags], return_df=True)
    dw = durbin_watson(residual)

    return {
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "is_stationary": adf_pvalue < 0.05,
        "ljung_box_stat": lb["lb_stat"].iloc[0],
        "ljung_box_pvalue": lb["lb_pvalue"].iloc[0],
        "has_autocorr": lb["lb_pvalue"].iloc[0] < 0.05,
        "durbin_watson": dw,  # 接近 2 表示无自相关
    }


def rolling_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    window: int = 60,
) -> pd.Series:
    """滚动 OLS 估计对冲比率，检验协整关系的时变稳定性

    对冲比率在滚动窗口内波动越小，说明协整关系越稳定。

    Args:
        price_a: 股票 A 价格序列
        price_b: 股票 B 价格序列
        window: 滚动窗口天数

    Returns:
        每日对冲比率序列
    """
    ratios = []
    for i in range(window, len(price_a) + 1):
        a = price_a.iloc[i - window : i]
        b = price_b.iloc[i - window : i]
        ratio = np.polyfit(b, a, 1)[0]
        ratios.append({"date": price_a.index[i - 1], "hedge_ratio": ratio})
    return pd.DataFrame(ratios).set_index("date")["hedge_ratio"]
