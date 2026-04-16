import pandas as pd


def equal_weight(factors: pd.DataFrame) -> pd.Series:
    """等权合成因子，每列是一个因子"""
    return factors.mean(axis=1)


def ic_weight(factors: pd.DataFrame, ic_scores: pd.DataFrame) -> pd.Series:
    """IC 加权合成， ic_scores 是每个因子对应的 IC"""
    weights = ic_scores / ic_scores.sum()
    return factors.mul(weights, axis=1).sum(axis=1)
