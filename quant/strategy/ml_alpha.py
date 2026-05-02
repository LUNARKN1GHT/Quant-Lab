"""机器学习 Alpha 模型：滚动时序验证（Walk-Forward）

Walk-Forward 是时序数据的标准验证方式：
- 训练集始终在预测集之前，不存在未来数据泄漏
- 每次向前滑动一个 predict_window，模拟真实上线后的滚动再训练
"""

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.linear_model import Ridge


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_window: int,
    predict_window: int,
    model: RegressorMixin = None,
) -> pd.Series:
    """单模型滚动时序验证，返回样本外预测值。

    Args:
        X: 特征矩阵（因子值），行为时间，列为因子
        y: 标签序列（下期收益），与 X 行对齐
        train_window: 每次训练使用的历史期数
        predict_window: 每轮预测的期数（向前滑动步长）
        model: 回归模型，默认 Ridge 回归

    Returns:
        样本外预测序列，前 train_window 期为 NaN（无预测）
    """
    if model is None:
        model = Ridge()
    predictions = pd.Series(index=y.index, dtype=float)
    total = len(X)
    start = train_window

    while start < total:
        end = min(start + predict_window, total)

        X_train = X.iloc[start - train_window : start]
        y_train = y.iloc[start - train_window : start]
        X_pred = X.iloc[start:end]

        # clone 确保每轮使用全新未训练的模型，避免状态污染
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        predictions.iloc[start:end] = fold_model.predict(X_pred)

        start += predict_window  # 滚动前进一个 predict_window

    return predictions


def walk_forward_stack(
    X: pd.DataFrame,
    y: pd.Series,
    base_models: list[RegressorMixin],
    meta_model: RegressorMixin,
    train_window: int,
    predict_window: int,
    holdout_ratio: float = 0.2,
) -> pd.Series:
    """Stacking 集成 + 滚动时序验证，返回样本外预测值。

    在每个滚动窗口内：
    1. 将训练集切分为 train1（80%）和 holdout（20%）
    2. 各基模型在 train1 上训练，预测 holdout，生成元特征训练集
    3. 各基模型在全训练集重训，预测当期，生成元特征预测集
    4. meta_model 学习如何加权组合基模型的预测

    Args:
        base_models: 基模型列表（如 Ridge、RandomForest）
        meta_model: 学习如何组合基模型的元模型
        holdout_ratio: 训练集中用于生成元特征的比例，默认 0.2
    """
    predictions = pd.Series(index=y.index, dtype=float)
    total = len(X)
    start = train_window

    while start < total:
        end = min(start + predict_window, total)

        X_train_all = X.iloc[start - train_window : start]
        y_train_all = y.iloc[start - train_window : start]

        # 切分训练集：train1 用于训练基模型，holdout 用于生成元特征
        split = int(len(X_train_all) * (1 - holdout_ratio))
        X_train1, y_train1 = X_train_all.iloc[:split], y_train_all.iloc[:split]
        X_holdout, y_holdout = X_train_all.iloc[split:], y_train_all.iloc[split:]

        X_pred = X.iloc[start:end]

        meta_train = np.zeros((len(X_holdout), len(base_models)))
        meta_pred = np.zeros((len(X_pred), len(base_models)))

        for i, base_model in enumerate(base_models):
            # 在 train1 训练 → 预测 holdout（生成元特征训练集）
            m1 = clone(base_model)
            m1.fit(X_train1, y_train1)
            meta_train[:, i] = m1.predict(X_holdout)

            # 在全训练集重训 → 预测当期（生成元特征预测集）
            m2 = clone(base_model)
            m2.fit(X_train_all, y_train_all)
            meta_pred[:, i] = m2.predict(X_pred)

        # meta-model 学习如何组合基模型，生成最终预测
        mm = clone(meta_model)
        mm.fit(meta_train, y_holdout)
        predictions.iloc[start:end] = mm.predict(meta_pred)

        start += predict_window

    return predictions
