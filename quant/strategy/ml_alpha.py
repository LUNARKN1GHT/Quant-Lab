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
    """随机游走验证

    Args:
        X (pd.DataFrame): 特征（因子值）
        y (pd.Series): 标签（下期收益）
        train_window (int): 训练窗口大小
        predict_window (int): 每次预测多少期

    Returns:
        pd.Series: 返回样本外预测值
    """
    if model is None:
        model = Ridge()
    predictions = pd.Series(index=y.index, dtype=float)
    total = len(X)
    start = train_window

    while start < total:
        end = min(start + predict_window, total)  # 本轮预测到的位置

        # 1. 切出训练集
        X_train = X.iloc[start - train_window : start]
        y_train = y.iloc[start - train_window : start]

        # 2. 切出预测集
        X_pred = X.iloc[start:end]

        # 3. 训练模型
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        predictions.iloc[start:end] = fold_model.predict(X_pred)

        start += predict_window  # 滚动前进

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
    predictions = pd.Series(index=y.index, dtype=float)
    total = len(X)
    start = train_window

    while start < total:
        end = min(start + predict_window, total)

        # 完整训练集
        X_train_all = X.iloc[start - train_window : start]
        y_train_all = y.iloc[start - train_window : start]

        # 切分出 train1 和 holdout
        split = int(len(X_train_all) * (1 - holdout_ratio))
        X_train1, y_train1 = X_train_all.iloc[:split], y_train_all.iloc[:split]
        X_holdout, y_holdout = X_train_all.iloc[split:], y_train_all.iloc[split:]

        # 预测集
        X_pred = X.iloc[start:end]

        # 为每个基模型生成元特征
        meta_train = np.zeros((len(X_holdout), len(base_models)))
        meta_pred = np.zeros((len(X_pred), len(base_models)))

        for i, base_model in enumerate(base_models):
            # 在 train1 训练 → 预测 holdout（生成元特征训练集）
            m1 = clone(base_model)
            m1.fit(X_train1, y_train1)
            meta_train[:, i] = m1.predict(X_holdout)

            # 在全训练集重训 → 预测当期（生成元特征测试集）
            m2 = clone(base_model)
            m2.fit(X_train_all, y_train_all)
            meta_pred[:, i] = m2.predict(X_pred)

        # meta-model 学习如何组合基模型
        mm = clone(meta_model)
        mm.fit(meta_train, y_holdout)
        predictions.iloc[start:end] = mm.predict(meta_pred)

        start += predict_window

    return predictions
