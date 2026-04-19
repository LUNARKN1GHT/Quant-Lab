import pandas as pd
from sklearn.base import RegressorMixin, clone


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model: RegressorMixin,
    train_window: int,
    predict_window: int,
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
    # 用于存预测结果
    predictions = pd.Series(index=y.index, dtype=float)
    total = len(X)
    start = train_window  # 从第一个完整训练窗口结束的位置开始

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
