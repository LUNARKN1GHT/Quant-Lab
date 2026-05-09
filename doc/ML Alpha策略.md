# ML Alpha 策略

**代码**：[quant/strategy/ml_alpha.py](../quant/strategy/ml_alpha.py) · [scripts/backtest_ml.py](../scripts/backtest_ml.py) · [scripts/compare_models.py](../scripts/compare_models.py)

## 模块功能

用机器学习模型预测股票截面收益排序，每月选出预测分最高的 Top N 只股票等权持仓。

与纯因子策略的区别：因子策略用单一因子或线性加权组合打分，ML 策略让模型自己学习多个因子之间的非线性关系，自动找到在历史数据中最有预测力的因子组合方式。

---

## 整体流程

```txt
历史价格数据
    ↓
因子特征矩阵（date × stock，5 个技术因子）
    ↓
Walk-Forward 训练：用过去 train_window 期训练，预测下一个 predict_window 期
    ↓
每期截面预测分（每只股票的预测收益排序）
    ↓
Top N 等权持仓 → 向量化回测 → 绩效评估
```

---

## 截面预测的本质

这是一个**截面回归问题**，而不是时序预测问题。

- **传统时序预测**：预测某只股票未来价格是涨还是跌（时间维度）
- **截面预测**：在同一时间点，预测哪些股票的涨幅会高于其他股票（横截面排序）

模型的目标不是预测收益的绝对大小，而是给出一个**相对排序**——只需要预测"A 会比 B 涨得多"，选出前 N 名即可。因此，即便预测值的绝对精度很差，只要排序方向对了，策略就能盈利。

**特征工程（`build_features()`）**

当前使用 5 个技术因子构成特征矩阵：

| 因子 | 含义 |
| --- | --- |
| `mom_20` | 20 日动量（短期趋势） |
| `mom_60` | 60 日动量（中期趋势） |
| `rsi_14` | 14 日 RSI（超买超卖） |
| `vol_20` | 20 日波动率（风险暴露） |
| `ma_bias_20` | 20 日均线偏离率（均值回归信号） |

宽表 `(date, stock)` → `stack()` 转为长格式 `MultiIndex(date, stock)`，每行是一只股票在某日的因子截面，每列是一个因子值。

---

## Walk-Forward 验证

Walk-Forward 是时序数据的标准验证方式，核心原则：**训练集始终在预测集之前**。

```txt
|<--- train_window --->|<-- predict_window -->|
      训练集                  预测期 1

                        |<--- train_window --->|<-- predict_window -->|
                                训练集                  预测期 2

                                               |<--- train_window --->|<-- predict_window -->|
                                                       训练集                  预测期 3
```

每次向前滑动一个 `predict_window`，模拟真实策略上线后的月度再训练流程。

**为什么不能用简单的训练集/测试集划分（train_test_split）？**

金融时序数据有时间依赖性，若随机划分：

1. 测试集的样本可能出现在训练集时间之前，相当于用未来数据预测过去
2. 同一时期的多只股票高度相关，随机切分会把"同期"的样本同时放入训练集和测试集，导致泄漏

Walk-Forward 完全避免了这两个问题。

**`clone(model)` 的作用**

每次循环前用 `clone()` 创建一个全新未训练的模型实例，防止上一期的训练状态污染下一期——这是 sklearn 标准范式，在循环训练中容易漏写。

---

## 支持的模型

通过统一的 `sklearn` 兼容接口，任意回归模型都可以直接传入：

| 模型 | 特点 |
| --- | --- |
| **LightGBM**（默认） | Boosting，速度快，对缺失值鲁棒，适合高维稀疏特征 |
| **XGBoost** | 与 LightGBM 类似，精度相近，速度略慢 |
| **Random Forest** | Bagging，并行训练，方差更低，不容易过拟合 |
| **Ridge** | 线性模型作为 baseline，正则化控制因子权重，可解释性最好 |

**Bagging vs Boosting 的区别**：

- **Boosting（LightGBM / XGBoost）**：串行训练，每棵树专注修正上一棵的错误，对训练集拟合更强，需要调 `n_estimators` / `learning_rate` 防止过拟合
- **Bagging（Random Forest）**：并行训练多棵树，每棵树在随机子样本上训练，天然有更低的方差，不容易过拟合，但偏差可能更高

---

## Stacking 集成

单模型预测依赖该模型的归纳偏置（假设数据分布的方式），不同市场环境下各模型表现差异大。Stacking 把多个基模型的预测作为元特征，让元模型学习如何组合它们：

```txt
每个调仓期内：

训练集
├── train1（前 80%）→ 训练基模型 → 预测 holdout → 元特征（训练集）
└── holdout（后 20%）→ 用于训练 meta_model

全训练集 → 重训基模型 → 预测当期截面 → 元特征（预测集）

meta_model.fit(元特征训练集) → meta_model.predict(元特征预测集) → 最终预测分
```

**为什么训练集要切出 holdout？**

如果基模型在全训练集上训练，再预测训练集自身，元特征会过拟合（基模型对训练集太准了，元特征里的信息不真实）。切出 holdout 确保元特征是真实的样本外预测。

---

## Data Snooping（数据窥视）风险

ML 模型参数多，在足够大的搜索空间里总能找到"历史最优"的参数组合，但这些参数对未来可能毫无意义——这就是 data snooping。

本项目防范 data snooping 的几个设计：

1. **Walk-Forward 保证时序隔离**：预测集永远在训练集之后，没有未来数据
2. **`clone()` 防止状态污染**：每期模型独立训练，不积累历史拟合状态
3. **特征数量有限（5 个）**：特征越多，过拟合风险越高；当前保持最小有效特征集
4. **线性 Ridge 作为 baseline**：如果 LightGBM 显著优于 Ridge，需要怀疑是否过拟合——因为线性模型更难过拟合，若 Ridge 结果也差，说明信号本身弱

---

## 局限与边界

- **标签设计简单**：直接用前向 N 期收益作为标签，没有做行业中性化或市值中性化，模型可能学到的是市值/行业暴露而非真正的 Alpha
- **特征工程局限**：当前只有 5 个技术因子，没有接入财务因子（ROE / 盈利质量等），信号来源单一
- **换手率高**：每月调仓，Top N 的成分变动较大，手续费摩擦不可忽视
- **容量限制**：截面选股类策略在规模大时会有冲击成本，小资金适用，大资金会影响价格

---

## 与其他模块的关系

- 上游：`quant/factor/` 各因子模块提供特征，`quant/data/cache.py` 提供收盘价数据
- 下游：`quant/backtest/engine.py` 执行回测，`quant/risk/metrics.py` 评估绩效
- 并列：`quant/strategy/factor_strategy.py`（纯因子选股）、`quant/strategy/pairs_trading.py`（统计套利），三者在 `scripts/compare_models.py` 中横向对比
- Dashboard：`pages/3_backtest_compare.py` 展示各模型的绩效对比
