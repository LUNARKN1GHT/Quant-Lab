# Quant-Lab 学习路线图

> 本文档记录个人学习目标与阶段计划，与 README（对外展示）分开维护。
>
> 三个核心目标：
> 1. 提升工程实现与测试能力
> 2. 深入理解量化交易平台架构
> 3. 提升量化分析能力

---

## 阶段一：工程基础（Engineering Foundation）

**目标：建立可维护的项目骨架**

- [x] 用 `uv` 初始化项目，写好 `pyproject.toml`（依赖分组：core / dev / viz）
- [x] 规划包结构：`quant/data/`, `quant/factor/`, `quant/backtest/`, `quant/risk/`, `quant/strategy/`
- [x] 配置 `ruff`（lint）+ `mypy`（类型检查），写 `Makefile` / `justfile` 统一命令
- [x] 写第一个 `CONTRIBUTING.md`，约定代码风格
- [x] 搭好 `pytest` 框架，加 `conftest.py` 和第一个空测试

---

## 阶段二：数据层（Data Layer）

**目标：理解数据获取、清洗、缓存的工程模式**

- [ ] 定义 `DataFetcher` Protocol（接口抽象），理解为什么用 Protocol 而非继承
- [ ] 实现 `AKShareAdapter`（A股）和 `YFinanceAdapter`（美股）两个适配器
- [ ] 用 DuckDB 做本地缓存层，理解读写分离、缓存失效策略
- [ ] 处理脏数据：复权、停牌、退市、涨跌停，写数据质量检查函数
- [ ] 为数据层写单元测试（mock 外部 API 调用）

**学习点：** Protocol vs ABC、数据管道设计、SQL on local files

---

## 阶段三：因子工程（Factor Engineering）

**目标：理解因子的本质与统计有效性**

- [ ] 实现基础因子：动量（1M/3M/12M）、波动率、换手率、价值类
- [ ] 学习并实现 IC（Information Coefficient）分析
- [ ] 计算 ICIR（IC / IC_std），理解它为什么衡量因子稳定性
- [ ] 做因子分层回测（五分位数），画收益分布图
- [ ] 实现多因子合成：等权、IC加权、PCA

**学习点：** 截面回归、因子暴露、多重共线性

---

## 阶段四：回测引擎（Backtesting Engine）

**目标：理解回测的核心逻辑与常见陷阱**

- [ ] 实现向量化回测引擎（先理解 vectorized 的局限，再对比 event-driven）
- [ ] 加入交易成本模型（双边手续费、冲击成本、持仓约束）
- [ ] 理解并处理**前视偏差（look-ahead bias）**
- [ ] 理解**生存者偏差（survivorship bias）**，思考如何在数据层规避
- [ ] 处理调仓频率、再平衡逻辑

**学习点：** 两大回测偏差、向量化 vs 事件驱动的取舍

---

## 阶段五：风险管理（Risk & Attribution）

**目标：能读懂并解释策略的风险来源**

- [ ] 实现完整风险指标：Sharpe / Sortino / Max Drawdown / Calmar / VaR / CVaR
- [ ] 理解 Beta / Alpha（CAPM框架），实现回归计算
- [ ] 实现 Brinson 归因（配置效应 vs 选股效应）
- [ ] 画 Drawdown 曲线，分析水下时间分布
- [ ] 写一份策略的风险报告（文字 + 数据）

**学习点：** CAPM 框架、尾部风险度量、归因分析方法论

---

## 阶段六：策略开发（Strategy Development）

**目标：从因子到可执行策略的全链路**

- [ ] **因子选股策略**：基于多因子打分的 Long-Only 组合
- [ ] **配对交易**：找协整股票对，实现均值回归策略（理解协整 vs 相关性）
- [ ] **ML Alpha**：用 LightGBM 预测截面收益，Walk-forward 验证（防止 data snooping）
- [ ] 对比三种策略在不同市场环境下的表现

**学习点：** 协整检验（ADF / Johansen）、Walk-forward vs 简单训练集划分

---

## 阶段七：可视化与 Dashboard

**目标：让结果可读、可演示**

- [ ] 用 Streamlit 搭 4 页面：回测结果 / 因子分析 / ML模型 / 风险报告
- [ ] 用 Plotly 画累计收益曲线、回撤图、因子 IC bar 图
- [ ] 加演示模式（无需联网，用内置数据）

---

## 阶段八：测试质量（Testing & Quality）

**目标：写真正有价值的测试，而不是形式主义**

- [ ] 区分单元测试 / 集成测试 / 属性测试（hypothesis）
- [ ] 对数学公式密集的模块（风险指标、IC计算）写数值正确性测试
- [ ] 用 `pytest-cov` 看覆盖率，找没被测到的边界条件
- [ ] 加 CI（GitHub Actions），每次 push 自动跑测试

> 测试不是最后才做的事——从阶段二开始就应该穿插进来。

---

## 进阶（可选，后续扩展）

- [ ] **实盘对接**：接入 IBKR API，理解 paper trading 流程
- [ ] **因子模型深化**：Barra 风险因子模型、行业中性化
- [ ] **替代数据**：新闻情感分析、财报文本 NLP
- [ ] **组合优化**：均值方差优化、Black-Litterman、风险平价
- [ ] **微观结构**：Level 2 数据、VWAP/TWAP 执行算法

---

## 推进顺序建议

```
阶段一 → 二 → 三 + 四（可并行）→ 五 → 六 → 七
                ↑
         阶段八穿插整个过程
```

每完成一个阶段，建议在 `docs/` 下写一篇简短的技术笔记，记录遇到的坑和决策理由。
