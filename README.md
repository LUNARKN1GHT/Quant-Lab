# Quant-Lab

> 个人量化研究平台，覆盖数据获取、因子研究、策略回测、风险分析与可视化看板的完整链路。

## 功能亮点

- **多因子研究**：15+ 技术/基本面因子，IC/ICIR 评价与五分位分层回测，自主因子研究流程
- **统计套利**：协整检验（EG/Johansen）、OU 过程拟合、Kalman Filter 动态对冲、PCA 篮子套利
- **市场中性**：Beta 中性 + 行业中性组合构建
- **仓位引擎**：Regime × 波动率 × 宏观 × 行业四路信号加权融合，权重在线可调
- **组合优化**：均值方差（MVO）/ 风险平价 / Black-Litterman，与实盘持仓实时对账
- **基金管理**：交易流水记录、持仓动态计算、定投计划、技术信号扫描
- **可视化看板**：8 个 Streamlit 页面，覆盖行情、因子、策略、持仓全链路

## 快速开始

**环境要求**：Python 3.11，使用 `uv` 管理依赖

```bash
# 安装依赖
uv sync

# 启动看板
uv run streamlit run dashboard/app.py

# 下载行情数据
uv run python scripts/download_data.py

# 运行测试
uv run pytest
```

## 项目结构

```txt
Quant-Lab/
├── quant/                    # 核心库
│   ├── data/                 # 数据层（AKShare 适配器 + DuckDB 缓存）
│   ├── factor/               # 因子库（动量/RSI/MACD/特质波动率等 15+ 因子）
│   ├── strategy/             # 策略（统计套利/市场中性/ML Alpha）
│   ├── backtest/             # 回测引擎（向量化，含手续费模型）
│   ├── risk/                 # 风险指标与归因分析
│   ├── portfolio/            # 组合优化（MVO/风险平价/BL）
│   ├── fund/                 # 基金管理（持仓流水/组合优化/定投计划）
│   ├── advisor/              # 仓位建议引擎
│   ├── regime/               # 市场状态检测
│   ├── sector/               # 行业轮动
│   └── macro/                # 宏观因子
├── dashboard/                # Streamlit 可视化看板（8 个页面）
├── scripts/                  # 研究脚本与数据下载
├── tests/                    # 单元测试
└── configs/                  # 配置文件
```

## 看板页面

| 页面     | 功能                                      |
| :------- | :---------------------------------------- |
| 数据中心 | 行情下载、DuckDB 管理、数据健康监控       |
| 市场环境 | Regime 历史、行业轮动热力图、宏观景气合成 |
| 因子工坊 | IC/ICIR 分析、参数调优、自主因子研究      |
| 策略库   | 多因子回测、统计套利研究、组合优化对比    |
| 仓位建议 | 四路信号权重可调、历史仓位走势            |
| 我的持仓 | 基金持仓管理、组合优化对账、定投计划      |
| 风险报告 | VaR/CVaR、Beta/Alpha、回撤分布与归因      |
| 今日报告 | 一键汇总当日信号、行业强弱、建议仓位      |

## 技术栈

- **数据**：AKShare、DuckDB、pandas
- **因子/策略**：numpy、scikit-learn、statsmodels、cvxpy
- **可视化**：Streamlit、Plotly
- **工程**：uv、ruff、mypy、pytest、GitHub Actions

## 数据说明

本项目使用 [AKShare](https://akshare.akfamily.xyz/) 获取 A 股公开数据，存储于本地 DuckDB，不依赖付费数据源。

## 免责声明

本项目仅用于学习和研究目的，不构成任何投资建议。
