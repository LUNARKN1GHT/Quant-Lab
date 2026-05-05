# Quant-Lab

> 个人量化研究平台，覆盖数据获取、因子研究、策略回测、风险分析与可视化看板的完整链路。

## 功能亮点

- **多因子研究**：15+ 技术/基本面因子，支持 IC/ICIR 评价与五分位分层回测
- **统计套利**：协整检验、OU 过程拟合、Kalman Filter 动态对冲、PCA 篮子套利
- **市场中性**：Beta 中性 + 行业中性组合构建
- **基金管理**：交易流水记录、持仓动态计算、自选追踪、技术信号扫描
- **可视化看板**：10 个 Streamlit 页面，覆盖行情、因子、策略、持仓全链路

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
│   ├── factor/               # 因子库（动量/RSI/MACD 等 15+ 因子）
│   ├── strategy/             # 策略（统计套利/市场中性/ML Alpha）
│   ├── backtest/             # 回测引擎
│   ├── risk/                 # 风险指标与归因分析
│   ├── fund/                 # 基金管理（持仓流水/自选追踪）
│   ├── regime/               # 市场状态检测
│   ├── sector/               # 行业轮动
│   └── macro/                # 宏观因子
├── dashboard/                # Streamlit 可视化看板（10 个页面）
├── scripts/                  # 研究脚本与数据下载
├── tests/                    # 单元测试
└── configs/                  # 配置文件（持仓/自选列表）
```

## 看板页面

| 页面 | 功能 |
| :------: | :------: |
| 数据管理 | 行情下载与数据库管理 |
| 持仓建议 | 基于因子的选股信号 |
| 市场状态 | 牛熊/震荡市识别 |
| 回测对比 | 多策略回测结果对比 |
| 因子分析 | IC/ICIR 分析与五分位分层 |
| 参数优化 | 策略参数网格搜索 |
| 行业轮动 | 申万行业强弱分析 |
| 宏观因子 | CPI/PMI/M2 与市场关系 |
| 因子研究（高级） | 因子 IC 衰减、市场环境分解 |
| 统计套利 | 协整/OU/Kalman/PCA 套利研究 |
| 我的持仓 | 基金持仓管理与收益分析 |

## 技术栈

- **数据**：AKShare、DuckDB、pandas
- **因子/策略**：numpy、scikit-learn、statsmodels
- **可视化**：Streamlit、Plotly
- **工程**：uv、ruff、mypy、pytest

## 数据说明

本项目使用 [AKShare](https://akshare.akfamily.xyz/) 获取 A 股公开数据，数据存储于本地 DuckDB，不包含付费数据源。

## 免责声明

本项目仅用于学习和研究目的，不构成任何投资建议。
