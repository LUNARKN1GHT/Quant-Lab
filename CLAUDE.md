# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 常用命令

本项目使用 `uv` 作为包管理器，Python 版本为 3.11。

```bash
# 运行全部测试（含覆盖率）
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_akshare.py

# 运行单个测试函数
uv run pytest tests/test_akshare.py::test_fetch_returns_dataframe

# 代码检查
uv run ruff check quant tests

# 自动修复 lint 问题
uv run ruff check --fix quant tests

# 类型检查
uv run mypy quant
```

## 架构概览

Quant-Lab 是一个按阶段迭代构建的模块化量化交易平台，主包为 `quant/`。

### 当前进度（第一阶段已完成）

**数据层**（`quant/data/`）：

- `base.py` — 定义 `DataFetcher` Protocol（接口），所有适配器须遵循此协议。
- `akshare_adapter.py` — 通过 `akshare` 获取 A 股历史 OHLCV 数据。
- `yfinance_adapter.py` — 通过 `yfinance` 获取美股历史 OHLCV 数据。
- 两个适配器均将输出标准化为统一 schema：DatetimeIndex + 标准列名。

### 规划中的模块

详见 `ROADMAP.md`：

- `quant/data/` — DuckDB 本地缓存、数据质量检查（第二阶段）
- `quant/factor/` — 技术指标、IC 分析、多因子合成（第三阶段）
- `quant/backtest/` — 向量化 + 事件驱动回测引擎（第四阶段）
- `quant/risk/` — Sharpe/Sortino/最大回撤、Beta/Alpha、归因分析（第五阶段）
- `quant/strategy/` — 因子选股、配对交易、ML Alpha（第六阶段）

### 关键设计决策

- **Protocol 适配器模式**：`DataFetcher` 使用 `typing.Protocol`，适配器无需继承即可实现鸭子类型。
- **统一数据 schema**：所有适配器输出格式一致，下游代码与数据源解耦。
- **DuckDB**：第二阶段本地数据缓存的选型数据库。
- **Streamlit + Plotly**：第七阶段可视化看板的技术选型。

### 行长度

统一使用 88 字符，配置在 `pyproject.toml` 的 `[tool.ruff]` 中。
