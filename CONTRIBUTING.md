# Contributing

感谢你对 Quant-Lab 的关注！

## 项目定位

这是一个**个人学习与展示项目**，用于探索量化交易系统的模块化设计与实现。目前由个人独立维护。

## 欢迎的贡献形式

- **Issue**：发现 Bug、有功能建议、或对某个设计有疑问，欢迎开 Issue 讨论
- **建议**：对架构设计、技术选型有想法，可以在 Issue 中提出

## 本地开发

本项目使用 `uv` 管理依赖，Python 版本为 3.11。

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest

# 代码检查
uv run ruff check quant tests

# 类型检查
uv run mypy quant
```

## 开发规范

- 行长度限制：88 字符（Ruff 配置）
- 提交前清确保测试全部通过
- 遵循现有的模块结构与命名风格
