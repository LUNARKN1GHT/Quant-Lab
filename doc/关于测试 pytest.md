# 关于测试 pytest

## 全部测试

如果需要运行全部的测试函数，可以运行一下命令

```shell
uv run pytest --cov=quant --cov-report=term-missing --cov-report=html
```

- `--cov=` 用于**指定统计覆盖率的目标包/目录**。不加入任何参数就死活统计整个目录。
- 后面的 `--cov-report=` 用于说明报告输出的格式。
  - 如果是 `html`，测试会输出在 `htmlcov` 这个文件夹下面，打开里面的 `index.html` 即可查看完整的测试结果。
  - 如果是 `term`，则是在终端打印结果，后面跟的 `missing` 是额外显示那些行没有被覆盖（行号）。

一次运行就可以同时输出终端报告和 HTML 报告。
