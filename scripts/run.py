import os
import sys
from datetime import datetime
from pathlib import Path

# 程序环境配置
os.environ["no_proxy"] = "*"
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant.pipeline import run_pipeline

result = run_pipeline(
    symbols=["600519", "600036", "601318", "000333", "000858"],
    start=datetime(2022, 1, 1),
    end=datetime(2024, 12, 31),
)

print("风险指标：")
for k, v in result["metrics"].items():
    print(f"  {k}: {v:.4f}")
