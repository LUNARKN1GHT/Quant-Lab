"""全局配置模块

使用 Python dataclass 定义各模块的参数，支持从 YAML 文件加载和序列化回文件。
各子配置类作为字段嵌套在 Config 中，通过 Config.from_yaml() 统一读取，
避免散落在各处的魔法数字（magic number）。
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class DataConfig:
    """数据源配置"""

    data_dir: str = "data/csi300"
    """收盘价 CSV 文件目录"""

    universe: str = "csi300"
    """股票池标识"""


@dataclass
class FactorConfig:
    """因子计算参数"""

    momentum_windows: list[int] = field(default_factory=lambda: [20, 60, 120])
    """动量因子的多个回看窗口（月/季/半年），对应 20/60/120 交易日"""

    rsi_window: int = 14
    """RSI 计算窗口，Wilder 原始默认值"""

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    ic_forward_window: int = 5
    """IC 计算时的前向收益窗口（交易日）"""

    n_groups: int = 5
    """分层回测的分位数组数"""


@dataclass
class BacktestConfig:
    """回测参数"""

    commission_rate: float = 0.0003
    """单边手续费率，默认万三"""

    rebalance_freq: Literal["ME", "D", "W"] = "ME"
    """调仓频率：ME=月末，D=每日，W=每周"""

    top_n: int = 20
    """每期选股数量"""

    train_window: int = 240
    """ML 模型训练窗口（交易日），约一年"""

    predict_window: int = 20
    """每次预测/调仓的步长（交易日），约一个月"""


@dataclass
class MLConfig:
    """机器学习模型参数"""

    n_estimators: int = 100
    """RandomForest / GradientBoosting 的决策树数量"""

    ridge_alpha: float = 1.0
    """Ridge 回归的正则化强度，值越大惩罚越强"""

    holdout_ratio: float = 0.2
    """Stacking 中留出用于生成元特征的比例（不参与基模型训练）"""


@dataclass
class AdvisorConfig:
    """仓位建议参数（波动率目标法）"""

    target_vol: float = 0.15
    """目标年化波动率，超过时自动降仓"""

    vol_window: int = 20
    """实现波动率的滚动计算窗口（交易日）"""

    max_position: float = 1.0
    """仓位上限，防止过度杠杆"""

    min_position: float = 0.0
    """仓位下限，防止做空"""


@dataclass
class RegimeConfig:
    """市场环境检测参数"""

    ma_window: int = 120
    """趋势判断用的长期均线窗口（约半年）"""

    breadth_window: int = 20
    """市场宽度：统计 N 日内上涨股票占比的窗口"""

    vol_short: int = 20
    """恐慌检测用的短期波动率窗口"""

    vol_long: int = 60
    """恐慌检测用的长期波动率窗口（基准）"""

    bull_scale: float = 1.0
    """牛市时的仓位倍数"""

    range_scale: float = 0.6
    """震荡市时的仓位倍数"""

    bear_scale: float = 0.2
    """熊市时的仓位倍数"""


@dataclass
class Config:
    """全局配置根对象，聚合所有子配置。

    用法：
        cfg = Config()                           # 使用所有默认值
        cfg = Config.from_yaml("configs/x.yaml") # 从 YAML 文件加载
        cfg.to_yaml("configs/x.yaml")            # 序列化回 YAML
    """

    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    advisor: AdvisorConfig = field(default_factory=AdvisorConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """从 YAML 文件加载配置，缺失的字段自动使用 dataclass 默认值。"""
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            factor=FactorConfig(**raw.get("factor", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            ml=MLConfig(**raw.get("ml", {})),
            regime=RegimeConfig(**raw.get("regime", {})),
            advisor=AdvisorConfig(**raw.get("advisor", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """将当前配置序列化为 YAML 文件，allow_unicode=True 保留中文字段名。"""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
