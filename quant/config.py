from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/csi300"
    universe: str = "csi300"


@dataclass
class FactorConfig:
    momentum_windows: list[int] = field(default_factory=lambda: [20, 60, 120])
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ic_forward_window: int = 5
    n_groups: int = 5


@dataclass
class BacktestConfig:
    commission_rate: float = 0.0003
    rebalance_freq: Literal["ME", "D", "W"] = "ME"
    top_n: int = 20
    train_window: int = 240
    predict_window: int = 20


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls(
            data=DataConfig(**raw.get("data", {})),
            factor=FactorConfig(**raw.get("factor", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
