import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from quant.config import Config
from scripts.backtest_ml import run, run_stack

cfg = Config.from_yaml(Path(__file__).parent.parent / "configs/default.yaml")

MODELS = {
    "LightGBM": LGBMRegressor(n_estimators=cfg.ml.n_estimators, verbosity=-1),
    "RandomForest": RandomForestRegressor(
        n_estimators=cfg.ml.n_estimators, n_jobs=-1, random_state=42
    ),
    "Ridge": Ridge(alpha=cfg.ml.ridge_alpha),
    "XGBoost": XGBRegressor(n_estimators=cfg.ml.n_estimators, verbosity=0),
}

results = {}
for name, model in MODELS.items():
    print(f"\n{'=' * 50}")
    print(f"模型：{name}")
    results[name] = run(model=model)

print(f"\n{'=' * 50}")
print("模型：Stacking（LightGBM + RandomForest + XGBoost + Ridge）")
results["Stacking"] = run_stack(
    base_models=list(MODELS.values()),
    meta_model=Ridge(alpha=cfg.ml.ridge_alpha),
)

# 汇总对比
print(f"\n{'=' * 50}")
print(f"{'模型':<16} {'年化收益':>9} {'Sharpe':>8} {'最大回撤':>9} {'Calmar':>8}")
print("=" * 50)
for name, r in results.items():
    print(
        f"{name:<16} {r['ann_ret']:>8.2%} {r['sharpe']:>8.3f}"
        f" {r['max_drawdown']:>8.2%} {r['calmar']:>8.3f}"
    )
