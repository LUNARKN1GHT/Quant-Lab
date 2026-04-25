import pytest

from quant.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.data.universe == "csi300"
    assert cfg.factor.rsi_window == 14
    assert cfg.backtest.commission_rate == pytest.approx(0.0003)
    assert cfg.ml.n_estimators == 100
    assert cfg.advisor.target_vol == pytest.approx(0.15)
    assert cfg.regime.bull_scale == pytest.approx(1.0)
    assert cfg.regime.bear_scale == pytest.approx(0.2)


def test_config_from_yaml(tmp_path):
    yaml_content = """\
data:
  data_dir: data/test
  universe: test500
factor:
  rsi_window: 21
backtest:
  commission_rate: 0.001
  top_n: 10
ml:
  n_estimators: 50
regime:
  bull_scale: 0.9
advisor:
  target_vol: 0.2
"""
    p = tmp_path / "config.yaml"
    p.write_text(yaml_content)
    cfg = Config.from_yaml(p)

    assert cfg.data.universe == "test500"
    assert cfg.factor.rsi_window == 21
    assert cfg.backtest.commission_rate == pytest.approx(0.001)
    assert cfg.ml.n_estimators == 50
    assert cfg.regime.bull_scale == pytest.approx(0.9)
    assert cfg.advisor.target_vol == pytest.approx(0.2)


def test_config_from_yaml_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("{}")
    cfg = Config.from_yaml(p)
    # 空配置应全部走默认值
    assert cfg.data.universe == "csi300"
    assert cfg.factor.rsi_window == 14


def test_config_to_yaml_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "out.yaml"
    cfg.to_yaml(p)

    cfg2 = Config.from_yaml(p)
    assert cfg2.data.universe == cfg.data.universe
    assert cfg2.backtest.commission_rate == pytest.approx(cfg.backtest.commission_rate)
    assert cfg2.advisor.target_vol == pytest.approx(cfg.advisor.target_vol)
    assert cfg2.regime.bear_scale == pytest.approx(cfg.regime.bear_scale)


def test_config_from_yaml_partial(tmp_path):
    # 只设置部分字段，其余走默认
    p = tmp_path / "partial.yaml"
    p.write_text("factor:\n  rsi_window: 28\n")
    cfg = Config.from_yaml(p)
    assert cfg.factor.rsi_window == 28
    assert cfg.data.universe == "csi300"  # 默认值未被覆盖
