from typing import cast

import numpy as np
import pandas as pd

from quant.factor.earnings_quality import earnings_quality_pit, winsorize
from quant.factor.revenue_acceleration import revenue_acceleration_pit
from quant.factor.valuation_momentum import valuation_momentum

# ── 测试数据工厂 ──────────────────────────────────────────────────────────────

_PRICE_DATES = pd.date_range("2023-01-01", periods=100, name="date")
_SYMBOLS = ["A", "B", "C"]


def _make_fundamental(n_per_symbol: int = 8) -> pd.DataFrame:
    """模拟 fundamental_quarterly 表"""
    rows = []
    for sym in _SYMBOLS:
        for i in range(n_per_symbol):
            disclose = pd.Timestamp("2023-01-01") + pd.DateOffset(months=i * 3)
            rows.append(
                {
                    "symbol": sym,
                    "report_date": (disclose - pd.DateOffset(months=1)).strftime(
                        "%Y%m%d"
                    ),
                    "disclose_date": disclose,
                    "cfo_to_profit": np.random.uniform(0.3, 1.5),
                    "revenue_yoy": np.random.uniform(-0.2, 0.5),
                }
            )
    return pd.DataFrame(rows)


def _make_valuation(n_per_symbol: int = 100) -> pd.DataFrame:
    """模拟 valuation_daily 表"""
    rows = []
    for sym in _SYMBOLS:
        for i in range(n_per_symbol):
            rows.append(
                {
                    "symbol": sym,
                    "date": _PRICE_DATES[i],
                    "pe_ttm": 15.0 + i * 0.1,
                }
            )
    return pd.DataFrame(rows)


# ── winsorize ────────────────────────────────────────────────────────────────


def test_winsorize_clips_extremes():
    s = pd.Series([0.0, 1.0, 2.0, 3.0, 100.0])
    result = winsorize(s)
    assert result.max() < 100.0
    assert result.min() >= s.quantile(0.01)


def test_winsorize_preserves_length():
    s = pd.Series(range(100), dtype=float)
    assert len(winsorize(s)) == 100


# ── earnings_quality_pit ──────────────────────────────────────────────────────


def test_earnings_quality_pit_shape():
    fund = _make_fundamental()
    result = earnings_quality_pit(fund, _PRICE_DATES, _SYMBOLS)
    assert result.shape == (len(_PRICE_DATES), len(_SYMBOLS))
    assert result.index.name == "date"


def test_earnings_quality_pit_columns():
    fund = _make_fundamental()
    result = earnings_quality_pit(fund, _PRICE_DATES, _SYMBOLS)
    assert set(result.columns) == set(_SYMBOLS)


def test_earnings_quality_pit_no_lookahead():
    """第一个披露日之前应为 NaN（不用未来数据）"""
    fund = _make_fundamental()
    result = earnings_quality_pit(fund, _PRICE_DATES, _SYMBOLS)
    first_disclose = pd.Timestamp("2023-01-01")
    before = result[result.index < first_disclose]
    assert before.isnull().all().all()


def test_earnings_quality_pit_missing_symbol():
    """fundamental 中不含某 symbol 时，结果列不存在"""
    fund = _make_fundamental()
    fund = fund[fund["symbol"] != "C"]
    result = earnings_quality_pit(fund, _PRICE_DATES, _SYMBOLS)
    assert "C" not in result.columns


# ── revenue_acceleration_pit ──────────────────────────────────────────────────


def test_revenue_acceleration_pit_shape():
    fund = _make_fundamental()
    result = revenue_acceleration_pit(fund, _PRICE_DATES, _SYMBOLS)
    assert result.shape == (len(_PRICE_DATES), len(_SYMBOLS))
    assert result.index.name == "date"


def test_revenue_acceleration_pit_is_diff():
    """加速度 = 相邻两期增速之差，符号正负皆可"""
    fund = _make_fundamental()
    result = revenue_acceleration_pit(fund, _PRICE_DATES, _SYMBOLS)
    valid = result.dropna(how="all")
    assert not valid.empty


def test_revenue_acceleration_pit_single_row_skipped():
    """只有 1 条记录时 diff 无法计算，该 symbol 应被跳过（列不存在）"""
    fund = _make_fundamental()
    # 只保留 B 的第一条记录
    b_idx = fund[fund["symbol"] == "B"].index[:1]
    fund = pd.concat([fund[fund["symbol"] != "B"], fund.loc[b_idx]])
    result = revenue_acceleration_pit(fund, _PRICE_DATES, _SYMBOLS)
    assert "B" not in result.columns


# ── valuation_momentum ────────────────────────────────────────────────────────


def test_valuation_momentum_shape():
    val = _make_valuation()
    result = valuation_momentum(val, _PRICE_DATES, _SYMBOLS)
    assert result.shape == (len(_PRICE_DATES), len(_SYMBOLS))


def test_valuation_momentum_first_window_is_nan():
    """前 window 期应为 NaN（不够数据计算变化率）"""
    val = _make_valuation()
    window = 20
    result = valuation_momentum(val, _PRICE_DATES, _SYMBOLS, window=window)
    assert result.iloc[:window].isnull().all().all()


def test_valuation_momentum_negative_pe_excluded():
    """负 PE 应置为 NaN，symbol 被跳过（列不存在）"""
    val = _make_valuation()
    val.loc[val["symbol"] == "A", "pe_ttm"] = -10.0
    result = valuation_momentum(val, _PRICE_DATES, ["A"])
    assert "A" not in result.columns


def test_valuation_momentum_custom_window():
    val = _make_valuation()
    result_20 = valuation_momentum(val, _PRICE_DATES, _SYMBOLS, window=20)
    result_5 = valuation_momentum(val, _PRICE_DATES, _SYMBOLS, window=5)
    # 窗口越小，有效值越早出现
    first_valid_20 = cast(pd.Timestamp, result_20["A"].first_valid_index())
    first_valid_5 = cast(pd.Timestamp, result_5["A"].first_valid_index())
    assert first_valid_5 <= first_valid_20
