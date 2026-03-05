import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ensure workspace root on path like other tests
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.weather_market_imbalance import (
    _select_buy_cols,
    _SUFFIX_RE,
    WeatherMarketImbalanceStrategy,
)


def make_mock_data():
    # three timestamps
    idx = pd.date_range("2026-01-01", periods=3, freq="h")
    prices = pd.DataFrame({"A": [10.0, 9.5, 9.0], "B": [5.0, 5.1, 5.2]}, index=idx)

    # buy volume has yes/no columns for A only
    buy = pd.DataFrame(
        {
            # first bar negative, second bar also negative to allow entry at idx=1
            "A__yes": [100, 50, 60],
            "A__no": [120, 80, 50],
            # B missing so should produce nan
        },
        index=idx,
    )
    return prices, buy


def test_select_buy_cols_basic():
    # existing yes/no names should be returned exactly
    df = pd.DataFrame(columns=["foo__yes", "foo__no", "bar"])
    no, yes = _select_buy_cols(df, "foo", "foo__yes", _SUFFIX_RE)
    assert no == "foo__no" and yes == "foo__yes"

    # suffix case where base column is present without explicit yes/no
    df2 = pd.DataFrame(columns=["foo"])
    no, yes = _select_buy_cols(df2, "foo", "foo__no", _SUFFIX_RE)
    assert no == "foo" and yes is None


def test_buy_delta_filtering():
    prices, buy = make_mock_data()

    strat = WeatherMarketImbalanceStrategy(
        buy_volume=buy,
        use_market_regime=False,
        use_vwap_slope_filter=False,
        use_vwap_volume_imbalance_filter=False,
        lookback_hours=None,  # small sample, disable multi-hour lookback
        lookback=2,          # allow at least two-bar window
    )

    # at index 1 cumulative delta = (-20) + (-30) = -50 -> entry allowed
    w1 = strat.generate_weights(prices, prices.pct_change(), 1)
    # both markets should be candidates now (B no longer blocked)
    assert w1[0] < 0 and w1[1] < 0

    # index 2 cumulative delta = (-20) + (-30) + 10 = -40 -> still allowed,
    # all markets
    w2 = strat.generate_weights(prices, prices.pct_change(), 2)
    assert w2[0] < 0 and w2[1] < 0


def test_delta_observed_before_regime_window():
    # use a nonzero lookback_hours to exercise the new logic: delta_history
    # should record values even when the regime window isn't ready.
    prices, buy = make_mock_data()
    strat2 = WeatherMarketImbalanceStrategy(
        buy_volume=buy,
        use_market_regime=True,
        lookback_hours=6.0,  # forces start==index for first 72 bars
        lookback=1,
    )
    # simulate three bars
    for idx in range(3):
        _ = strat2.generate_weights(prices, prices.pct_change(), idx)
    # delta_history should contain three entries per asset even though no
    # positions were opened (regime not ready)
    assert len(strat2.delta_history) == 6
    # the plotting helper shifts the series so the first value for A is zero
    deltas_A = [d for (i, a, d) in strat2.delta_history if a == "A"]
    assert deltas_A[0] == 0.0
    # second A reading should be (-20 + -30) - (-20) = -30
    assert deltas_A[1] == -30.0


def test_lookback_hours_ignored_when_regime_disabled():
    # when use_market_regime=False, the lookback_hours value should not delay
    # the buy-volume gate; window-length requirement still applies.
    prices, buy = make_mock_data()
    strat3 = WeatherMarketImbalanceStrategy(
        buy_volume=buy,
        use_market_regime=False,
        use_vwap_slope_filter=False,
        use_vwap_volume_imbalance_filter=False,
        lookback_hours=6.0,
        lookback=1,
    )
    # at index 1 the asset_window len == 2 so entry is allowed; delta is negative
    w = strat3.generate_weights(prices, prices.pct_change(), 1)
    assert w[0] < 0 and w[1] < 0




def make_fake_volumes():
    idx = pd.date_range("2026-02-20", periods=10, freq="5min")
    # create yes and no columns for a market
    buy = pd.DataFrame({"mkt__yes": [10, 5, 0, 8, 2, 0, 0, 3, 1, 0]}, index=idx)
    sell = pd.DataFrame({"mkt__no": [0, 0, 5, 1, 2, 0, 0, 0, 4, 0]}, index=idx)
    return buy, sell


def test_buy_delta_negative_and_positive():
    buy, sell = make_fake_volumes()
    prices = pd.DataFrame({"mkt": np.arange(len(buy))}, index=buy.index)
    combined = pd.concat([buy, sell], axis=1)
    strat = WeatherMarketImbalanceStrategy(buy_volume=combined)
    # with baseline shift the first cumulative value is zero
    assert strat._buy_delta("mkt", prices, 0) == 0.0
    # third bar raw cum = 10 + 5 - 5 = 10, shifted -> 0
    assert strat._buy_delta("mkt", prices, 2) == 0.0


def test_delta_gate_blocks_positive():
    idx = pd.date_range("2026-02-20", periods=3, freq="min")
    prices = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]}, index=idx)
    buy = pd.DataFrame({
        "a__yes": [0, 0, 0],
        "a__no":  [0, 1, 0],
        "b__yes": [0, 1, 0],
        "b__no":  [0, 0, 0],
    }, index=idx)
    strat = WeatherMarketImbalanceStrategy(
        buy_volume=buy,
        lookback_hours=None,
        lookback=2,
        use_market_regime=False,
        use_vwap_slope_filter=False,
        use_vwap_volume_imbalance_filter=False,
    )
    # at time 1 the delta for a is -1 (should be kept), for b is +1 (should be skipped)
    weights = strat.generate_weights(prices, prices.pct_change(), 1)
    assert weights[0] < 0 and weights[1] == 0


def test_buy_delta_basic():
    idx = pd.date_range("2026-02-20", periods=4, freq="min")
    buy = pd.DataFrame({
        "mkt__yes": [0, 1, 0, 2],
        "mkt__no":  [0, 0, 3, 1],
    }, index=idx)
    prices = pd.DataFrame({"mkt": [1, 1, 1, 1]}, index=idx)
    strat = WeatherMarketImbalanceStrategy(buy_volume=buy)
    # at index1 the cumulative delta is 1, at index2 the running sum drops
    # to -2 (0 + 1 - 3)
    assert strat._buy_delta("mkt", prices, 1) == 1.0
    assert strat._buy_delta("mkt", prices, 2) == -2.0


def test_no_buy_volume_does_not_block():
    # an empty or None buy-volume matrix should allow entries; delta_history
    # will contain NaN but gate ignores it.
    idx2 = pd.date_range("2026-01-01", periods=3, freq="h")
    prices2 = pd.DataFrame({"A": [1, 1, 1]}, index=idx2)
    # use an empty DataFrame
    strat3 = WeatherMarketImbalanceStrategy(
        buy_volume=pd.DataFrame(),
        use_market_regime=False,
        use_vwap_slope_filter=False,
        use_vwap_volume_imbalance_filter=False,
        lookback_hours=None,
        lookback=2,  # need two bars for entry
    )
    w4 = strat3.generate_weights(prices2, prices2.pct_change(), 1)
    # should produce a short weight because there are no gates
    assert w4[0] < 0
    assert len(strat3.delta_history) == 1
    assert pd.isna(strat3.delta_history[0][2])

    # same behaviour when buy_volume is None explicitly
    strat4 = WeatherMarketImbalanceStrategy(
        buy_volume=None,
        use_market_regime=False,
        use_vwap_slope_filter=False,
        use_vwap_volume_imbalance_filter=False,
        lookback_hours=None,
        lookback=2,
    )
    w5 = strat4.generate_weights(prices2, prices2.pct_change(), 1)
    assert w5[0] < 0
    assert len(strat4.delta_history) == 1
    assert pd.isna(strat4.delta_history[0][2])

if __name__ == "__main__":
    test_select_buy_cols_basic()
    test_buy_delta_filtering()
    print("\n✅ buy-volume delta tests passed")
