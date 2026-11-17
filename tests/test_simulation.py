import numpy as np

from mmsim.market import MarketConfig, simulate_day
from mmsim.strategies import FixedSpreadMaker, InventorySkewMaker, QuoteState

SHORT_DAY = MarketConfig(steps_per_day=500)


def test_day_ends_flat_and_is_reproducible():
    first = simulate_day(InventorySkewMaker(), SHORT_DAY, seed=3)
    second = simulate_day(InventorySkewMaker(), SHORT_DAY, seed=3)
    assert first.pnl == second.pnl
    assert first.pnl_path[-1] == first.pnl
    assert first.fills > 0


def test_inventory_limit_is_respected():
    strategy = FixedSpreadMaker(max_inventory=20)
    result = simulate_day(strategy, SHORT_DAY, seed=1)
    # A single fill can be at most one quote in size past the check.
    assert np.max(np.abs(result.inventory_path)) <= 20 + strategy.quote_size


def test_skew_moves_quotes_against_inventory():
    strategy = InventorySkewMaker(gamma=0.02, horizon=60)
    flat = QuoteState(mid=10_000, best_bid=9_999, best_ask=10_001,
                      inventory=0, sigma=1.0, steps_left=1_000)
    long = QuoteState(mid=10_000, best_bid=9_999, best_ask=10_001,
                      inventory=20, sigma=1.0, steps_left=1_000)
    flat_quote = strategy.quote(flat)
    long_quote = strategy.quote(long)
    assert long_quote.ask < flat_quote.ask
    assert long_quote.bid < flat_quote.bid

