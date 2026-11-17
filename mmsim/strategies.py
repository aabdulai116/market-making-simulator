"""Market making quoting strategies.

Each strategy looks at the current market state and returns the bid and ask
prices (in ticks) it wants to quote. Returning None for a side means "don't
quote that side right now".
"""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuoteState:
    mid: float               # current mid price in ticks
    best_bid: int
    best_ask: int
    inventory: int           # current position in lots (positive = long)
    sigma: float             # estimated mid price volatility, ticks per step
    steps_left: int          # steps until the end of the trading day


@dataclass
class Quote:
    bid: Optional[int]
    ask: Optional[int]


def _respect_limits(bid, ask, inventory, max_inventory, quote_size):
    """Stop quoting the side that would push inventory past the limit."""
    if inventory + quote_size > max_inventory:
        bid = None
    if inventory - quote_size < -max_inventory:
        ask = None
    return Quote(bid, ask)


class FixedSpreadMaker:
    """Baseline: quote a fixed distance either side of mid, ignore inventory."""

    def __init__(self, half_spread=2.0, quote_size=5, max_inventory=60):
        self.name = "Fixed spread"
        self.half_spread = half_spread
        self.quote_size = quote_size
        self.max_inventory = max_inventory

    def quote(self, state: QuoteState) -> Quote:
        bid = math.floor(state.mid - self.half_spread)
        ask = math.ceil(state.mid + self.half_spread)
        return _respect_limits(
            bid, ask, state.inventory, self.max_inventory, self.quote_size
        )


class InventorySkewMaker:
    """Avellaneda and Stoikov (2008) style market maker.

    The maker prices around a reservation price instead of the mid. When it is
    long, the reservation price drops below mid, so its bid becomes less
    attractive and its ask more attractive. That pulls inventory back to zero.

        reservation = mid - inventory * gamma * sigma^2 * horizon
        half_spread = 0.5 * gamma * sigma^2 * horizon + (1 / gamma) * ln(1 + gamma / k)

    gamma is risk aversion and k describes how quickly fill probability falls
    as quotes move away from mid. The original model uses time left until the
    close as the horizon. Here the horizon is a rolling window, capped by the
    time left in the day, which keeps quotes sensible early in the session.
    """

    def __init__(
        self,
        gamma=0.02,
        k=0.6,
        horizon=60,
        quote_size=5,
        max_inventory=60,
        min_half_spread=1.0,
    ):
        self.name = "Inventory skew (A-S)"
        self.gamma = gamma
        self.k = k
        self.horizon = horizon
        self.quote_size = quote_size
        self.max_inventory = max_inventory
        self.min_half_spread = min_half_spread

    def quote(self, state: QuoteState) -> Quote:
        horizon = min(self.horizon, max(state.steps_left, 1))
        variance_term = self.gamma * (state.sigma ** 2) * horizon

        reservation = state.mid - state.inventory * variance_term
        half_spread = 0.5 * variance_term
        half_spread += (1.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        half_spread = max(half_spread, self.min_half_spread)

        bid = math.floor(reservation - half_spread)
        ask = math.ceil(reservation + half_spread)
        return _respect_limits(
            bid, ask, state.inventory, self.max_inventory, self.quote_size
        )
