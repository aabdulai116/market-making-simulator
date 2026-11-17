"""Agent based market simulation.

Each trading day has a hidden "fundamental" value that moves as a random walk
with occasional jumps. Four kinds of participants interact through one
limit order book:

* Liquidity providers post and cancel passive limit orders near their
  (noisy) view of value. They keep the book populated.
* Noise traders send random market orders. This is the flow a market maker
  earns the spread from.
* Informed traders know the fundamental and only trade when the book is
  mispriced. This is the flow that causes adverse selection.
* Our market maker, which runs one of the strategies in strategies.py.

Units: prices are in ticks of $0.01 and quantities are in lots of 100
shares, so one tick on one lot is exactly $1.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .orderbook import BUY, SELL, OrderBook
from .strategies import QuoteState

MM = "MM"
DOLLARS_PER_TICK_LOT = 1.0


@dataclass
class MarketConfig:
    steps_per_day: int = 4680          # 6.5 hour session in 5 second steps
    start_price: int = 10_000          # $100.00 in ticks
    fundamental_vol: float = 0.9       # ticks per step
    jump_prob: float = 0.0008          # chance of a news jump each step
    jump_size: float = 12.0            # std dev of a jump, ticks
    lp_rate: float = 3.0               # liquidity provider orders per step
    lp_view_noise: float = 1.5         # how noisy LP value estimates are
    lp_order_life: float = 40.0        # mean steps before an LP order is pulled
    noise_rate: float = 0.6            # noise trader market orders per step
    informed_rate: float = 0.06        # informed trader arrivals per step
    informed_edge: float = 2.0         # mispricing (ticks) informed traders need
    max_taker_size: int = 8
    markout_steps: int = 12            # one minute markout
    vol_ewma: float = 0.02             # decay for the volatility estimate


@dataclass
class DayResult:
    pnl: float
    pnl_path: np.ndarray
    inventory_path: np.ndarray
    mid_path: np.ndarray
    fundamental_path: np.ndarray
    bid_path: np.ndarray
    ask_path: np.ndarray
    fills: int
    volume: int
    spread_capture: float
    markout: float
    flatten_cost: float
    max_abs_inventory: int
    mean_abs_inventory: float


@dataclass
class MarketMakerAccount:
    cash: float = 0.0
    inventory: int = 0
    bid_id: int = None
    ask_id: int = None
    bid_price: int = None
    ask_price: int = None
    fills: List[tuple] = field(default_factory=list)   # (step, side, price, qty)


def simulate_day(strategy, config: MarketConfig, seed: int) -> DayResult:
    """Run one trading day and return the market maker's results.

    The seed fixes everything outside the market maker's control, so two
    strategies run with the same seed face the same news and order arrivals.
    """
    rng = np.random.default_rng(seed)
    book = OrderBook()
    account = MarketMakerAccount()
    steps = config.steps_per_day

    fundamental = float(config.start_price)
    lp_expiry: Dict[int, int] = {}
    _seed_book(book, rng, config, lp_expiry)

    sigma_sq = config.fundamental_vol ** 2
    last_mid = book.mid()

    pnl_path = np.zeros(steps)
    inventory_path = np.zeros(steps, dtype=int)
    mid_path = np.zeros(steps)
    fundamental_path = np.zeros(steps)
    bid_path = np.full(steps, np.nan)
    ask_path = np.full(steps, np.nan)
    spread_capture = 0.0

    for step in range(steps):
        # 1. News: the hidden value moves.
        fundamental += config.fundamental_vol * rng.standard_normal()
        if rng.random() < config.jump_prob:
            fundamental += config.jump_size * rng.standard_normal()

        # 2. Liquidity providers pull stale orders and post new ones.
        _expire_lp_orders(book, lp_expiry, step)
        fills = []
        for _ in range(rng.poisson(config.lp_rate)):
            fills += _post_lp_order(book, rng, config, fundamental, lp_expiry, step)
        _ensure_two_sided(book, fundamental, lp_expiry, step, config)

        # 3. Market maker updates its quotes.
        mid = book.mid()
        change = mid - last_mid
        sigma_sq = (1 - config.vol_ewma) * sigma_sq + config.vol_ewma * change ** 2
        last_mid = mid

        state = QuoteState(
            mid=mid,
            best_bid=book.best_bid(),
            best_ask=book.best_ask(),
            inventory=account.inventory,
            sigma=math.sqrt(sigma_sq),
            steps_left=steps - step,
        )
        quote = strategy.quote(state)
        fills += _update_quotes(book, account, quote, strategy.quote_size)

        # 4. Takers arrive in random order.
        takers = ["noise"] * rng.poisson(config.noise_rate)
        takers += ["informed"] * rng.poisson(config.informed_rate)
        rng.shuffle(takers)
        for kind in takers:
            size = int(rng.integers(1, config.max_taker_size + 1))
            if kind == "noise":
                side = BUY if rng.random() < 0.5 else SELL
                fills += book.market("noise", side, size)
            else:
                if fundamental - book.best_ask() > config.informed_edge:
                    fills += book.market("informed", BUY, size)
                elif book.best_bid() - fundamental > config.informed_edge:
                    fills += book.market("informed", SELL, size)
            _ensure_two_sided(book, fundamental, lp_expiry, step, config)

        # 5. Book keeping for the market maker.
        mid_now = book.mid()
        for fill in fills:
            spread_capture += _apply_fill(account, fill, step, mid)
        _sync_quote_ids(book, account)

        pnl_path[step] = account.cash + account.inventory * mid_now
        inventory_path[step] = account.inventory
        mid_path[step] = mid_now
        fundamental_path[step] = fundamental
        if account.bid_id is not None:
            bid_path[step] = account.bid_price
        if account.ask_id is not None:
            ask_path[step] = account.ask_price

    # End of day: cancel quotes and flatten inventory at market.
    pnl_before_flatten = account.cash + account.inventory * book.mid()
    _cancel_quotes(book, account)
    _flatten(book, account)
    final_pnl = account.cash
    flatten_cost = pnl_before_flatten - final_pnl
    pnl_path[-1] = final_pnl

    markout = _markout(account.fills, mid_path, config.markout_steps)
    volume = sum(qty for _, _, _, qty in account.fills)

    return DayResult(
        pnl=final_pnl * DOLLARS_PER_TICK_LOT,
        pnl_path=pnl_path * DOLLARS_PER_TICK_LOT,
        inventory_path=inventory_path,
        mid_path=mid_path,
        fundamental_path=fundamental_path,
        bid_path=bid_path,
        ask_path=ask_path,
        fills=len(account.fills),
        volume=volume,
        spread_capture=spread_capture * DOLLARS_PER_TICK_LOT,
        markout=markout * DOLLARS_PER_TICK_LOT,
        flatten_cost=flatten_cost * DOLLARS_PER_TICK_LOT,
        max_abs_inventory=int(np.max(np.abs(inventory_path))),
        mean_abs_inventory=float(np.mean(np.abs(inventory_path))),
    )


# ----------------------------------------------------------------------
# Liquidity providers
# ----------------------------------------------------------------------
def _seed_book(book, rng, config, lp_expiry):
    for offset in range(1, 11):
        for side in (BUY, SELL):
            price = config.start_price - side * offset
            size = int(rng.integers(2, 10))
            order_id, _ = book.add_limit("lp", side, price, size)
            lp_expiry[order_id] = int(rng.exponential(config.lp_order_life))


def _post_lp_order(book, rng, config, fundamental, lp_expiry, step):
    view = fundamental + config.lp_view_noise * rng.standard_normal()
    side = BUY if rng.random() < 0.5 else SELL
    offset = 1 + rng.geometric(0.35)
    if side == BUY:
        price = math.floor(view) - offset + 1
    else:
        price = math.ceil(view) + offset - 1
    size = int(rng.integers(1, 10))
    order_id, fills = book.add_limit("lp", side, price, size)
    if order_id is not None:
        lp_expiry[order_id] = step + int(rng.exponential(config.lp_order_life)) + 1
    return fills


def _expire_lp_orders(book, lp_expiry, step):
    expired = [oid for oid, expiry in lp_expiry.items() if expiry <= step]
    for order_id in expired:
        book.cancel(order_id)
        del lp_expiry[order_id]
    # Forget orders that were already filled.
    filled = [oid for oid in lp_expiry if oid not in book.orders]
    for order_id in filled:
        del lp_expiry[order_id]


def _ensure_two_sided(book, fundamental, lp_expiry, step, config):
    """A backstop so the book never goes empty on one side."""
    if book.best_bid() is None:
        order_id, _ = book.add_limit("lp", BUY, math.floor(fundamental) - 3, 5)
        lp_expiry[order_id] = step + int(config.lp_order_life)
    if book.best_ask() is None:
        order_id, _ = book.add_limit("lp", SELL, math.ceil(fundamental) + 3, 5)
        lp_expiry[order_id] = step + int(config.lp_order_life)


# ----------------------------------------------------------------------
# Market maker plumbing
# ----------------------------------------------------------------------
def _update_quotes(book, account, quote, size):
    """Cancel and replace only when a price changes, so unchanged quotes keep
    their place in the queue."""
    fills = []

    best_ask = book.best_ask()
    best_bid = book.best_bid()
    bid = quote.bid
    ask = quote.ask
    # Quotes stay passive: never cross the opposite best price.
    if bid is not None and best_ask is not None and bid >= best_ask:
        bid = best_ask - 1
    if ask is not None and best_bid is not None and ask <= best_bid:
        ask = best_bid + 1

    if bid != account.bid_price or account.bid_id not in book.orders:
        if account.bid_id is not None:
            book.cancel(account.bid_id)
        account.bid_id = None
        account.bid_price = None
        if bid is not None:
            order_id, new_fills = book.add_limit(MM, BUY, bid, size)
            fills += new_fills
            account.bid_id = order_id
            account.bid_price = bid

    if ask != account.ask_price or account.ask_id not in book.orders:
        if account.ask_id is not None:
            book.cancel(account.ask_id)
        account.ask_id = None
        account.ask_price = None
        if ask is not None:
            order_id, new_fills = book.add_limit(MM, SELL, ask, size)
            fills += new_fills
            account.ask_id = order_id
            account.ask_price = ask

    return fills


def _apply_fill(account, fill, step, mid):
    """Update cash and inventory. Returns spread captured versus mid."""
    if fill.buyer == MM and fill.seller == MM:
        return 0.0
    if fill.buyer == MM:
        account.inventory += fill.qty
        account.cash -= fill.price * fill.qty
        account.fills.append((step, BUY, fill.price, fill.qty))
        return (mid - fill.price) * fill.qty
    if fill.seller == MM:
        account.inventory -= fill.qty
        account.cash += fill.price * fill.qty
        account.fills.append((step, SELL, fill.price, fill.qty))
        return (fill.price - mid) * fill.qty
    return 0.0


def _sync_quote_ids(book, account):
    if account.bid_id is not None and account.bid_id not in book.orders:
        account.bid_id = None
        account.bid_price = None
    if account.ask_id is not None and account.ask_id not in book.orders:
        account.ask_id = None
        account.ask_price = None


def _cancel_quotes(book, account):
    if account.bid_id is not None:
        book.cancel(account.bid_id)
    if account.ask_id is not None:
        book.cancel(account.ask_id)
    account.bid_id = None
    account.ask_id = None


def _flatten(book, account):
    if account.inventory == 0:
        return
    side = SELL if account.inventory > 0 else BUY
    fills = book.market(MM, side, abs(account.inventory))
    for fill in fills:
        if side == SELL:
            account.cash += fill.price * fill.qty
            account.inventory -= fill.qty
        else:
            account.cash -= fill.price * fill.qty
            account.inventory += fill.qty
    # If the book was too thin, close the rest one tick worse than the last fill.
    if account.inventory != 0:
        last_price = fills[-1].price if fills else book.mid()
        penalty_price = last_price + side * 1
        account.cash += account.inventory * penalty_price
        account.inventory = 0


def _markout(fills, mid_path, horizon):
    """Average profit per lot measured `horizon` steps after each fill.

    Negative markouts mean the maker is being picked off by informed flow.
    """
    if not fills:
        return 0.0
    total = 0.0
    lots = 0
    last = len(mid_path) - 1
    for step, side, price, qty in fills:
        future_mid = mid_path[min(step + horizon, last)]
        total += side * (future_mid - price) * qty
        lots += qty
    return total / lots
