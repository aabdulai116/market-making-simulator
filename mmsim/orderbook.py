"""A limit order book with price-time priority matching.

Prices are stored as integer ticks so there is no floating point drift.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

BUY = 1
SELL = -1


@dataclass
class Order:
    order_id: int
    owner: str
    side: int
    price: int
    qty: int
    timestamp: int


@dataclass
class Fill:
    buyer: str
    seller: str
    price: int
    qty: int
    aggressor_side: int
    passive_order_id: int


@dataclass
class OrderBook:
    bids: Dict[int, Deque[Order]] = field(default_factory=dict)
    asks: Dict[int, Deque[Order]] = field(default_factory=dict)
    orders: Dict[int, Order] = field(default_factory=dict)
    next_id: int = 1
    clock: int = 0

    # ------------------------------------------------------------------
    # Book state
    # ------------------------------------------------------------------
    def best_bid(self) -> Optional[int]:
        if not self.bids:
            return None
        return max(self.bids)

    def best_ask(self) -> Optional[int]:
        if not self.asks:
            return None
        return min(self.asks)

    def mid(self) -> Optional[float]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def best_excluding(self, side: int, owner: str) -> Optional[int]:
        """Best price on one side, ignoring orders that belong to `owner`.

        A market maker should value the stock from everyone else's orders.
        If it used a mid that includes its own quotes, its quotes would feed
        back into its own fair value.
        """
        book = self.bids if side == BUY else self.asks
        for price in sorted(book, reverse=(side == BUY)):
            for order in book[price]:
                if order.owner != owner:
                    return price
        return None

    def mid_excluding(self, owner: str) -> Optional[float]:
        bid = self.best_excluding(BUY, owner)
        ask = self.best_excluding(SELL, owner)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def spread(self) -> Optional[int]:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return ask - bid

    def depth(self, side: int, levels: int = 5) -> List[tuple]:
        """Return (price, total_qty) for the best few levels on one side."""
        book = self.bids if side == BUY else self.asks
        prices = sorted(book, reverse=(side == BUY))[:levels]
        result = []
        for price in prices:
            total = sum(order.qty for order in book[price])
            result.append((price, total))
        return result

    # ------------------------------------------------------------------
    # Order entry
    # ------------------------------------------------------------------
    def add_limit(self, owner: str, side: int, price: int, qty: int):
        """Submit a limit order. Any part that crosses the book trades
        immediately; the rest rests in the book.

        Returns the resting order id (or None if fully filled) and fills.
        """
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.clock += 1
        fills = self._match(owner, side, qty, limit_price=price)
        filled_qty = sum(fill.qty for fill in fills)
        remaining = qty - filled_qty
        if remaining == 0:
            return None, fills

        order = Order(self.next_id, owner, side, price, remaining, self.clock)
        self.next_id += 1
        book = self.bids if side == BUY else self.asks
        if price not in book:
            book[price] = deque()
        book[price].append(order)
        self.orders[order.order_id] = order
        return order.order_id, fills

    def market(self, owner: str, side: int, qty: int) -> List[Fill]:
        """Submit a market order. Unfilled quantity is dropped."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.clock += 1
        return self._match(owner, side, qty, limit_price=None)

    def cancel(self, order_id: int) -> bool:
        order = self.orders.pop(order_id, None)
        if order is None:
            return False
        book = self.bids if order.side == BUY else self.asks
        level = book[order.price]
        level.remove(order)
        if not level:
            del book[order.price]
        return True

    # ------------------------------------------------------------------
    # Matching engine
    # ------------------------------------------------------------------
    def _match(self, owner, side, qty, limit_price):
        fills = []
        opposite = self.asks if side == BUY else self.bids
        remaining = qty

        while remaining > 0 and opposite:
            if side == BUY:
                best_price = min(opposite)
                crosses = limit_price is None or best_price <= limit_price
            else:
                best_price = max(opposite)
                crosses = limit_price is None or best_price >= limit_price
            if not crosses:
                break

            level = opposite[best_price]
            while remaining > 0 and level:
                resting = level[0]
                traded = min(remaining, resting.qty)

                if side == BUY:
                    buyer = owner
                    seller = resting.owner
                else:
                    buyer = resting.owner
                    seller = owner

                fills.append(
                    Fill(buyer, seller, best_price, traded, side, resting.order_id)
                )
                resting.qty -= traded
                remaining -= traded
                if resting.qty == 0:
                    level.popleft()
                    del self.orders[resting.order_id]

            if not level:
                del opposite[best_price]

        return fills
