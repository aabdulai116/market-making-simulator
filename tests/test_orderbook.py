from mmsim.orderbook import BUY, SELL, OrderBook


def test_resting_orders_set_best_prices():
    book = OrderBook()
    book.add_limit("a", BUY, 99, 5)
    book.add_limit("b", SELL, 101, 5)
    assert book.best_bid() == 99
    assert book.best_ask() == 101
    assert book.mid() == 100
    assert book.spread() == 2


def test_market_order_walks_the_book():
    book = OrderBook()
    book.add_limit("a", SELL, 101, 3)
    book.add_limit("b", SELL, 102, 4)
    fills = book.market("taker", BUY, 5)

    assert len(fills) == 2
    assert fills[0].price == 101
    assert fills[0].qty == 3
    assert fills[1].price == 102
    assert fills[1].qty == 2
    assert book.best_ask() == 102
    assert book.depth(SELL)[0] == (102, 2)


def test_time_priority_at_same_price():
    book = OrderBook()
    first_id, _ = book.add_limit("first", BUY, 100, 2)
    book.add_limit("second", BUY, 100, 2)
    fills = book.market("taker", SELL, 3)

    assert fills[0].buyer == "first"
    assert fills[0].qty == 2
    assert fills[1].buyer == "second"
    assert fills[1].qty == 1
    assert first_id not in book.orders


def test_crossing_limit_order_trades_then_rests():
    book = OrderBook()
    book.add_limit("maker", SELL, 100, 2)
    order_id, fills = book.add_limit("taker", BUY, 101, 5)

    assert sum(fill.qty for fill in fills) == 2
    assert fills[0].price == 100          # trades at the resting price
    assert book.best_bid() == 101
    assert book.orders[order_id].qty == 3


def test_cancel_removes_order_and_empty_level():
    book = OrderBook()
    order_id, _ = book.add_limit("a", BUY, 99, 5)
    assert book.cancel(order_id)
    assert book.best_bid() is None
    assert not book.cancel(order_id)
