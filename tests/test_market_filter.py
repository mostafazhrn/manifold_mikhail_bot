from mikhail_bot.market_filter import get_markets_by_creator


def test_get_markets_by_creator_returns_list():
    markets = get_markets_by_creator(creator_username="MikhailTal")
    assert isinstance(markets, list)