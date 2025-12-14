from src.mikhail_bot.ledger import (
    _load_jsonl,
    evaluate_resolved_market,
    TRADE_LEDGER_PATH,
)
from src.mikhail_bot.api import get_market_by_id


def check_resolved_trades():
    """Fetch markets the bot traded on and evaluate resolved ones."""

    trades = _load_jsonl(TRADE_LEDGER_PATH)
    seen = set()

    for trade in trades:
        market_id = trade["marketId"]
        if market_id in seen:
            continue
        seen.add(market_id)

        market = get_market_by_id(market_id)
        if market and market.get("isResolved"):
            evaluate_resolved_market(full_market=market)
