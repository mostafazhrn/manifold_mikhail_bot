# -------------------------
# FILE: ledger.py
# -------------------------

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Optional

# =========================================================
# Paths
# =========================================================

DATA_DIR = "data"

TRADE_LEDGER_PATH = os.path.join(DATA_DIR, "trade_ledger.jsonl")
TRADE_RESULTS_PATH = os.path.join(DATA_DIR, "trade_results.jsonl")
RESOLVED_MARKETS_PATH = os.path.join(DATA_DIR, "resolved_markets.jsonl")


# =========================================================
# Internal helpers
# =========================================================

_traded_cache: Optional[Set[str]] = None
_resolved_cache: Optional[Set[str]] = None


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: str, obj: Dict):
    _ensure_data_dir()
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# TRADE MEMORY
# =========================================================

def get_traded_market_ids() -> Set[str]:
    """
    Cached set of marketIds already traded on.
    """
    global _traded_cache

    if _traded_cache is None:
        trades = _load_jsonl(TRADE_LEDGER_PATH)
        _traded_cache = {
            t.get("marketId")
            for t in trades
            if t.get("marketId")
        }

    return _traded_cache


def has_traded(market_id: str) -> bool:
    return market_id in get_traded_market_ids()


def record_trade(
    *,
    market_id: str,
    question: str,
    predicted_outcome: str,
    probability: Optional[float],
    amount: float,
    strategy: str,
    mode: str = "paper",  # paper | live
):
    """
    Record a trade made by the bot.
    """

    trade = {
        "marketId": market_id,
        "question": question,
        "predictedOutcome": predicted_outcome,
        "probabilityAtTrade": probability,
        "amount": amount,
        "strategy": strategy,
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _append_jsonl(TRADE_LEDGER_PATH, trade)

    # Update cache immediately
    get_traded_market_ids().add(market_id)


# =========================================================
# RESOLUTION + EVALUATION
# =========================================================

def _get_resolved_market_ids() -> Set[str]:
    global _resolved_cache

    if _resolved_cache is None:
        rows = _load_jsonl(RESOLVED_MARKETS_PATH)
        _resolved_cache = {
            r.get("id") or r.get("marketId")
            for r in rows
            if r.get("id") or r.get("marketId")
        }

    return _resolved_cache


def evaluate_resolved_market(*, full_market: Dict):
    """
    Evaluate resolved market against recorded trades
    and store results for ML training.
    """

    market_id = full_market.get("id") or full_market.get("marketId")
    if not market_id:
        return

    resolution = full_market.get("resolution")
    if resolution is None:
        return

    trades = _load_jsonl(TRADE_LEDGER_PATH)
    related_trades = [
        t for t in trades if t.get("marketId") == market_id
    ]

    if not related_trades:
        return

    for trade in related_trades:
        predicted = trade.get("predictedOutcome")

        is_correct = False
        if isinstance(resolution, str):
            is_correct = predicted == resolution
        elif isinstance(resolution, (int, float)):
            # Numeric markets (future extension)
            is_correct = True

        result = {
            "marketId": market_id,
            "question": trade.get("question"),
            "predictedOutcome": predicted,
            "resolvedOutcome": resolution,
            "probabilityAtTrade": trade.get("probabilityAtTrade"),
            "amount": trade.get("amount"),
            "strategy": trade.get("strategy"),
            "mode": trade.get("mode"),
            "isCorrect": is_correct,
            "resolutionTime": full_market.get("resolutionTime"),
            "evaluatedAt": datetime.utcnow().isoformat(),
        }

        _append_jsonl(TRADE_RESULTS_PATH, result)

    # Save resolved market ONCE (for ML training)
    if market_id not in _get_resolved_market_ids():
        _append_jsonl(RESOLVED_MARKETS_PATH, full_market)
        _get_resolved_market_ids().add(market_id)
