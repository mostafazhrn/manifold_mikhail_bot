# -------------------------
# FILE: simple_stats.py
# -------------------------

# --- bootstrap src/ path so script runs standalone ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # project root
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ----------------------------------------------------

import time
from collections import Counter

from mikhail_bot.api import get_market as get_market_by_id
from mikhail_bot.ledger import _load_jsonl, TRADE_LEDGER_PATH

CHECK_DELAY = 1.0  # seconds between API calls


def run_simple_evaluation():
    trades = _load_jsonl(TRADE_LEDGER_PATH)

    if not trades:
        print("[STATS] No trades found.")
        return

    print(f"[STATS] Loaded {len(trades)} trades from ledger.")

    stats = Counter()
    resolved_seen = set()

    for trade in trades:
        market_id = trade.get("marketId")
        if not market_id or market_id in resolved_seen:
            continue

        market = get_market_by_id(market_id)
        if not market:
            continue

        if not market.get("isResolved"):
            continue

        resolved_seen.add(market_id)

        predicted = trade.get("predictedOutcome")
        actual = market.get("resolution")

        is_correct = predicted == actual

        stats["total"] += 1
        stats["wins"] += int(is_correct)
        stats["losses"] += int(not is_correct)

        print(
            f"[RESOLVED] {market_id} | "
            f"Predicted={predicted} | "
            f"Actual={actual} | "
            f"{'WIN' if is_correct else 'LOSS'}"
        )

        time.sleep(CHECK_DELAY)

    if stats["total"] == 0:
        print("\n[STATS] No resolved markets yet.")
        return

    win_rate = stats["wins"] / stats["total"]

    print("\n================ SIMPLE PERFORMANCE =================")
    print(f"Resolved trades : {stats['total']}")
    print(f"Wins            : {stats['wins']}")
    print(f"Losses          : {stats['losses']}")
    print(f"Win rate        : {win_rate:.2%}")
    print("=====================================================\n")


if __name__ == "__main__":
    run_simple_evaluation()
