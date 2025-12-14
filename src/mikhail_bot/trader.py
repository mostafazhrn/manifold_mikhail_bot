# -------------------------
# FILE: trader.py (clean version preserving all logic)
# -------------------------
"""
Trading loop and execution (uses smart_strategy_v3).
This version respects .env settings for max_pages and max_markets unless the
caller explicitly overrides them.

Notes / Safety:
 - Trader expects get_markets_by_creator to return FULL market objects.
 - Strategy functions should accept a single argument: the full market dict.
 - This file preserves paper/live modes and supports strategy injection.
"""

import time
from typing import Optional, Callable, Any
from src.mikhail_bot.market_filter import get_markets_by_creator
from src.mikhail_bot.smart_strategy_v4 import smart_strategy  # fallback default
from src.mikhail_bot.api import place_trade
from src.mikhail_bot.ledger import has_traded, record_trade



class Trader:
    def __init__(
        self,
        paper: bool = True,
        strategy: Optional[Callable[[dict], Optional[dict]]] = None,
        max_pages: Optional[int] = None,
        max_markets: Optional[int] = None,
    ):
        """
        Trader will use .env values for max_pages / max_markets unless overrides
        are explicitly passed from run_bot.py.

        Args:
            paper: True means simulate trades only.
            strategy: A strategy function injected from run_bot.py (callable(market) -> decision dict).
            max_pages: Optional override for pagination.
            max_markets: Optional override for filtered markets.
        """

        self.paper = paper
        # Use injected strategy if provided, otherwise use the default smart_strategy function.
        self.strategy = strategy if strategy is not None else smart_strategy
        self.seen = set()

        # Store overrides (can be None → means "use .env")
        self.max_pages = max_pages
        self.max_markets = max_markets

        print(f"[TRADER] Initialized Trader(paper={self.paper})")
        print(f"[TRADER] Strategy injected: {getattr(self.strategy, '__name__', str(self.strategy))}")
        print(f"[TRADER] Page override = {self.max_pages}, Market override = {self.max_markets}")

    # -------------------------------------------------------------
    # One trading pass
    # -------------------------------------------------------------
    def run_once(self):
        print("[TRADER] Calling get_markets_by_creator()...")

        # Important: These go straight to market_filter which reads .env
        markets = get_markets_by_creator(
            creator_username="MikhailTal",
            max_pages=self.max_pages,     # None → use .env
            max_markets=self.max_markets  # None → use .env
        )

        print("Markets to evaluate:", len(markets))
        print("Fetched", len(markets), "filtered markets.")

        display_count = 0

        # -------------------------------------------------------------
        # Loop through markets
        # -------------------------------------------------------------
        for m in markets:
            # try several keys for robust id retrieval
            mid = m.get("id") or m.get("marketId") or m.get("slug")
            if not mid:
                print("[TRADER][WARN] Market without id/slug, skipping:", m.get("question") or m.get("name", "<unknown>"))
                continue
            #### 
            # Skip if already processed in this run
            if mid in self.seen:
                continue

            # Skip if already traded in previous runs (ledger memory)
            if has_traded(mid):
                print(f"[TRADER][SKIP] Market {mid} already traded on. Skipping.")
                self.seen.add(mid)
                continue

            # mark seen early to avoid duplicate work in long runs
            self.seen.add(mid)
            question = m.get("question") or m.get("name") or m.get("slug") or "Unknown Market"

            print("-----------------------------------------------------")
            print("Evaluating market:")
            print(" ID: ", mid)
            print(" Q: ", question)
            display_count += 1

            # Strategy call: pass FULL market dict (so strategy can inspect outcomeType, answers, etc.)
            try:
                # Some strategies accept debug flag; call in the simplest way if strategy signature unknown
                decision = None
                try:
                    # prefer the (market, debug=True) signature if available
                    decision = self.strategy(m, debug=True)
                except TypeError:
                    # fallback to simple call
                    decision = self.strategy(m)
            except Exception as e:
                print(f"[TRADER][ERROR] Strategy raised exception for market {mid}: {e}")
                # don't crash whole run; skip this market
                continue

            if not decision:
                print(f"[PAPER] Smart strategy skipped market {mid}.")
                continue

            # Decision expected to be a dict with at least 'side' and 'amount'
            side = decision.get("side")
            amount = decision.get("amount")
            probability = decision.get("probability")

            if not side or amount is None or amount <= 0:
                print(f"[PAPER] Smart strategy decided to skip market {mid} (no actionable side/amount). Decision: {decision}")
                continue

            # Paper vs LIVE
            if self.paper:
                print(f"[PAPER] Would place {amount} on {side} for market {mid}")
            else:
                print(f"[LIVE] Placing trade for {mid}: {decision}")
                try:
                    res = place_trade(m, decision)
                    print("[LIVE] Trade result:", res)

                    record_trade(
                        market_id=mid,
                        question=question,
                        predicted_outcome=side,
                        probability=probability,
                        amount=amount,
                        strategy=getattr(self.strategy, "__name__", "unknown"),
                    )


                except Exception as e:
                    print(f"[LIVE][ERROR] Trade failed for {mid}: {e}")


        print("Total markets displayed:", display_count)
        print("Finished run_once.")

    # -------------------------------------------------------------
    # Looping mode
    # -------------------------------------------------------------
    def loop(self, delay: int = 30):
        while True:
            try:
                self.run_once()
            except Exception as e:
                print("[TRADER][ERROR] Error in run_once:", e)

            time.sleep(delay)
