"""Example strategies.
Keep strategies small and composable: input market dict -> score/probability -> action.
"""
from typing import Dict


def baseline_strategy(market: Dict) -> dict:
    """A very small baseline: if price < 0.5 -> small YES; else small NO.
    Returns a decision dict with 'side' and 'amount'.
    """
    price = market.get("probability") or market.get("price") or 0.5
    side = "YES" if price < 0.5 else "NO"
    return {"side": side, "amount": 5.0}



def value_estimate_strategy(market: Dict) -> dict:
    """A slightly smarter placeholder: uses title heuristics to tilt toward YES/NO.
    Replace with LLM or model ensemble for real performance testing.
    """
    title = (market.get("name") or "").lower()
    if "no" in title or "not" in title:
        tilt = 0.6
    else:
        tilt = 0.45
    # pick side by tilt vs current price
    price = market.get("probability") or market.get("price") or 0.5
    side = "YES" if tilt > price else "NO"
    return {"side": side, "amount": 10.0}