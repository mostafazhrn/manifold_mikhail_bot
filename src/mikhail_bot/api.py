from typing import List, Dict, Optional
import requests
from .config import config

API_BASE = "https://api.manifold.markets"
HEADERS = {"Authorization": f"Key {config.api_key}"}


# -----------------------------------------------------
# 1) Basic single-page market fetch (unchanged)
# -----------------------------------------------------
def list_markets(limit: int = 1000, before: Optional[str] = None) -> List[Dict]:
    """
    Fetch up to {limit} markets from Manifold.
    If 'before' is provided, fetch markets created before that market ID.
    """
    params = {"limit": limit}
    if before:
        params["before"] = before

    resp = requests.get(f"{API_BASE}/v0/markets", params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    data = resp.json()
    return data if isinstance(data, list) else []


# -----------------------------------------------------
# 2) NEW — Fetch ALL markets using pagination
# -----------------------------------------------------
def list_all_markets(limit: int = 1000, max_pages: int | None = None) -> List[Dict]:
    """
    Fetch ALL markets (active + closed + resolved) across pages, up to max_pages if specified.
    """
    all_markets: List[Dict] = []
    before: str | None = None
    page = 0

    while True:
        if max_pages is not None and page >= max_pages:
            print(f"[INFO] Reached max_pages={max_pages}, stopping fetch.")
            break

        print(f"\n--- Fetching page {page} ---")
        batch = list_markets(limit=limit, before=before)

        if not batch:
            print("\nNo more markets returned. Pagination complete.")
            break

        all_markets.extend(batch)
        print(f"Fetched {len(batch)} markets this page, total so far: {len(all_markets)}")

        before = batch[-1]["id"]
        page += 1

    print(f"\nDONE! Total markets fetched: {len(all_markets)}")
    return all_markets


# -----------------------------------------------------
# 3) Fetch single market (unchanged)
# -----------------------------------------------------
def get_market(market_id: str) -> Dict:
    resp = requests.get(f"{API_BASE}/v0/market/{market_id}", headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


# -----------------------------------------------------
# 4) Place a trade (unchanged)
# -----------------------------------------------------
def place_trade(market, decision, *args, **kwargs):
    market_id = market["id"]
    amount = decision["amount"]

    # 1) Fetch full market info
    full = get_market(market_id)
    kind = full.get("outcomeType")

    print(f"\n[DEBUG] Market kind = {kind}")
    print(f"[DEBUG] Decision = {decision}")

    # === BINARY MARKET ===
    if kind == "BINARY":
        side = decision["side"]

        if side not in ("YES", "NO"):
            print(f"[ERROR] Invalid binary side: {side}")
            return None

        payload = {
            "contractId": market_id,
            "amount": amount,
            "outcome": side
        }

        print("[DEBUG] Binary bet payload →", payload)

        r = requests.post(API_BASE + "/v0/bet",
                          json=payload,
                          headers=HEADERS)

        print("[DEBUG] Response:", r.text)
        return r

    # === MCQ / FREE RESPONSE ===
    if kind in ("MULTIPLE_CHOICE", "FREE_RESPONSE"):
        answer_id = decision["side"]

        payload = {
            "contractId": market_id,
            "amount": amount,
            "answerIds": [answer_id]
        }

        print("[DEBUG] MCQ bet payload →", payload)

        r = requests.post(API_BASE + "/v0/multi-bet",
                          json=payload,
                          headers=HEADERS)

        print("[DEBUG] Response:", r.text)
        return r

    print(f"[ERROR] Unsupported market type: {kind}")
    return None
