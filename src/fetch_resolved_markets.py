import json
import os
import time
import requests
from dotenv import load_dotenv
from pathlib import Path

"""
================================================================================
 FULL RESOLVED MARKET FETCHER (MCQ-ENABLED, MERGED SUMMARY + DETAIL)
================================================================================

 This script fetches ALL resolved Manifold markets in two stages:

   1) Fetch summary pages from /v0/markets (fast, paginated)
   2) For each resolved market → fetch full detail via /v0/market/:id
      This includes MCQ answers, free-response answers, numeric resolutions, etc.

 Summary + detail objects are MERGED to ensure:
      - No data is lost
      - All answer options appear for MCQ/free-response/poll markets
      - All metadata remains from the summary object

 Output is saved in JSONL: one complete object per line.

 All original behavior preserved:
      - Debugs
      - Pagination
      - MAX_RESOLVED_MARKETS limit
      - Skips duplicates
      - Resumes where you left off
      - Waits between requests to avoid rate limits

================================================================================
"""

# -----------------------------
# Load .env
# -----------------------------
ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ROOT_ENV)

# -----------------------------
# Configuration
# -----------------------------
MAX_MARKETS = int(os.getenv("MAX_RESOLVED_MARKETS", 50000))

RESOLVED_FILE = "data/resolved_markets.jsonl"
API_LIST = "https://api.manifold.markets/v0/markets"
API_SINGLE = "https://api.manifold.markets/v0/market"  # + "/<id>"


def append_jsonl(path, obj):
    """Append one JSON object (dict) as a line in JSONL."""
    with open(path, "a", encoding="utf8") as f:
        f.write(json.dumps(obj) + "\n")


def load_existing_ids(path):
    """Return a set of IDs already stored."""
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except:
                pass
    return ids


def fetch_full_market(mid):
    """
    Fetch full market detail with MCQ answers etc.
    Returns None on failure.
    """
    url = f"{API_SINGLE}/{mid}"

    try:
        full = requests.get(url, timeout=10).json()
        return full
    except Exception as e:
        print(f"[ERROR] Failed to fetch full market {mid}: {e}")
        return None


def fetch_all_resolved():
    print("============================================================")
    print("  Fetching FULL resolved markets (MCQ-enabled, merged mode)")
    print("============================================================")
    print(f"[CONFIG] MAX_RESOLVED_MARKETS = {MAX_MARKETS}")
    print(f"[SAVE]   Saving to: {RESOLVED_FILE}")

    seen = load_existing_ids(RESOLVED_FILE)
    print(f"[INFO] Already have {len(seen)} resolved markets stored.\n")

    before = None
    total_added = 0
    page = 0

    while True:

        # Stop if limit reached
        if len(seen) >= MAX_MARKETS:
            print(f"\nReached MAX_RESOLVED_MARKETS={MAX_MARKETS}. Stopping.")
            break

        params = {"limit": 500}
        if before:
            params["before"] = before

        print(f"\n[PAGE] Requesting page {page}: params={params}")
        page += 1

        try:
            res = requests.get(API_LIST, params=params, timeout=10).json()
        except Exception as e:
            print("[ERROR] Failed to fetch page:", e)
            time.sleep(1)
            continue

        if not res:
            print("\n[INFO] No more pages available. Exiting.")
            break

        print(f"[INFO] Fetched {len(res)} markets in this page.")

        for summary_market in res:

            # Only resolved markets
            if not summary_market.get("isResolved"):
                continue

            mid = summary_market["id"]

            # Skip duplicates
            if mid in seen:
                continue

            # --------------------------
            # FETCH FULL DETAIL MARKET
            # --------------------------
            print(f"[DETAIL] Fetching full market: {mid}")
            full = fetch_full_market(mid)
            if not full:
                print(f"[WARN] Skipping {mid} due to failed detail fetch.")
                continue

            # --------------------------
            # MERGE SUMMARY + FULL DETAIL
            # --> full overwrites summary if conflicts
            # --------------------------
            merged = {**summary_market, **full}

            # Save final enriched object
            append_jsonl(RESOLVED_FILE, merged)
            seen.add(mid)
            total_added += 1

            print(f"[ADDED] {mid} | {summary_market.get('question')}")

            if len(seen) >= MAX_MARKETS:
                break

            # Short sleep to be nice to API
            time.sleep(0.15)

        # Prepare next page
        before = res[-1]["id"]
        time.sleep(0.3)

    print("\n============================================================")
    print(f"Done. Added {total_added} new fully-detailed resolved markets.")
    print(f"Total stored resolved markets: {len(seen)}")
    print("============================================================")


if __name__ == "__main__":
    fetch_all_resolved()
