"""
Fetch and filter markets created by a specific creator (e.g., MikhailTal),
using pagination and local filtering. Supports max_pages and max_markets
loaded from .env so that users can easily customize behavior.

Key behavior (kept deliberately simple and safe):
 - This function RETURNS the full market objects fetched from the API.
 - It does NOT strip/normalize fields. Strategy code will receive full market dict.
 - Pagination uses the last market.id as cursor (the API returns market objects).
"""

from typing import List, Dict, Optional
from src.mikhail_bot.api import list_markets
from src.mikhail_bot.config import config
import os
from dotenv import load_dotenv
from pathlib import Path

# -------------------------
# Load .env
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# -------------------------
# Defaults from .env
# -------------------------
MAX_PAGES = int(os.getenv("MAX_PAGES", 20))
MAX_MARKETS = int(os.getenv("MAX_MARKETS", 5000))


def _extract_creator_username(market: Dict) -> Optional[str]:
    """
    Robustly extract the creator username from a market object.
    Can handle different shapes (creatorUsername, createdBy, creator dict).
    """
    creator = market.get("creatorUsername") or market.get("createdBy") or market.get("creator")
    if isinstance(creator, dict):
        return creator.get("username")
    return creator


# ---------------------------------------------------------
# Main Function: get_markets_by_creator
# ---------------------------------------------------------
def get_markets_by_creator(
    creator_username: Optional[str] = None,
    max_pages: Optional[int] = None,
    max_markets: Optional[int] = None
) -> List[Dict]:
    """
    Fetch Manifold markets in pages and filter by the given creator.

    Args:
        creator_username: Username to keep (default = config.designated_username)
        max_pages: Max pagination pages to fetch (default = .env)
        max_markets: Max number of filtered markets to return (default = .env)

    Returns:
        List of full market dicts filtered by creator (UNMODIFIED).
    """

    creator_username = creator_username or getattr(config, "designated_username", None)
    max_pages = max_pages or MAX_PAGES
    max_markets = max_markets or MAX_MARKETS

    if not creator_username:
        raise ValueError("get_markets_by_creator: creator_username not provided and config.designated_username not set")

    print(f"[FILTER] Fetching markets for creator='{creator_username}'")
    print(f"[FILTER] Max pages = {max_pages}, Max markets = {max_markets}")

    markets: List[Dict] = []
    before: Optional[str] = None
    pages_fetched = 0

    # -----------------------------------------------------
    # Pagination Loop
    # -----------------------------------------------------
    while True:
        pages_fetched += 1
        print(f"\n[FILTER] Fetching page {pages_fetched}/{max_pages} (before={before})...")

        # list_markets returns up to `limit` markets in a single page
        try:
            batch = list_markets(limit=1000, before=before)
        except Exception as e:
            print(f"[FILTER][ERROR] list_markets failed: {e}")
            break

        if not batch:
            print("[FILTER] No more markets returned. Stopping pagination.")
            break

        print(f"[FILTER] Got {len(batch)} markets in this batch.")

        # Stop if page limit reached (we increment pages_fetched at top)
        if pages_fetched > max_pages:
            print("[FILTER] Reached max_pages limit. Stopping.")
            break

        # -----------------------------------------------------
        # Filter by creator (keep the full market object)
        # -----------------------------------------------------
        for m in batch:
            try:
                c = _extract_creator_username(m)
            except Exception:
                c = None

            if c == creator_username:
                markets.append(m)
                if len(markets) >= max_markets:
                    print("[FILTER] Reached max_markets limit. Stopping.")
                    break

        if len(markets) >= max_markets:
            break

        # -----------------------------------------------------
        # Correct pagination: use last market id as cursor
        # -----------------------------------------------------
        try:
            before = batch[-1]["id"]
        except Exception:
            print("[FILTER] WARNING: Missing ID in last market or unexpected batch shape; stopping pagination.")
            break

        print(f"[FILTER] Next pagination cursor (before): {before}")

    print(f"\n[FILTER] Final filtered markets count: {len(markets)}")
    return markets
