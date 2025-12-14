"""
Example runner: starts the bot in paper or live mode.

Usage:
    python examples/run_bot.py --paper
    python examples/run_bot.py --live
    python examples/run_bot.py --paper --super   # enable super LLM mode
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from pathlib import Path
import os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

print("[RUN_BOT] ENV STRATEGY_MODE =", os.getenv("STRATEGY_MODE"))
print("[RUN_BOT] ENV LLM_PROVIDER  =", os.getenv("LLM_PROVIDER"))
print("[RUN_BOT] ENV OPENAI KEY    =", bool(os.getenv("OPENAI_API_KEY")))
# ------------------------------------------------------------
# Path setup
# ------------------------------------------------------------
#ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(ROOT)

#print("[RUN_BOT] Root path added:", ROOT)

# ------------------------------------------------------------
# Import strategy module + trader
# ------------------------------------------------------------
SmartStrategy_module = None
SmartStrategy_set_mode = None
SmartStrategy_callable = None

try:
    # import the module so we can call set_mode() and smart_strategy()
    from src.mikhail_bot import smart_strategy_v4 as SmartStrategy_module
    # helper and callable
    SmartStrategy_set_mode = getattr(SmartStrategy_module, "set_mode", None)
    SmartStrategy_callable = getattr(SmartStrategy_module, "smart_strategy", None)
    print("[RUN_BOT] Loaded SmartStrategyV4 module successfully.")
except Exception as e:
    print("[RUN_BOT][ERROR] Failed to import SmartStrategyV3:", e)
    SmartStrategy_module = None
    SmartStrategy_set_mode = None
    SmartStrategy_callable = None

#try:
 #   from src.mikhail_bot.trader import Trader
 #  print("[RUN_BOT] Trader imported successfully.")
#except Exception as e:
 #   print("[RUN_BOT][CRITICAL] Cannot import Trader:", e)
  #  raise

# Optional: for paper trade logging
try:
    from src.mikhail_bot.simulator import log_trade_row
except Exception:
    pass


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("[RUN_BOT] Starting bot...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="Run in paper mode (default)")
    parser.add_argument("--live", action="store_true",
                        help="Run live (place real trades)")
    parser.add_argument("--strategy", type=str, default="v3",
                        help="Which strategy to use: v2 or v3 (default v3)")
    parser.add_argument("--super", action="store_true",
                        help="Enable super mode (call LLM for every market) ")
    args = parser.parse_args()

    # Decide live or paper mode
    if args.live:
        paper = False
    elif args.paper:
        paper = True
    else:
        paper = True  # default safety
    
    # 🔒 CLI is source of truth for paper/live
    os.environ["PAPER_MODE"] = "true" if paper else "false"
    print(f"[RUN_BOT] ENV PAPER_MODE overridden to {os.environ['PAPER_MODE']}")


    mode = "LIVE" if not paper else "PAPER"
    print(f"[RUN_BOT] Mode selected: {mode}")
    from src.mikhail_bot.trader import Trader
    print("[RUN_BOT] Trader imported AFTER PAPER_MODE was set.")

    
    # If user requested super, call set_mode on the strategy module
    if args.super:
        if SmartStrategy_set_mode is None:
            print("[RUN_BOT][WARN] --super requested but set_mode() not available on SmartStrategy module.")
        else:
            try:
                SmartStrategy_set_mode("super")
                print("[RUN_BOT] Super mode ENABLED (LLM will be called for every market).")
            except Exception as e:
                print("[RUN_BOT][ERROR] Failed to enable super mode:", e)
    else:
        print("[RUN_BOT] Super mode NOT enabled; running in normal mode.")

    # Decide which strategy to use
    strategy_choice = args.strategy.lower()
    print(f"[RUN_BOT] Requested strategy: {strategy_choice}")

    if strategy_choice == "v3":
        if SmartStrategy_callable is None:
            print("[RUN_BOT][ERROR] SmartStrategyV3 unavailable. Cannot continue.")
            return
        strategy = SmartStrategy_callable
        print("[RUN_BOT] Using SmartStrategyV3.")
    else:
        print("[RUN_BOT][ERROR] Only v3 is supported now. Please pass --strategy v3")
        return

    # ------------------------------------------------------------
    # Initialize Trader
    # ------------------------------------------------------------
    try:
        print("[RUN_BOT] Initializing Trader...")
        # Pass strategy callable into Trader if your Trader accepts it.
        # If your Trader signature doesn't accept 'strategy', remove that kwarg.
        try:
            t = Trader(paper=paper, strategy=strategy)
        except TypeError:
            # fallback: Trader doesn't accept a strategy param; initialize with paper only
            print("[RUN_BOT] Trader.__init__ does not accept 'strategy' parameter; initializing with paper only.")
            t = Trader(paper=paper)
        print("[RUN_BOT] Trader initialized successfully.")
    except Exception as e:
        print("[RUN_BOT][CRITICAL] Failed while creating Trader instance:", e)
        raise

    # ------------------------------------------------------------
    # Run one full cycle
    # ------------------------------------------------------------
    print("[RUN_BOT] Running one iteration with run_once()...")

    try:
        t.run_once()
        print("[RUN_BOT] run_once() completed.")
    except Exception as e:
        print("[RUN_BOT][CRITICAL] run_once() crashed:", e)
        raise


if __name__ == "__main__":
    main()
