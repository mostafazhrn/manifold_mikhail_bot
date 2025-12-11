"""Very small simulator that records decisions to CSV for offline analysis."""
import csv
from datetime import datetime
from pathlib import Path


LOG = Path("logs")
LOG.mkdir(exist_ok=True)




def log_trade_row(market, decision, paper=True):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_id": market.get("id") or market.get("slug"),
        "title": market.get("name"),
        "side": decision.get("side"),
        "amount": decision.get("amount"),
        "paper": paper,
    }
    csvfile = LOG / "trades.csv"
    file_existed = csvfile.exists()
    with open(csvfile, "a", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_existed:
            writer.writeheader()
            writer.writerow(row)