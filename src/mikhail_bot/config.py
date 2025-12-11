from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # NEW preferred field
    api_key: str = os.getenv("MANIFOLD_API_KEY", "")

    # BACKWARD-COMPATIBILITY for old code
    MANIFOLD_API_KEY: str = os.getenv("MANIFOLD_API_KEY", "")

    designated_username: str = os.getenv("DESIGNATED_USERNAME", "MikhailTal")
    paper_mode: bool = os.getenv("PAPER_MODE", "true").lower() in ("1", "true", "yes")

config = Config()
