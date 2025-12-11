import os
import pickle
import numpy as np
from typing import Dict, Any

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def debug(msg: str):
    print(f"[DEBUG] {msg}")


# ============================================================
# Load machine-learning artifacts ONCE (global)
# ============================================================
debug("Loading SmartStrategy v2 artifacts...")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

vec_dict = pickle.load(open(VECTORIZER_PATH, "rb"))
TFIDF = vec_dict["tfidf"]
SVD = vec_dict["svd"]

with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)

debug("SmartStrategy v2 artifacts loaded successfully.")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
def _numeric_features(m: Dict[str, Any]):
    """Generate the numeric features used during training."""
    try:
        volume = float(m.get("volume", 0.0))
        volume24 = float(m.get("volume24Hours", 0.0))
        liquidity = float(m.get("totalLiquidity", 0.0))
        bettors = float(m.get("uniqueBettorCount", 0.0))
        created = float(m.get("createdTime", 0.0))
        close = float(m.get("closeTime", 0.0))

        ms_day = 86400000.0
        days_to_close = (close - created) / ms_day
        age_days = 0.0

        return np.array([
            np.log1p(volume),
            np.log1p(volume24),
            np.log1p(liquidity),
            bettors,
            days_to_close,
            age_days,
        ], dtype=float)
    except Exception as e:
        debug(f"[ERROR] numeric feature creation failed: {e}")
        return None


# ============================================================
# CORE SMART STRATEGY FUNCTION
# ============================================================
def smart_strategy(market: Dict[str, Any], debug: bool = False):
    """
    Function interface required by trader.py:
        decision = smart_strategy(m, debug=True)

    Returns dict: { 'side': 'YES'/'NO', 'amount': int } or None
    """

    # ----------------- 1. TEXT PROCESSING -----------------
    question = market.get("question", "")
    if not question:
        if debug:
            print("[DEBUG] No question → skipping.")
        return None

    try:
        X_sparse = TFIDF.transform([question])
        X_text = SVD.transform(X_sparse)
    except Exception as e:
        if debug:
            print("[DEBUG] ❌ Text transform failed:", e)
        return None

    # ----------------- 2. NUMERIC FEATURES -----------------
    X_num = _numeric_features(market)
    if X_num is None:
        return None

    try:
        X_num_scaled = SCALER.transform([X_num])
    except Exception as e:
        if debug:
            print("[DEBUG] ❌ Scaling failed:", e)
        return None

    # ----------------- 3. COMBINE & PREDICT -----------------
    X = np.hstack([X_text, X_num_scaled])

    try:
        p_yes = MODEL.predict_proba(X)[0][1]
    except Exception as e:
        if debug:
            print("[DEBUG] ❌ Prediction failed:", e)
        return None

    if debug:
        print(f"[DEBUG] p_yes={p_yes:.3f}")

    # ----------------- 4. DECISION LOGIC -----------------
    # YES if confident
    if p_yes > 0.62:
        if debug:
            print("[DEBUG] Decision: YES (p_yes > 0.62)")
        return {"side": "YES", "amount": 25}

    # NO if p_yes is weak
    if p_yes < 0.50:
        if debug:
            print("[DEBUG] Decision: NO (p_yes < 0.50)")
        return {"side": "NO", "amount": 25}

    # Skip the uncertain middle band
    if debug:
        print("[DEBUG] Decision: SKIP (0.50 ≤ p_yes ≤ 0.62)")
    return None
