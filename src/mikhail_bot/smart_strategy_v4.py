"""
smart_strategy_v4.py — unified strategy using ML + LLM with universal LLM handler

- All LLM output normalized via `universal_llm_handler`.
- Combines ML + LLM votes for binary and multi-choice markets.
- Handles numeric / pseudo-numeric markets with median heuristic.
- ML-only fallback if LLM unavailable.
- Cleaner, maintainable, no duplicate logic.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import os
from pathlib import Path
import joblib
import requests
import numpy as np


# -------------------------
# Try to reuse legacy artifacts
# -------------------------
legacy = None
TFIDF = SVD = SCALER = MODEL = None

try:
    from src.mikhail_bot import smart_strategy as legacy  # type: ignore
    TFIDF = getattr(legacy, "TFIDF", None)
    SVD = getattr(legacy, "SVD", None)
    SCALER = getattr(legacy, "SCALER", None)
    MODEL = getattr(legacy, "MODEL", None)
except Exception:
    legacy = None

# -------------------------
# Load artifacts from disk if missing (FIXED)
# -------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "model_artifacts"

if TFIDF is None or SVD is None or SCALER is None or MODEL is None:
    try:
        TFIDF   = joblib.load(ARTIFACT_DIR / "TFIDF.joblib")
        SVD     = joblib.load(ARTIFACT_DIR / "SVD.joblib")
        SCALER  = joblib.load(ARTIFACT_DIR / "SCALER.joblib")
        MODEL   = joblib.load(ARTIFACT_DIR / "MODEL.joblib")
    except Exception as e:
        TFIDF = SVD = SCALER = MODEL = None
        print(f"[SMART_V4] Artifact load failed from {ARTIFACT_DIR}: {e}")

# -------------------------
# LLM reasoner
# -------------------------
try:
    from src.mikhail_bot.llm_reasoner_llama_web import reason as evaluate_question_local
except Exception:
    try:
        from src.mikhail_bot.llm_reasoner_local import evaluate_question_local
    except Exception:
        evaluate_question_local = None

# -------------------------
# Config / hyperparams
# -------------------------
def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

ML_CONFIDENT_HIGH = _get_float("ML_CONFIDENT_HIGH", 0.70)
ML_CONFIDENT_LOW  = _get_float("ML_CONFIDENT_LOW", 0.30)
ML_GRAY_LOW       = _get_float("ML_GRAY_LOW", 0.50)
ML_GRAY_HIGH      = _get_float("ML_GRAY_HIGH", 0.62)

WEIGHT_ML  = _get_float("WEIGHT_ML", 0.6)
WEIGHT_LLM = _get_float("WEIGHT_LLM", 0.4)

BET_LARGE  = _get_int("BET_LARGE", 50)
BET_MEDIUM = _get_int("BET_MEDIUM", 25)
BET_SMALL  = _get_int("BET_SMALL", 10)

BET_LARGE_THRESHOLD  = _get_float("BET_LARGE_THRESHOLD", 0.15)
BET_MEDIUM_THRESHOLD = _get_float("BET_MEDIUM_THRESHOLD", 0.30)


# -------------------------
# Debug helper
# -------------------------
def debug(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[SMART_V4 {ts}] {msg}")

# -------------------------
# Dynamic weight + smarter bet size helpers
# -------------------------
def compute_dynamic_weights(p_ml: float, p_llm: Optional[float]):
    """Compute weights dynamically based on confidence."""
    if p_llm is None:
        return 1.0, 0.0  # Only ML available

    total_conf = p_ml + p_llm
    if total_conf == 0:
        return 0.5, 0.5  # fallback if both are 0

    weight_ml = p_ml / total_conf
    weight_llm = p_llm / total_conf

    # Clip weights to avoid extremes
    weight_ml = max(0.3, min(0.7, weight_ml))
    weight_llm = 1.0 - weight_ml
    return weight_ml, weight_llm

def determine_bet_size_from_prob(p_final: float):
    """Decide bet size based on final blended probability (confidence)."""
    if 0.0 <= p_final < 0.55:
        return BET_SMALL
    elif 0.55 <= p_final < 0.7:
        return BET_MEDIUM
    else:  # 0.7+
        return BET_LARGE

# -------------------------
# Universal LLM handler
# -------------------------
def universal_llm_handler(raw_output: Any, market_type: str):
    if raw_output is None:
        return {"p_llm": None, "best_option": None, "reasoning": "LLM returned None.", "raw": raw_output}

    if not isinstance(raw_output, dict):
        try:
            import json
            raw_output = json.loads(raw_output)
        except Exception:
            return {"p_llm": None, "best_option": None, "reasoning": f"LLM returned non-dict/non-JSON:\n{raw_output}", "raw": raw_output}

    best_option = raw_output.get("best_option")
    confidence  = raw_output.get("confidence")
    reasoning   = raw_output.get("reasoning") or raw_output.get("reason") or ""
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    
    try:
        p_llm = float(confidence)
    except Exception:
        p_llm = None
    if p_llm is not None:
        p_llm = max(0.0, min(1.0, p_llm))

    if best_option is not None:
        best_option = str(best_option)

    return {"p_llm": p_llm, "best_option": best_option, "reasoning": reasoning, "raw": raw_output}

# -------------------------
# Utilities
# -------------------------
def _determine_bet_size(p_ml: float, p_llm: float) -> int:
    agreement = abs(p_ml - (p_llm or 0))
    if agreement <= BET_LARGE_THRESHOLD:
        return BET_LARGE
    if agreement <= BET_MEDIUM_THRESHOLD:
        return BET_MEDIUM
    return BET_SMALL

# -------------------------
# ML computation helpers
# -------------------------
def _compute_ml_prob_binary(market: Dict[str, Any]) -> Optional[float]:
    """
    Compute ML probability for a binary market.
    Returns probability of YES (0..1), blending:
      - 50% ML model
      - 50% market probability
    """
    # Legacy fallback first
    if legacy is not None and hasattr(legacy, "_compute_ml_prob_binary"):
        try:
            return legacy._compute_ml_prob_binary(market)
        except:
            pass

    question = market.get("question") or market.get("name") or ""
    if not question or TFIDF is None or SVD is None or SCALER is None or MODEL is None:
        return None

    try:
        # --- ML model compute ---
        X_text = SVD.transform(TFIDF.transform([question]))
        X_num = np.zeros((SCALER.scale_.shape[0],)) if SCALER else np.zeros((5,))
        X_num_scaled = SCALER.transform([X_num]) if SCALER else X_num
        X = np.hstack([X_text, X_num_scaled])
        p_ml = float(MODEL.predict_proba(X)[0][1])

        # --- Market probability (fallback to 0 if missing) ---
        market_prob = float(market.get("market_probability")
                            or market.get("probability")
                            or 0.0)

        # --- NEW: equal weight blend ---
        blended = 0.5 * p_ml + 0.5 * market_prob
        blended = max(0.0, min(1.0, blended))
        # ---- NO-bias correction (minimal safe patch) ----
        # If ML is not strongly YES, apply a small tilt toward NO.
        if blended < 0.65:  
            blended *= 0.92   # reduce YES probability by ~8%

        return blended

    except Exception as e:
        debug(f"[ERROR _compute_ml_prob_binary] {e}")
        return None

def _compute_ml_probs_for_answers(market: Dict[str, Any], debug_mode=False, market_id=""):
    answers = market.get("answers") or market.get("options") or []
    if not answers:
        return None

    per_scores = {}
    for a in answers:
        key = str(a.get("id") or a.get("text") or "")

        # ML prob from dataset (same as before)
        ml_prob = float(a.get("probability") or 0.0)

        # 🔥 FIX: Real Manifold market probability for each answer
        market_prob = float(a.get("probability") or 0.0)

        # Blend (same formula)
        blended_prob = 0.7 * ml_prob + 0.3 * market_prob

        per_scores[key] = blended_prob

        if debug_mode:
            debug(
                f"[MARKET {market_id}] Option ID={a.get('id')} Text='{a.get('text')}' "
                f"ML={ml_prob}, Market={market_prob}, Blended={blended_prob}"
            )

        per_scores[key] = blended_prob

        if debug_mode:
            debug(f"[MARKET {market_id}] Option ID={a.get('id')} Text='{a.get('text')}' "
                  f"ML={ml_prob}, Market={market_prob}, Blended={blended_prob}")

    ssum = sum(per_scores.values()) or 1.0
    return {k: v/ssum for k, v in per_scores.items()}

### LLM merge logic
def llm_to_p_yes(llm_choice: Optional[str], llm_conf: Optional[float]) -> float:
    """
    Convert LLM (choice, confidence) into P(YES).

    Rules:
    - Confidence < 0.5 → abstain (0.5)
    - YES with strong confidence → conf
    - NO with strong confidence → 1 - conf
    - Any invalid output → abstain
    """
    if llm_conf is None or llm_conf < 0.5:
        return 0.5

    if llm_choice == "YES":
        return llm_conf

    if llm_choice == "NO":
        return 1.0 - llm_conf

    return 0.5

# -------------------------
# Merge helpers
# -------------------------
def merge_ml_llm_binary(p_ml: float, llm_raw: Any):
    """
    Merge ML+market blended probability with LLM probability for binary YES/NO markets.
    """
    llm_out = universal_llm_handler(llm_raw, market_type="BINARY")
    llm_choice = llm_out.get("best_option")    # YES or NO
    llm_conf   = llm_out.get("p_llm")          # 0..1 or None
    reason_llm = llm_out.get("reasoning", "")

    # Convert LLM into YES probability
    # Convert LLM output into YES probability (safe, fool-proof)
    p_llm_yes = llm_to_p_yes(llm_choice, llm_conf)

    # -------------------------
    # Dynamic merge (ML vs LLM)
    # -------------------------
    weight_ml, weight_llm = compute_dynamic_weights(p_ml, p_llm_yes)

    p_final = weight_ml * p_ml + weight_llm * p_llm_yes
    p_final = max(0.0, min(1.0, p_final))

    # -------------------------
    # Side + stake
    # -------------------------
    side = "YES" if p_final >= 0.5 else "NO"
    amount = determine_bet_size_from_prob(p_final)

    return {
        "side": side,
        "amount": amount,
        "p_ml": p_ml,
        "p_llm": p_llm_yes,
        "p_final": p_final,
        "reason_llm": reason_llm,
        "llm_raw": llm_raw,
        "weights": {"ml": weight_ml, "llm": weight_llm}
    }

def merge_ml_llm_mcq(per_scores: Dict[str, float], llm_raw: Any, answer_map: Dict[str, Tuple[Any,str]]):
    llm_out = universal_llm_handler(llm_raw, market_type="MULTIPLE_CHOICE")
    best_option = llm_out.get("best_option")
    reason_llm = llm_out.get("reasoning", "")

    final_candidates = []
    for key, (aid, atext) in answer_map.items():
        ml_score = per_scores.get(str(aid), 0.0)
        llm_conf = llm_out.get("p_llm")
        # If LLM picked this option → apply real confidence
        if str(aid) == str(best_option):
            p_llm = llm_conf if llm_conf is not None else 0.5
        else:
            # LLM confidence contributes 0 to non-selected options
            p_llm = 0.0    

        p_final = WEIGHT_ML * ml_score + WEIGHT_LLM * p_llm

        final_candidates.append((aid, atext, ml_score, p_llm, p_final))

    final_candidates.sort(key=lambda x: x[4], reverse=True)
    chosen_id, chosen_text, top_ml, top_llm, top_final = final_candidates[0]
    amount = determine_bet_size_from_prob(top_final)
    return {"side": chosen_id, "amount": amount, "p_ml": top_ml, "p_llm": top_llm, "p_final": top_final, "reason_llm": reason_llm, "llm_raw": llm_raw}

# -------------------------
# Core strategy function
# -------------------------
# Inside smart_strategy_v4.py (replace existing smart_strategy_v4 function)

from src.mikhail_bot import api  # your api.py fetch functions

def smart_strategy_v4(market: dict, debug_mode=False, mode: str = None):
    # 🔒 Runtime source of truth
    if mode is None:
        mode = os.getenv("STRATEGY_MODE", "smart").lower()

    if mode not in ("smart", "super"):
        debug(f"[WARN] Invalid STRATEGY_MODE='{mode}', falling back to 'smart'")
        mode = "smart"

    if debug_mode:
        debug(f"[MODE] Active strategy mode = {mode.upper()}")
    """
    Unified strategy: binary + MCQ + free-response markets.
    Automatically fetches full market details for MCQs if answers missing.
    Returns dict with side, amount, ML/LLM probabilities, reasoning.
    """
    market_id = market.get("id") or ""
    question = market.get("question") or market.get("name") or ""
    market_type = (market.get("outcomeType") or market.get("type") or "").upper()

    if not question:
        if debug_mode:
            debug(f"[SKIP {market_id}] No question text found.")
        return None

    if debug_mode:
        debug(f"[EVAL {market_id}] Market type: {market_type}, Question: {question}")

    # ------------------------------
    # Fetch full market for MCQ/Free-response if answers missing
    # ------------------------------
    answers = market.get("answers") or market.get("options")
    if market_type in ("MULTIPLE_CHOICE", "FREE_RESPONSE") and not answers:
        if debug_mode:
            debug(f"[DEBUG {market_id}] Answers missing. Fetching full market via API...")
        try:
            full_market = api.get_market(market_id)
            answers = full_market.get("answers") or full_market.get("options")
            if debug_mode:
                debug(f"[DEBUG {market_id}] Fetched answers: {[(a.get('id'), a.get('text')) for a in answers]}")
        except Exception as e:
            if debug_mode:
                debug(f"[ERROR {market_id}] Failed to fetch full market: {e}")
            answers = None

    # ------------------------------
    # Binary market
    # ------------------------------
    if market_type == "BINARY" or market.get("isBinary"):
        # --- Ensure probability exists for BINARY markets ---
        prob = market.get("probability") or market.get("market_probability")

        if prob is None:
            try:
                if debug_mode:
                    debug(f"[FETCH {market_id}] probability missing → fetching full market via API...")
                full = api.get_market(market_id)
                if full and ("probability" in full):
                    prob = full["probability"]
                    market["probability"] = prob  # inject into market object
                    if debug_mode:
                        debug(f"[FETCH {market_id}] Filled missing probability: {prob}")
                else:
                    if debug_mode:
                        debug(f"[WARN {market_id}] Full market fetched but probability still missing.")
            except Exception as e:
                if debug_mode:
                    debug(f"[ERROR {market_id}] Failed to fetch market probability: {e}")
                prob = None

        # Final fallback if still None
        if prob is None:
            if debug_mode:
                debug(f"[WARN {market_id}] Using neutral fallback market probability = 0.5")
            market["probability"] = 0.5
        # ------------------------------
        p_ml = _compute_ml_prob_binary(market)
        # Show market probability
        market_prob = market.get("market_probability") or market.get("probability")
        if debug_mode:
            debug(f"[MARKET {market_id}] Market probability YES: {market_prob}")
        if p_ml is None:
            if debug_mode:
                debug(f"[SKIP {market_id}] ML probability could not be computed.")
            return None

        ####
        # ------------------------------
        # Decide whether to call LLM
        # ------------------------------
        ask_llm = False

        if mode == "super":
            ask_llm = True
        elif ML_GRAY_LOW <= p_ml <= ML_GRAY_HIGH:
            ask_llm = True

        if debug_mode:
            debug(f"[LLM {market_id}] ask_llm={ask_llm} (mode={mode}, p_ml={p_ml:.3f})")

        # ------------------------------
        # Call LLM if required
        # ------------------------------
        llm_raw = None

        if ask_llm and evaluate_question_local:
            try:
                reason = "SUPER MODE" if mode == "super" else "Gray-zone binary"
                if debug_mode:
                    debug(f"[LLM {market_id}] {reason} → calling LLM")
                llm_raw = evaluate_question_local(question)
            except Exception as e:
                if debug_mode:
                    debug(f"[LLM {market_id}] LLM call failed: {e}")
                llm_raw = None


        result = merge_ml_llm_binary(p_ml, llm_raw)
        if debug_mode:
            debug(f"[RESULT {market_id}] {result}")
        return result

    # ------------------------------
    # MCQ / Free-response
    # ------------------------------
    if market_type in ("MULTIPLE_CHOICE", "FREE_RESPONSE"):
        if not answers:
            if debug_mode:
                debug(f"[SKIP {market_id}] No answer options available after fetch. Skipping market.")
            return None

        per_scores = _compute_ml_probs_for_answers({"answers": answers}, debug_mode=debug_mode, market_id=market_id) or {}

        answer_map = {str(a.get("id") or a.get("text")):(a.get("id") or a.get("text"), a.get("text") or "") for a in answers}

        llm_raw = None
        if evaluate_question_local:
            llm_prompt = f"Question: {question}\nChoices:\n"
            for aid, atext in answer_map.values():
                llm_prompt += f"- ID={aid}: {atext}\n"
            if debug_mode:
                debug(f"[LLM {market_id}] Calling LLM reasoner for MCQ/Free-response market...")
            try:
                llm_raw = evaluate_question_local(llm_prompt)
            except Exception as e:
                if debug_mode:
                    debug(f"[LLM {market_id}] Exception during LLM call: {e}")
                llm_raw = None

        try:
            result = merge_ml_llm_mcq(per_scores, llm_raw, answer_map)
            if debug_mode:
                debug(f"[RESULT {market_id}] {result}")
            return result
        except Exception as e:
            if debug_mode:
                debug(f"[ERROR {market_id}] Exception during MCQ merge: {e}\nML scores: {per_scores}\nLLM raw: {llm_raw}")
            return None

    if debug_mode:
        debug(f"[SKIP {market_id}] Market type not handled: {market_type}")
    return None


# Backward compatibility
def smart_strategy(market: Dict[str, Any], debug: bool=False):
    return smart_strategy_v4(market, debug_mode=debug, mode=None)
