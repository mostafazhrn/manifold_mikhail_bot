"""
smart_strategy_v3.py — upgraded, backwards-compatible strategy

Key features:
- Fully supports BINARY, MULTIPLE_CHOICE, FREE_RESPONSE, PSEUDO_NUMERIC (heuristic).
- Keeps existing ML/LLM blend for binary markets (unchanged logic).
- For multi-answer markets:
    * compute ML per-answer scores (use market answers' probability if present; else fallback to
      scoring "question + answer_text" through legacy TFIDF+MODEL)
    * pick top ML answer, then reuse existing binary LLM pipeline by asking:
        "Is answer '<answer_text>' the most likely answer to: '<question>'?"
      - This allows use of evaluate_question_local unchanged.
    * if top answer gets rejected by LLM (NO), test next answer, etc.
- For pseudo-numeric markets: heuristic binary wrap around median (configurable later).
- Conservative: skips markets when core pieces are missing rather than making unsafe bets.
- Keeps existing API and function names: smart_strategy_v3 and smart_strategy.
"""

from typing import Dict, Any, Optional, List, Tuple
import time
import os
from pathlib import Path

# load .env from repo root (no-op if already in env)
REPO_ROOT = Path(__file__).resolve().parents[2]
_dotenv = REPO_ROOT / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(_dotenv))
except Exception:
    pass

# Reuse legacy ML artifacts for backward compatibility
try:
    from src.mikhail_bot import smart_strategy as legacy
except Exception:
    # Fallback (rare) - try direct import of components
    from src.mikhail_bot.smart_strategy import TFIDF, SVD, MODEL, SCALER  # type: ignore

# Import existing LLM reasoner (binary interface)
try:
    from src.mikhail_bot.llm_reasoner_llama_web import evaluate_question_local  # type: ignore
except Exception:
    try:
        from src.mikhail_bot.llm_reasoner_local import evaluate_question_local  # type: ignore
    except Exception:
        evaluate_question_local = None  # gracefully handled later

# -------------------------
# Config / hyperparams (read from env where applicable)
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
ML_CONFIDENT_LOW  = _get_float("ML_CONFIDENT_LOW",  0.30)
ML_GRAY_LOW       = _get_float("ML_GRAY_LOW",       0.50)
ML_GRAY_HIGH      = _get_float("ML_GRAY_HIGH",      0.62)

WEIGHT_ML  = _get_float("WEIGHT_ML",  0.6)
WEIGHT_LLM = _get_float("WEIGHT_LLM", 0.4)

BET_LARGE  = _get_int("BET_LARGE",  50)
BET_MEDIUM = _get_int("BET_MEDIUM", 25)
BET_SMALL  = _get_int("BET_SMALL",  10)

BET_LARGE_THRESHOLD  = _get_float("BET_LARGE_THRESHOLD", 0.15)
BET_MEDIUM_THRESHOLD = _get_float("BET_MEDIUM_THRESHOLD", 0.30)

DEFAULT_MODE = os.getenv("STRATEGY_MODE", "smart").lower()

# numeric median heuristic for pseudo-numeric markets
NUMERIC_HEURISTIC_USE_MEDIAN = True

# -------------------------
# Debug helper
# -------------------------
def debug(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[SMART_V3 {ts}] {msg}")

# -------------------------
# Helpers: ML probability computation
# -------------------------
def _compute_ml_prob_binary(market: Dict[str, Any]) -> Optional[float]:
    """
    Existing ML pipeline for binary-style predictions.
    Returns p_yes in 0..1 or None on failure.
    """
    question = market.get("question", "") or market.get("name", "")
    if not question:
        return None

    try:
        X_sparse = legacy.TFIDF.transform([question])
        X_text = legacy.SVD.transform(X_sparse)
    except Exception as e:
        debug(f"Text transform failed: {e}")
        return None

    X_num = legacy._numeric_features(market)
    if X_num is None:
        debug("Numeric features failure")
        return None

    try:
        X_num_scaled = legacy.SCALER.transform([X_num])
    except Exception as e:
        debug(f"Scaling failed: {e}")
        return None

    try:
        import numpy as _np
        X = _np.hstack([X_text, X_num_scaled])
    except Exception as e:
        debug(f"Feature hstack failed: {e}")
        return None

    try:
        p_yes = legacy.MODEL.predict_proba(X)[0][1]
        return float(p_yes)
    except Exception as e:
        debug(f"ML predict_proba failed: {e}")
        return None

def _compute_ml_probs_for_answers(market: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Compute per-answer ML probabilities for multiple-choice / free-response markets.

    Approach:
    - If market answers already include probability-like fields (probability, p, prob),
      use them (normalize to sum=1).
    - Else: for each answer, build a short text (question + answer text) and score
      with legacy MODEL.predict_proba (same pipeline used for binary). We treat the
      predicted 'yes' probability as the answer score, then normalize across answers.
    Returns dict: {answer_id_or_text: score_in_0_1} or None on failure.
    """
    answers = market.get("answers") or market.get("options") or []
    if not answers:
        debug("No answers found in market for per-answer ML")
        return None

    # Try to read explicit probabilities from market answers
    explicit_scores: Dict[str, float] = {}
    for a in answers:
        key = str(a.get("id") or a.get("text") or a.get("label") or "")
        # try various fields that might hold probabilities
        p = None
        for field in ("probability", "p", "prob", "probabilityPercent", "probPercent", "probability_pct"):
            if field in a and a[field] is not None:
                try:
                    p = float(a[field])
                    # if percent-style (>1), convert to 0..1
                    if p > 1.01:
                        p = p / 100.0
                except Exception:
                    p = None
        if p is not None:
            explicit_scores[key] = max(0.0, min(1.0, p))

    if explicit_scores:
        # normalize to sum=1
        s = sum(explicit_scores.values())
        if s <= 0:
            # fallback uniform
            n = len(explicit_scores)
            return {k: 1.0 / n for k in explicit_scores.keys()}
        return {k: v / s for k, v in explicit_scores.items()}

    # Fallback: compute ML score per (question + answer_text)
    per_answer_scores: Dict[str, float] = {}
    question_text = market.get("question") or market.get("name") or ""
    try:
        texts = []
        keys = []
        for a in answers:
            ans_text = str(a.get("text") or a.get("label") or a.get("id") or "")
            key = str(a.get("id") or ans_text)
            keys.append(key)
            texts.append((question_text + " " + ans_text).strip())

        # Vectorize texts using TFIDF + SVD then use numeric features repeated (copy) to match shapes
        X_sparse = legacy.TFIDF.transform(texts)
        X_text = legacy.SVD.transform(X_sparse)

        # numeric features: use same numeric features but we need same shape for each example
        X_num = legacy._numeric_features(market)
        if X_num is None:
            # use zeros if no numeric features available
            try:
                import numpy as _np
                X_num_scaled = _np.zeros((len(texts), legacy.SCALER.scale_.shape[0]))
            except Exception:
                X_num_scaled = None
        else:
            import numpy as _np
            # repeat numeric vector for each answer
            try:
                X_num_scaled = legacy.SCALER.transform([X_num] * len(texts))
            except Exception:
                X_num_scaled = _np.zeros((len(texts), legacy.SCALER.scale_.shape[0]))

        import numpy as _np
        X = _np.hstack([X_text, X_num_scaled])
        probs = legacy.MODEL.predict_proba(X)[:, 1]  # treat as 'score' for answer
        # ensure non-negative and normalize
        for k, v in zip(keys, probs.tolist()):
            per_answer_scores[k] = float(max(0.0, v))
        s = sum(per_answer_scores.values())
        if s <= 0:
            # uniform fallback
            n = len(per_answer_scores)
            return {k: 1.0 / n for k in per_answer_scores.keys()}
        return {k: v / s for k, v in per_answer_scores.items()}
    except Exception as e:
        debug(f"Per-answer ML scoring failed: {e}")
        return None

# -------------------------
# Bet sizing helper (unchanged)
# -------------------------
def _determine_bet_size(p_ml: float, p_llm: float) -> int:
    agreement = abs(p_ml - p_llm)
    debug(f"Agreement (|p_ml - p_llm|) = {agreement:.4f}")
    if agreement <= BET_LARGE_THRESHOLD:
        return BET_LARGE
    if agreement <= BET_MEDIUM_THRESHOLD:
        return BET_MEDIUM
    return BET_SMALL

# -------------------------
# Utility: make a binary question for LLM to evaluate a candidate answer
# -------------------------
def _make_binary_for_answer(question: str, answer_text: str) -> str:
    # This string will be fed to evaluate_question_local which expects a binary question.
    # Keep it short and explicit.
    return f'Is the answer "{answer_text}" the most likely answer to the question: "{question}"?'

# -------------------------
# Core public function (compatible with Trader)
# -------------------------
MODE = DEFAULT_MODE

def set_mode(mode_str: str):
    global MODE
    mode_str = (mode_str or "").lower()
    if mode_str not in ("smart", "super"):
        raise ValueError("set_mode: invalid mode (expected 'smart' or 'super')")
    MODE = mode_str
    debug(f"MODE set to '{MODE}' via set_mode()")

def smart_strategy_v3(market: Dict[str, Any], debug_mode: bool = False, mode: str = DEFAULT_MODE) -> Optional[Dict[str, Any]]:
    """
    Unified strategy supporting binary, multiple-choice, free-response, and pseudo-numeric.

    Returns same structure as before for binary:
      {"side": "YES"/"NO", "amount": <int>, "p_ml": <0..1>, "p_llm": <0..1>|None, "p_final": <0..1>}
    For multi-answer, returns:
      {"side": <answer_id_or_text>, "amount": <int>, "p_ml": <0..1>, "p_llm": <0..1>, "p_final": <0..1>, "reason_llm": str, "llm_raw": ...}
      Note: side contains the chosen answer id (or text) for multi-choice/free-response.
    """
    if debug_mode:
        debug("smart_strategy_v3 called")

    # identify market type robustly
    market_type = (market.get("outcomeType") or market.get("type") or "").upper()
    question = market.get("question") or market.get("name") or ""
    if not question:
        if debug_mode:
            debug("Missing question text; skipping market")
        return None

    # ---- 1) BINARY markets: keep exact old behaviour ----
    if market_type == "BINARY" or market.get("isBinary") or market.get("outcomeType") == "BINARY":
        p_ml = _compute_ml_prob_binary(market)
        if p_ml is None:
            if debug_mode:
                debug("ML could not compute probability; skipping binary market")
            return None

        if debug_mode:
            debug(f"ML prob p_yes = {p_ml:.4f}")

        # quick confident decisions
        if p_ml >= ML_CONFIDENT_HIGH:
            if debug_mode:
                debug(f"ML confident high -> YES")
            return {"side": "YES", "amount": BET_MEDIUM, "p_ml": p_ml, "p_llm": None, "p_final": p_ml}
        if p_ml <= ML_CONFIDENT_LOW:
            if debug_mode:
                debug(f"ML confident low -> NO")
            return {"side": "NO", "amount": BET_MEDIUM, "p_ml": p_ml, "p_llm": None, "p_final": p_ml}

        # decide whether to call LLM (smart vs super)
        ask_llm = (mode == "super") or (ML_GRAY_LOW <= p_ml <= ML_GRAY_HIGH)
        if not ask_llm:
            side = "YES" if p_ml > 0.5 else "NO"
            return {"side": side, "amount": BET_MEDIUM, "p_ml": p_ml, "p_llm": None, "p_final": p_ml}

        # call LLM binary pipeline
        if evaluate_question_local is None:
            debug("LLM reasoner not available; skipping binary market")
            return None

        context = f"Market ML probability (0..1): {p_ml:.4f}"
        try:
            start = time.time()
            llm_out = evaluate_question_local(question, market_prob=p_ml, context=context)
            elapsed = time.time() - start
            if debug_mode:
                debug(f"LLM call finished in {elapsed:.2f}s")
        except Exception as e:
            debug(f"LLM call failed: {e}")
            return None

        raw_yes = llm_out.get("yes")
        if raw_yes is None:
            debug("LLM returned no 'yes' field; skipping")
            return None

        p_llm = float(raw_yes) / 100.0 if raw_yes > 1.01 else float(raw_yes)
        p_final = WEIGHT_ML * p_ml + WEIGHT_LLM * p_llm
        side = "YES" if p_final >= 0.5 else "NO"
        amount = _determine_bet_size(p_ml, p_llm)
        if debug_mode:
            debug(f"p_ml={p_ml:.4f}, p_llm={p_llm:.4f}, p_final={p_final:.4f} -> {side} amount={amount}")
        return {"side": side, "amount": amount, "p_ml": p_ml, "p_llm": p_llm, "p_final": p_final, "reason_llm": llm_out.get("reasoning", ""), "llm_raw": llm_out}

    # ---- 2) MULTIPLE_CHOICE / FREE_RESPONSE markets ----
    if market_type in ("MULTIPLE_CHOICE", "FREE_RESPONSE") or market.get("answers"):
        if debug_mode:
            debug("Market appears multi-choice / free-response; computing per-answer ML scores")

        per_scores = _compute_ml_probs_for_answers(market)
        if per_scores is None:
            if debug_mode:
                debug("Per-answer ML scoring failed; falling back to LLM-only selection")
            per_scores = {}

        # Build answers list (maintain original answer dicts for id/text lookup)
        answers = market.get("answers") or market.get("options") or []
        # canonical map: key -> (id, text)
        answer_map: Dict[str, Tuple[Any, str]] = {}
        for a in answers:
            aid = a.get("id')") if isinstance(a.get("id"), str) and a.get("id").endswith("'") else a.get("id")
            # some markets use 'id' or numeric; fallback to text
            aid = a.get("id") or a.get("id")  # keep original id
            atext = str(a.get("text") or a.get("label") or a.get("name") or "")
            key = str(aid) if aid is not None else atext
            answer_map[key] = (aid if aid is not None else atext, atext)

        # If per_scores has keys by text instead of id, try to map them
        # Build sorted list of candidates by ML score (descending)
        candidates: List[Tuple[str, float]] = []
        if per_scores:
            # per_scores keys may be ids or text; try to match both
            for k, v in per_scores.items():
                if k in answer_map:
                    candidates.append((k, v))
                else:
                    # try matching by text (case-insensitive)
                    match = None
                    for ak, (aid, atext) in answer_map.items():
                        if ak.lower() == str(k).lower() or atext.lower() == str(k).lower():
                            match = ak
                            break
                    if match:
                        candidates.append((match, v))
                    else:
                        # unknown key; include anyway (will attempt text fallback)
                        candidates.append((k, v))
        else:
            # No ML per-answer scores: fallback to uniform ranking
            for ak in answer_map.keys():
                candidates.append((ak, 1.0))
        # normalize candidate scores
        ssum = sum(v for _, v in candidates) or 1.0
        candidates = [(k, v / ssum) for k, v in candidates]
        # sort descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        if debug_mode:
            debug(f"Top candidates by ML (key, score): {candidates[:6]}")

        # If mode super: we will still evaluate top answers via LLM; if mode smart, only evaluate if ML uncertain
        # define ML certainty for top answer
        top_key, top_ml_score = candidates[0] if candidates else (None, 0.0)
        # If ML is extremely confident (top_ml_score >> others) we may skip LLM and pick top
        # We'll use simple rule: if top_ml_score >= 0.75 -> choose it without LLM
        if top_ml_score >= 0.75 and len(candidates) >= 1:
            chosen_id, chosen_text = answer_map.get(top_key, (top_key, str(top_key)))
            amount = BET_MEDIUM
            if debug_mode:
                debug(f"Top ML answer very confident (score={top_ml_score:.3f}) -> choose {chosen_text}")
            return {"side": chosen_id, "amount": amount, "p_ml": top_ml_score, "p_llm": None, "p_final": top_ml_score}

        # Otherwise, iterate candidates and ask LLM per-answer (binary wrapper) until LLM says YES or we exhaust
        if evaluate_question_local is None:
            debug("LLM reasoner not available; cannot evaluate multi-choice; skipping")
            return None

        for candidate_key, candidate_ml_score in candidates:
            # resolve candidate text
            if candidate_key in answer_map:
                chosen_id, chosen_text = answer_map[candidate_key]
            else:
                # fallback assume key is text
                chosen_id = candidate_key
                chosen_text = str(candidate_key)

            binary_q = _make_binary_for_answer(question, chosen_text)
            context = f"Per-answer ML score (normalized 0..1) for this candidate: {candidate_ml_score:.4f}"
            if debug_mode:
                debug(f"Querying LLM for candidate '{chosen_text}' with ML score {candidate_ml_score:.4f}")
            try:
                start = time.time()
                llm_out = evaluate_question_local(binary_q, market_prob=candidate_ml_score, context=context)
                elapsed = time.time() - start
                if debug_mode:
                    debug(f"LLM answered in {elapsed:.2f}s -> {llm_out}")
            except Exception as e:
                debug(f"LLM call failed for candidate {chosen_text}: {e}")
                continue

            raw_yes = llm_out.get("yes")
            if raw_yes is None:
                debug("LLM returned no 'yes' field for candidate; skipping candidate")
                continue
            p_llm = float(raw_yes) / 100.0 if raw_yes > 1.01 else float(raw_yes)
            # blend ML(top) and LLM using same weights as binary
            p_final = WEIGHT_ML * candidate_ml_score + WEIGHT_LLM * p_llm
            side_decision_yes = p_final >= 0.5
            amount = _determine_bet_size(candidate_ml_score, p_llm)
            if debug_mode:
                debug(f"Candidate '{chosen_text}' p_ml={candidate_ml_score:.4f}, p_llm={p_llm:.4f}, p_final={p_final:.4f} -> {'TAKE' if side_decision_yes else 'REJECT'} amount={amount}")
            if side_decision_yes:
                # return chosen answer id/text as side (Trader/place_trade handles id/text mapping)
                return {"side": chosen_id, "amount": amount, "p_ml": candidate_ml_score, "p_llm": p_llm, "p_final": p_final, "reason_llm": llm_out.get("reasoning", ""), "llm_raw": llm_out}

        # If no candidate accepted by LLM, skip
        if debug_mode:
            debug("No multi-answer candidate accepted by LLM -> skipping market")
        return None

    # ---- 3) PSEUDO_NUMERIC / NUMBER markets (heuristic) ----
    if market_type in ("PSEUDO_NUMERIC", "NUMBER", "SCALAR"):
        if debug_mode:
            debug("Pseudo-numeric market detected; applying conservative median-based heuristic")

        # Try to get min/max or range from market metadata
        min_val = None
        max_val = None
        # Common fields may vary; try safe extraction
        try:
            if "min" in market and "max" in market:
                min_val = float(market["min"])
                max_val = float(market["max"])
            else:
                # Try nested 'scale' or 'range' fields if they exist
                if isinstance(market.get("scale"), dict):
                    min_val = float(market["scale"].get("min", min_val)) if market["scale"].get("min") is not None else min_val
                    max_val = float(market["scale"].get("max", max_val)) if market["scale"].get("max") is not None else max_val
        except Exception:
            min_val = None
            max_val = None

        # Fallback: if we cannot determine bounds, skip numeric market to be safe
        if min_val is None or max_val is None:
            if debug_mode:
                debug("Could not extract numeric bounds; skipping pseudo-numeric market (safe fallback)")
            return None

        median = (min_val + max_val) / 2.0
        # Make a binary question: "Will value be >= median?"
        binary_q = f"Will the value in this market be >= {median:.6g}? Question: {question}"
        # For ML probability we try to use legacy numeric features to obtain a proxy p_ml
        p_ml_proxy = _compute_ml_prob_binary(market) or 0.5

        if debug_mode:
            debug(f"Numeric heuristic: bounds=({min_val},{max_val}), median={median:.4g}, p_ml_proxy={p_ml_proxy:.4f}")

        if evaluate_question_local is None:
            debug("LLM reasoner not available; skipping numeric market")
            return None

        try:
            llm_out = evaluate_question_local(binary_q, market_prob=p_ml_proxy, context=f"Numeric median binary test (median={median:.6g})")
        except Exception as e:
            debug(f"LLM call failed for numeric market: {e}")
            return None

        raw_yes = llm_out.get("yes")
        if raw_yes is None:
            debug("LLM returned no 'yes' field for numeric candidate; skipping")
            return None
        p_llm = float(raw_yes) / 100.0 if raw_yes > 1.01 else float(raw_yes)
        p_final = WEIGHT_ML * p_ml_proxy + WEIGHT_LLM * p_llm
        side = "YES" if p_final >= 0.5 else "NO"
        amount = BET_SMALL  # conservative for numeric heuristic
        if debug_mode:
            debug(f"Numeric decision p_ml={p_ml_proxy:.4f}, p_llm={p_llm:.4f}, p_final={p_final:.4f} -> {side} amount={amount}")
        # Represent side as YES/NO for numeric binary wrapper
        return {"side": side, "amount": amount, "p_ml": p_ml_proxy, "p_llm": p_llm, "p_final": p_final, "reason_llm": llm_out.get("reasoning", ""), "llm_raw": llm_out}

    # ---- Unknown market type: fallback (ask LLM directly if available) ----
    if debug_mode:
        debug(f"Unknown market type '{market_type}'. Falling back to LLM-only binary interpretation.")

    if evaluate_question_local is None:
        debug("LLM reasoner missing; skipping unknown market type")
        return None

    # Ask LLM to treat the market as binary. We'll pass the question and allow LLM to interpret.
    try:
        llm_out = evaluate_question_local(question, market_prob=None, context="Unknown market type — please interpret as binary and answer yes/no.")
    except Exception as e:
        debug(f"LLM fallback failed: {e}")
        return None

    raw_yes = llm_out.get("yes")
    if raw_yes is None:
        debug("LLM returned no 'yes' field in fallback; skipping")
        return None
    p_llm = float(raw_yes) / 100.0 if raw_yes > 1.01 else float(raw_yes)
    p_final = p_llm  # no ML available
    side = "YES" if p_final >= 0.5 else "NO"
    amount = BET_SMALL
    if debug_mode:
        debug(f"Fallback LLM decision -> {side} ({p_final:.4f})")
    return {"side": side, "amount": amount, "p_ml": None, "p_llm": p_llm, "p_final": p_final, "reason_llm": llm_out.get("reasoning", ""), "llm_raw": llm_out}

# keep wrapper name for backward compatibility
def smart_strategy(market: Dict[str, Any], debug: bool = False):
    mode = MODE
    return smart_strategy_v3(market, debug_mode=debug, mode=mode)
