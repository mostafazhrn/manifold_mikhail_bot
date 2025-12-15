"""
train_model.py

Train a per-option RandomForest classifier that uses:
 - text features: TFIDF(TITLE+ANSWER) -> SVD
 - market features: option_probability, option_volume, relative_probability, rank, gap_to_next_best
 - optionally any legacy numeric features if present

Expected input: newline-delimited JSON file of resolved markets (default path: data/resolved_markets.jsonl)
Each market should be a dict with keys like:
 - id, question (or name), type/outcomeType
 - answers: list of answer dicts, where each answer dict may contain:
     - text / label, probability (0..1 or 0..100), volume / totalShares / totalBets
     - winner flag (e.g. 'isWinner' or 'won' or compare to market['resolvedOutcome'])
If the file format deviates, adapt the _extract_answer_rows() function.

Usage:
    python tools/train_model.py --input data/resolved_markets.jsonl
"""
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss

ARTIFACT_DIR = Path(__file__).resolve().parent / "mikhail_bot" / "model_artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def _norm_prob(p):
    try:
        p = float(p)
        if p > 1.01:
            p = p / 100.0
        return max(0.0, min(1.0, p))
    except Exception:
        return None

def _extract_answer_rows(market: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert one market into per-answer rows.
    Handles both MCQ (answers/options list)
    and BINARY (pool YES/NO).
    """
    rows = []
    q = market.get("question") or market.get("name") or ""

    # ----------- CASE 1: BINARY MARKET -----------
    if market.get("outcomeType") == "BINARY":
        pool = market.get("pool") or {}
        yes_vol = float(pool.get("YES", 0))
        no_vol = float(pool.get("NO", 0))

        total = yes_vol + no_vol
        if total <= 0:
            yes_prob = no_prob = 0.5
        else:
            yes_prob = yes_vol / total
            no_prob = no_vol / total

        resolved = (market.get("resolution") or "").upper()

        rows.append({
            "market_id": market.get("id"),
            "question": q,
            "answer_text": "YES",
            "option_prob": yes_prob,
            "option_volume": yes_vol,
            "is_winner": 1 if resolved == "YES" else 0,
        })
        rows.append({
            "market_id": market.get("id"),
            "question": q,
            "answer_text": "NO",
            "option_prob": no_prob,
            "option_volume": no_vol,
            "is_winner": 1 if resolved == "NO" else 0,
        })

        return rows

    # ----------- CASE 2: MULTIPLE CHOICE MARKET -----------
    answers = market.get("answers") or market.get("options") or []
    if not answers and "outcomes" in market:
        answers = market["outcomes"]
    if not answers:
        return rows

    resolved = market.get("resolvedOutcome") or market.get("resolution") or market.get("resolved")

    for a in answers:
        text = str(a.get("text") or a.get("label") or a.get("name") or a.get("id") or "")

        p = None
        for f in ("probability", "prob", "p", "probabilityPercent", "probPercent"):
            if f in a and a[f] is not None:
                p = _norm_prob(a[f])
                break

        vol = None
        for vf in ("volume", "totalShares", "totalBets", "betVolume"):
            if vf in a and a[vf] is not None:
                try:
                    vol = float(a[vf])
                except:
                    vol = None

        won = False
        if a.get("isWinner") or a.get("won"):
            won = True
        else:
            if resolved is not None:
                if str(a.get("id")) == str(resolved) or text.strip().lower() == str(resolved).strip().lower():
                    won = True

        rows.append({
            "market_id": market.get("id"),
            "question": q,
            "answer_text": text,
            "option_prob": p,
            "option_volume": vol,
            "is_winner": 1 if won else 0
        })

    return rows

def build_feature_rows(markets: List[Dict[str, Any]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    For each market, compute per-option derived features like normalized probabilities,
    rank and gap to next best. Return:
      texts: list[str] (question + " " + answer)
      X_num: numpy array shape (n_samples, n_numeric_features)
      y: numpy array labels (0/1)
    """
    rows_all = []
    for m in markets:
        rows = _extract_answer_rows(m)
        if not rows:
            continue
        # compute normalizations per market
        probs = [r["option_prob"] for r in rows]
        # if probs missing, try to compute from volumes or skip normalization
        if any(p is None for p in probs):
            # try volumes
            vols = [r["option_volume"] if r["option_volume"] is not None else 0.0 for r in rows]
            s = sum(vols)
            if s > 0:
                for r, v in zip(rows, vols):
                    r["option_prob"] = v / s
            else:
                # uniform if nothing present
                n = len(rows)
                for r in rows:
                    r["option_prob"] = 1.0 / n
        else:
            # ensure numeric and normalize if they don't sum to 1
            vals = [float(p) if p is not None else 0.0 for p in probs]
            s = sum(vals)
            if s <= 0:
                n = len(vals)
                vals = [1.0 / n] * n
            else:
                vals = [v / s for v in vals]
            for r, v in zip(rows, vals):
                r["option_prob"] = float(v)

        # compute ranks & gaps
        sorted_probs = sorted([r["option_prob"] for r in rows], reverse=True)
        for r in rows:
            r["relative_prob"] = r["option_prob"]
            r["rank"] = 1 + sorted_probs.index(r["option_prob"])
            # gap to next best
            next_best = None
            higher = [p for p in sorted_probs if p < r["option_prob"]]
            next_best = higher[0] if higher else 0.0
            r["gap_to_next"] = r["option_prob"] - next_best

        rows_all.extend(rows)

    # assemble features
    texts = []
    numerics = []
    labels = []
    for r in rows_all:
        texts.append((str(r["question"]) + " " + str(r["answer_text"])).strip())
        # numeric vector: option_prob, relative_prob, rank, gap_to_next, option_volume (fill 0)
        numerics.append([
            float(r.get("option_prob") or 0.0),
            float(r.get("relative_prob") or 0.0),
            float(r.get("rank") or 0.0),
            float(r.get("gap_to_next") or 0.0),
            float(r.get("option_volume") or 0.0)
        ])
        labels.append(int(r.get("is_winner", 0)))
    X_num = np.asarray(numerics, dtype=float)
    y = np.asarray(labels, dtype=int)
    return texts, X_num, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", default="data/resolved_markets.jsonl", help="newline JSONL of resolved markets")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path} - please provide resolved markets JSONL")

    markets = []
    with open(input_path, "r", encoding="utf8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                markets.append(json.loads(line))
            except Exception:
                continue

    print(f"[train] loaded {len(markets)} markets from {input_path}")

    texts, X_num, y = build_feature_rows(markets)
    print(f"[train] constructed {len(texts)} per-option rows; positive labels: {int(y.sum())}")

    # Text pipeline
    tfidf = TfidfVectorizer(max_features=16000, ngram_range=(1,2), stop_words="english")
    X_text_sparse = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=128, random_state=args.random_state)
    X_text = svd.fit_transform(X_text_sparse)

    # Scale numeric features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Stack
    X = np.hstack([X_text, X_num_scaled])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    print("[train] training RandomForestClassifier")
    clf = RandomForestClassifier(n_estimators=args.n_estimators, n_jobs=-1, random_state=args.random_state, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    print("[train] classification report:")
    print(classification_report(y_test, y_pred))
    try:
        print("[train] log loss:", log_loss(y_test, y_proba))
    except Exception:
        pass

    # Save artifacts
    joblib.dump(tfidf, ARTIFACT_DIR / "TFIDF.joblib")
    joblib.dump(svd, ARTIFACT_DIR / "SVD.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "SCALER.joblib")
    joblib.dump(clf, ARTIFACT_DIR / "MODEL.joblib")
    print(f"[train] saved artifacts to {ARTIFACT_DIR} (TFIDF, SVD, SCALER, MODEL)")

def train_model():
    """Wrapper so the GUI can call training."""
    return main()

if __name__ == "__main__":
    main()
