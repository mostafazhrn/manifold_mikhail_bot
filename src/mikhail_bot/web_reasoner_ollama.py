#!/usr/bin/env python3
"""
web_reasoner_ollama.py

- Uses the first available web search provider: BING -> SERPAPI -> BRAVE -> DUCKDUCKGO (fallback)
- Calls local Ollama LLM at localhost:11434 to evaluate snippets and return YES/NO probabilities.
- Debug prints every step. Prints final weighted choice (YES/NO) with percent.
- Usage:
    python3 web_reasoner_ollama.py "Will Pakistan default by May 2026?" [market_yes_prob]
"""

import os
import sys
import time
import json
import requests
from typing import List, Dict, Optional, Tuple

# -----------------------------
# CONFIG (environment overrides)
# -----------------------------
BING_API_KEY = os.getenv("BING_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # change to your installed model name if needed

# -----------------------------
# TUNABLES
# -----------------------------
SEARCH_RESULT_COUNT = 6
WEB_TIMEOUT_SECONDS = 10
LLM_TIMEOUT_SECONDS = 30

# Blend weights (sum should be 1.0)
WEIGHT_LLM = 0.7
WEIGHT_MARKET = 0.3

# -----------------------------
# DEBUG
# -----------------------------
def debug(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[DEBUG {ts}] {msg}")

# -----------------------------
# Choose provider by priority
# -----------------------------
def choose_search_provider() -> str:
    if BING_API_KEY:
        return "bing"
    if SERPAPI_KEY:
        return "serpapi"
    if BRAVE_API_KEY:
        return "brave"
    return "duckduckgo"

# -----------------------------
# SEARCH FUNCTIONS
# -----------------------------
def search_bing(query: str, count: int = SEARCH_RESULT_COUNT) -> List[Dict]:
    debug("search_bing: starting")
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count, "textDecorations": False, "textFormat": "Raw"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=WEB_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("webPages", {}).get("value", [])[:count]:
            items.append({
                "title": it.get("name"),
                "snippet": it.get("snippet"),
                "url": it.get("url")
            })
        debug(f"search_bing: got {len(items)} results")
        return items
    except Exception as e:
        debug(f"search_bing: failed: {e}")
        return []

def search_serpapi(query: str, count: int = SEARCH_RESULT_COUNT) -> List[Dict]:
    debug("search_serpapi: starting")
    url = "https://serpapi.com/search"
    params = {"q": query, "engine": "google", "num": count, "api_key": SERPAPI_KEY}
    try:
        r = requests.get(url, params=params, timeout=WEB_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("organic_results", [])[:count]:
            items.append({
                "title": it.get("title"),
                "snippet": it.get("snippet") or it.get("description"),
                "url": it.get("link")
            })
        debug(f"search_serpapi: got {len(items)} results")
        return items
    except Exception as e:
        debug(f"search_serpapi: failed: {e}")
        return []

def search_brave(query: str, count: int = SEARCH_RESULT_COUNT) -> List[Dict]:
    debug("search_brave: starting")
    url = "https://api.search.brave.com/res/v1/web"
    headers = {"Accept": "application/json"}
    params = {"q": query, "size": count}
    if BRAVE_API_KEY:
        headers["Authorization"] = f"Bearer {BRAVE_API_KEY}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=WEB_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("web", {}).get("results", [])[:count]:
            items.append({
                "title": it.get("title"),
                "snippet": it.get("description"),
                "url": it.get("url")
            })
        debug(f"search_brave: got {len(items)} results")
        return items
    except Exception as e:
        debug(f"search_brave: failed: {e}")
        return []

def search_duckduckgo(query: str, count: int = SEARCH_RESULT_COUNT) -> List[Dict]:
    debug("search_duckduckgo: starting (instant answer, limited)")
    url = "https://api.duckduckgo.com/"
    params = {"q": query, "format": "json", "no_redirect": 1, "skip_disambig": 1}
    try:
        r = requests.get(url, params=params, timeout=WEB_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        items = []
        abstract = data.get("AbstractText")
        if abstract:
            items.append({"title": data.get("Heading", "DuckDuckGo Abstract"), "snippet": abstract, "url": data.get("AbstractURL", "")})
        for topic in data.get("RelatedTopics", [])[:count]:
            if isinstance(topic, dict):
                snippet = topic.get("Text") or topic.get("Result")
                title = topic.get("Name") or (snippet[:80] if snippet else "RelatedTopic")
                urlt = topic.get("FirstURL") or ""
                items.append({"title": title, "snippet": snippet, "url": urlt})
        debug(f"search_duckduckgo: got {len(items)} items")
        return items[:count]
    except Exception as e:
        debug(f"search_duckduckgo: failed: {e}")
        return []

# -----------------------------
# Ollama LLM call
# -----------------------------
def call_ollama_llm(question: str, snippets: List[Dict]) -> Tuple[float, float, str]:
    """
    Returns (yes_pct, no_pct, reasoning_text)
    If Ollama not reachable or parsing fails, returns neutral (50,50).
    """
    debug("call_ollama_llm: preparing prompt")
    if not snippets:
        snippet_block = "(no snippets found)"
    else:
        lines = []
        for i, s in enumerate(snippets[:SEARCH_RESULT_COUNT]):
            t = s.get("snippet") or s.get("title") or ""
            lines.append(f"{i+1}. {t} (source: {s.get('url','')})")
        snippet_block = "\n".join(lines)

    prompt = f"""
You are an objective evaluator for a binary question.

Question:
\"\"\"{question}\"\"\"

Here are web snippets relevant to the question:
{snippet_block}

Task:
1) Based on the snippets, estimate the probability (0-100) that the answer is YES.
2) Also provide the probability for NO.
3) Provide a concise 1-2 sentence reasoning.

Return ONLY a JSON object like:
{{"yes": 23.5, "no": 76.5, "reasoning": "short explanation"}}
If uncertain, be conservative and say so.
"""

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": 400,
        "temperature": 0.0
    }

    try:
        debug(f"call_ollama_llm: calling Ollama at {url} (model={OLLAMA_MODEL})")
        r = requests.post(url, json=payload, timeout=LLM_TIMEOUT_SECONDS)
        r.raise_for_status()
        data = r.json()
        debug(f"call_ollama_llm: raw response keys: {list(data.keys())}")

        # Ollama responses can differ by version — try to extract text robustly
        text = None
        # Common places:
        # - data.get("completion") or data.get("response") or data.get("text")
        if "completion" in data and isinstance(data["completion"], str):
            text = data["completion"]
        elif "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            # Some Ollama outputs wrap choices with message/text
            first = data["choices"][0]
            # possible keys: "message", "content", "text"
            if isinstance(first, dict):
                text = first.get("message") or first.get("text") or first.get("content")
                if isinstance(text, dict):
                    # if message is dict with 'content'
                    text = text.get("content") if text.get("content") else str(text)
        elif "response" in data:
            text = data["response"]
        elif "output" in data:
            text = data["output"]
        else:
            # Fallback: stringify response
            text = json.dumps(data)

        if not text:
            debug("call_ollama_llm: could not find textual response; using raw JSON string")
            text = json.dumps(data)

        debug(f"call_ollama_llm: model text (first 800 chars): {text[:800]!s}")

        # Try strict JSON parse
        try:
            parsed = json.loads(text)
            yes = float(parsed.get("yes", 50.0))
            no = float(parsed.get("no", 50.0))
            reason = parsed.get("reasoning", parsed.get("explanation", "") ) or ""
            return yes, no, reason
        except Exception:
            # If not strict JSON, try to heuristically extract numbers
            import re
            # look for "yes: 12" or '"yes": 12' or "YES: 12"
            m_yes = re.search(r'["\']?yes["\']?\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)
            m_no = re.search(r'["\']?no["\']?\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)', text, flags=re.IGNORECASE)

            if m_yes:
                yes_v = float(m_yes.group(1))
                if m_no:
                    no_v = float(m_no.group(1))
                else:
                    no_v = max(0.0, 100.0 - yes_v)
                reason = text.strip()[:400]
                return yes_v, no_v, reason

            # final fallback: try to find two numbers in the text and assume they are yes/no
            nums = re.findall(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', text)
            if len(nums) >= 2:
                yes_v = float(nums[0])
                no_v = float(nums[1])
                reason = text.strip()[:400]
                return yes_v, no_v, reason

            debug("call_ollama_llm: could not parse numeric probabilities from model output; returning neutral 50/50")
            return 50.0, 50.0, text

    except Exception as e:
        debug(f"call_ollama_llm: request failed: {e}")
        return 50.0, 50.0, f"Ollama call failed: {e}"

# -----------------------------
# Evaluation flow
# -----------------------------
def evaluate_question(question: str, market_prob: Optional[float] = None) -> Dict:
    debug("evaluate_question: start")
    provider = choose_search_provider()
    debug(f"evaluate_question: chosen provider = {provider}")

    # Fetch snippets
    snippets = []
    if provider == "bing":
        snippets = search_bing(question)
    elif provider == "serpapi":
        snippets = search_serpapi(question)
    elif provider == "brave":
        snippets = search_brave(question)
    else:
        snippets = search_duckduckgo(question)

    if not snippets:
        debug("evaluate_question: no snippets obtained; returning neutral result")
        print("\nFINAL OUTPUT: NO WEB DATA — Neutral result\nYES = 50.0% | NO = 50.0%\n")
        return {"yes": 50.0, "no": 50.0, "reasoning": "No web search results available."}

    debug(f"evaluate_question: obtained {len(snippets)} snippets; sample snippet[0]: {snippets[0].get('snippet','')[:200]}")

    # Call Ollama
    yes_p, no_p, reason = call_ollama_llm(question, snippets)

    # Normalize
    s = (yes_p + no_p)
    if s <= 0:
        yes_p, no_p = 50.0, 50.0
    else:
        yes_p = (yes_p / s) * 100.0
        no_p = (no_p / s) * 100.0

    debug(f"LLM normalized: YES={yes_p:.2f}, NO={no_p:.2f}")
    print("\nLLM (web-based) estimate:")
    print(f"  YES = {yes_p:.2f}%")
    print(f"  NO  = {no_p:.2f}%")
    print(f"  Reasoning: {reason}\n")

    # Combine with market probability if provided
    if market_prob is not None:
        if market_prob <= 1.01:
            market_yes_pct = market_prob * 100.0
        else:
            market_yes_pct = market_prob
        debug(f"Combining with market_prob={market_yes_pct:.2f}% (weights LLM={WEIGHT_LLM}, Market={WEIGHT_MARKET})")
        final_yes = WEIGHT_LLM * yes_p + WEIGHT_MARKET * market_yes_pct
        final_no = 100.0 - final_yes
        final_yes = max(0.0, min(final_yes, 100.0))
        final_no = max(0.0, min(final_no, 100.0))

        # Decide side
        if final_yes >= 50.0:
            side = "YES"
            weight = final_yes
        else:
            side = "NO"
            weight = final_no

        print("FINAL WEIGHTED DECISION (LLM + MARKET):")
        print(f"  => {side}  ({weight:.2f}%)")
        print(f"\nRaw blended probabilities: YES={final_yes:.2f}% | NO={final_no:.2f}%")
        return {"yes": final_yes, "no": final_no, "reasoning": reason, "side": side, "weight": weight}
    else:
        # No market blending — use LLM-only
        if yes_p >= 50.0:
            side = "YES"
            weight = yes_p
        else:
            side = "NO"
            weight = no_p
        print("FINAL WEIGHTED DECISION (LLM-only):")
        print(f"  => {side}  ({weight:.2f}%)")
        print(f"\nRaw LLM probabilities: YES={yes_p:.2f}% | NO={no_p:.2f}%")
        return {"yes": yes_p, "no": no_p, "reasoning": reason, "side": side, "weight": weight}

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 web_reasoner_ollama.py \"Your question here\" [market_yes_prob]")
        sys.exit(1)

    question = sys.argv[1]
    market_prob = None
    if len(sys.argv) >= 3:
        try:
            market_prob = float(sys.argv[2])
        except Exception:
            market_prob = None

    start = time.time()
    result = evaluate_question(question, market_prob)
    debug(f"Total elapsed: {time.time() - start:.2f}s")
    print("\n=== RAW OUTPUT JSON ===")
    print(json.dumps(result, indent=2))
