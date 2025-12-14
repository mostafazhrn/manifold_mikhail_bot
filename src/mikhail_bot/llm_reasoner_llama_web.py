#!/usr/bin/env python3
"""
llm_reasoner_llama_web.py

v3/v4-style Ollama reasoner with integrated web search and robust parsing.
Drop-in replacement for previous reasoner; keeps web search order and debug messages.

**Fix included:** supports the legacy call pattern where the second positional
argument may be a numeric ML-probability hint (float) instead of an answers list.
This was the root cause of "'float' object is not iterable" when smart_strategy
called the reasoner with (question, p_ml, context=...).

API kept compatible:
    reason(question, answers=None, use_search=True, context=None)
Also robust to being called as:
    reason(question, 0.523, context="...")  # p_ml hint passed positionally
"""

from __future__ import annotations

import os
import time
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

import requests
from dotenv import load_dotenv

# -------------------- optional OpenAI API --------------------
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# -------------------- optional OpenAI API --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

from openai import OpenAI

_openai_client = None

def call_openai(prompt: str, timeout: int = 60) -> Tuple[bool, str]:
    global _openai_client

    if not OPENAI_API_KEY:
        return False, "No OpenAI key"

    try:
        if _openai_client is None:
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)

        response = _openai_client.responses.create(
            model=OPENAI_MODEL,  # e.g. "gpt-4.1" or "gpt-4o"
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt}
                    ]
                }
            ],
             # tools=[{"type": "web_search"}],   # ← enable only if you want OpenAI browsing
             # tool_choice="auto",               # ← must be enabled together with tools
            max_output_tokens=1024,
            temperature=0.0,
            timeout=timeout,
        )

        text = response.output_text
        return True, text.strip()

    except Exception as e:
        debug(f"OpenAI call failed: {e}")
        return False, str(e)

# -------------------- configuration --------------------
REPO_ROOT = Path(__file__).resolve().parents[0]
dotenv_path = REPO_ROOT / ".env"
try:
    load_dotenv(dotenv_path=str(dotenv_path))
except Exception:
    pass

# Primary names used by rest of code:
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

# Provide legacy-compatible aliases used in some snippets/calls:
OLLAMA_URL = OLLAMA_HOST
MODEL_NAME = OLLAMA_MODEL

SEARCH_RESULTS_MAX = int(os.getenv("SEARCH_RESULTS_MAX", "4"))
SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "6"))

BING_API_KEY = os.getenv("BING_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

DEBUG = os.getenv("LLAMA_REASONER_DEBUG", "1") != "0"


def debug(*args, **kwargs):
    """Consistent debug printer used across the module."""
    if DEBUG:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[LLM_WEB {ts}]", *args, **kwargs)


# -------------------- web search helpers --------------------
# Each provider returns a list of dicts: {title, snippet, url}
def _call_bing(q: str, timeout: int) -> List[Dict[str, str]]:
    if not BING_API_KEY:
        raise RuntimeError("Bing API key missing")
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": q, "count": SEARCH_RESULTS_MAX}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = []
    for it in data.get("webPages", {}).get("value", [])[:SEARCH_RESULTS_MAX]:
        items.append({"title": it.get("name", ""), "snippet": it.get("snippet", ""), "url": it.get("url", "")})
    return items


def _call_brave(q: str, timeout: int) -> List[Dict[str, str]]:
    if not BRAVE_API_KEY:
        raise RuntimeError("Brave API key missing")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-API-Key": BRAVE_API_KEY}
    params = {"q": q, "size": SEARCH_RESULTS_MAX}
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = []
    for it in data.get("results", [])[:SEARCH_RESULTS_MAX]:
        items.append({"title": it.get("title", ""), "snippet": it.get("snippet", ""), "url": it.get("url", "")})
    return items


def _call_serpapi(q: str, timeout: int) -> List[Dict[str, str]]:
    if not SERPAPI_KEY:
        raise RuntimeError("SerpAPI key missing")
    url = "https://serpapi.com/search.json"
    params = {"q": q, "engine": "google", "api_key": SERPAPI_KEY, "num": SEARCH_RESULTS_MAX}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = []
    for it in data.get("organic_results", [])[:SEARCH_RESULTS_MAX]:
        items.append({"title": it.get("title", ""), "snippet": it.get("snippet", ""), "url": it.get("link", "")})
    return items


def _call_serper(q: str, timeout: int) -> List[Dict[str, str]]:
    if not SERPER_API_KEY:
        raise RuntimeError("Serper API key missing")
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    r = requests.post(url, json={"q": q, "num": SEARCH_RESULTS_MAX}, headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = []
    for it in data.get("organic", [])[:SEARCH_RESULTS_MAX]:
        items.append({"title": it.get("title", ""), "snippet": it.get("snippet", ""), "url": it.get("link", "")})
    return items


def _call_duckduckgo(q: str, timeout: int) -> List[Dict[str, str]]:
    url = "https://api.duckduckgo.com/"
    params = {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    items = []
    if data.get("AbstractText"):
        items.append({"title": "DDG Abstract", "snippet": data.get("AbstractText"), "url": data.get("AbstractURL", "")})
    for t in data.get("RelatedTopics", [])[:SEARCH_RESULTS_MAX]:
        if isinstance(t, dict):
            text = t.get("Text") or t.get("Result") or ""
            items.append({"title": t.get("Name", "Related"), "snippet": text, "url": t.get("FirstURL", "")})
    return items[:SEARCH_RESULTS_MAX]


def fetch_search_results(question: str, timeout: int = SEARCH_TIMEOUT) -> List[Dict[str, str]]:
    """
    Try providers in order: bing, brave, serpapi, serper, duckduckgo.
    Returns the first non-empty list of results or empty list if none succeeded.
    """
    providers = [
        ("bing", _call_bing),
        ("brave", _call_brave),
        ("serpapi", _call_serpapi),
        ("serper", _call_serper),
        ("duckduckgo", _call_duckduckgo),
    ]
    last_err = None
    for name, func in providers:
        try:
            if name == "bing" and not BING_API_KEY:
                debug("skip bing (no key)")
                continue
            if name == "brave" and not BRAVE_API_KEY:
                debug("skip brave (no key)")
                continue
            if name == "serpapi" and not SERPAPI_KEY:
                debug("skip serpapi (no key)")
                continue
            if name == "serper" and not SERPER_API_KEY:
                debug("skip serper (no key)")
                continue
            debug(f"search via {name}")
            res = func(question, timeout=timeout)
            if res:
                debug(f"{name} returned {len(res)} results")
                return res
        except Exception as e:
            last_err = e
            debug(f"{name} failed: {e}")
            continue
    debug(f"no search provider succeeded. last error: {last_err}")
    return []


# -------------------- prompt builder --------------------
def _clean_answers(answers: Optional[List[str]]) -> Optional[List[str]]:
    if not answers:
        return None
    clean = []
    for a in answers:
        s = re.sub(r"\(.*?prob.*?[:=]?\s*[0-9.]+.*?\)", "", a, flags=re.I)
        s = s.strip()
        clean.append(s)
    return clean


def _answers_imply_binary(answers: List[str]) -> bool:
    if len(answers) != 2:
        return False
    a0 = answers[0].strip().lower()
    a1 = answers[1].strip().lower()
    yes_set = {"yes", "y", "true", "t"}
    no_set = {"no", "n", "false", "f"}
    return (a0 in yes_set and a1 in no_set) or (a1 in yes_set and a0 in no_set)


def build_prompt(question: str, answers: Optional[List[str]], web_ctx: Optional[str], detected_hint: Optional[str]) -> str:
    q = question.strip()
    choices_block = ""
    if answers:
        clean = _clean_answers(answers)
        for i, a in enumerate(clean, start=1):
            choices_block += f"{i}. {a}\n"

    web_section = f"WEB CONTEXT:\n{web_ctx}\n" if web_ctx else ""
    hint_line = f"PREFERRED_TYPE_HINT: {detected_hint}\n" if detected_hint else ""

    instruction = ( 
    "You are an LLM that must answer ONLY in valid, minified JSON with the exact schema below."
    "No explanations, no markdown, no natural language outside JSON. Never include comments."

    "You must output ONLY ONE JSON OBJECT."
    "You will be given a market question and answer options."
    "Never output explanations, natural language, markdown, or multiple JSON objects."
    "Your job:"
    "1. Interpret the question."
    "2. Reason step-by-step INTERNALLY (not in the output)."
    "3. Output the final JSON object strictly in JSON even your thinking is it happended to be shown it shall be in JSON your answer shall only be in this form:"
    "{"
        "/best_option/: /<the EXACT ID of the answer you choose>/,"
        "/confidence/: /<float between 0 and 1>/,"
        "/reasoning/: /<short explanation, 1–3 sentences>/"        
    "}"

        " Strict Rules:"
        "- best_option: must match one of the provided answers IDs EXACTLY. or YES/NO if binary."
        "- confidence: must be a float (e.g., 0.52). Never a string."
        "- reasoning: must be short and factual (not more than 3 sentences)."
        "- Never invent options."
        "- Output must be EXACTLY one JSON object"
        "- Never break JSON format."
        "- Never include markdown or backticks in output."
        "- No characters before or after the JSON. No trailing text. No intro text."
        "- No duplicate JSON objects."
        "- No additional sentences or comments."

        "You MUST always produce valid JSON that can be parsed by Python json.loads()."
        "If you are uncertain, choose the most plausible option but still return valid JSON."
        "If the question is unanswerable or nonsensical, return the option that is LEAST unlikely."
   )

    prompt = "\n".join([instruction, hint_line, f"QUESTION: {q}", f"OPTIONS:\n{choices_block}" if choices_block else "", web_section, "JSON ONLY."])
    return prompt


# -------------------- Ollama caller (robust to streaming) --------------------
def call_ollama(prompt: str, model: str = MODEL_NAME, host: str = OLLAMA_URL, timeout: int = 60) -> Tuple[bool, str]:
    """
    Call Ollama /api/generate. Return (ok, text) where ok indicates HTTP success.
    Handles direct JSON responses and streaming/newline-delimited JSON fragments.
    """
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "max_tokens": 1024, "temperature": 0.0}
    try:
        debug(f"calling ollama @ {url} model={model}")
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()

        # Try standard JSON
        try:
            js = r.json()
            if isinstance(js, dict):
                # direct response fields
                if "response" in js and isinstance(js["response"], str) and js["response"].strip():
                    return True, js["response"].strip()
                if "text" in js and isinstance(js["text"], str) and js["text"].strip():
                    return True, js["text"].strip()
                # sometimes outputs are embedded
                return True, json.dumps(js)
        except Exception:
            pass

        # If not parseable JSON, try streaming / concatenated JSON fragments from r.text
        body = r.text
        pieces: List[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                frag = json.loads(line)
                if isinstance(frag, dict):
                    resp = frag.get("response") or frag.get("text") or frag.get("output")
                    if isinstance(resp, str) and resp:
                        pieces.append(resp)
                        continue
                    thinking = frag.get("thinking")
                    if isinstance(thinking, str) and thinking:
                        pieces.append(thinking)
                        continue
                else:
                    pieces.append(str(frag))
            except Exception:
                # not JSON per line
                pieces.append(line)
        if pieces:
            out = "".join(pieces).strip()
            return True, out

        # Fallback: raw body
        return True, body
    except Exception as e:
        debug(f"ollama call failed: {e}")
        return False, str(e)

# -------------------- JSON extraction helper --------------------
def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Find and parse the FIRST balanced JSON object in text.
    Returns (parsed_obj, raw_json_fragment) or (None, None) if not found/parsable.
    This is robust to noise, duplicated objects, and text around JSON.
    """
    if not text:
        return None, None

    # Find each '{' and attempt to find a matching '}' by tracking brace depth
    idx = text.find('{')
    while idx != -1:
        depth = 0
        start = None
        for i in range(idx, len(text)):
            ch = text[i]
            if ch == '{':
                if start is None:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start:i+1]
                    try:
                        parsed = json.loads(candidate)
                        return parsed, candidate
                    except Exception:
                        # Parsing failed for this candidate; break to try next '{'
                        break
        # try next '{' occurrence
        idx = text.find('{', idx + 1)

    # fallback: try to find any { ... } with a simple regex (last resort)
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        candidate = m.group(1)
        try:
            parsed = json.loads(candidate)
            return parsed, candidate
        except Exception:
            return None, None

    return None, None

# -------------------- Main reasoning function --------------------
def reason(question: str,
           answers: Optional[List[str]] = None,
           use_search: bool = True,
           context: Optional[str] = None) -> Dict[str, Any]:
    """
    Clean + simplified reasoning function.
    - No JSON molds
    - No normalization
    - No forcing structure
    - LLM outputs free text
    - We extract JSON ONLY if present
    - Output is always: a *flat* dict matching smart_strategy expectations
    """

    debug("reason: start")

    # Convert question to string
    if not isinstance(question, str):
        question = str(question)

    # Convert answers to clean list (if any)
    clean_answers = None
    if isinstance(answers, (list, tuple)):
        clean_answers = _clean_answers(list(answers))

    # Build web snippet
    web_ctx = ""
    if use_search:
        try:
            results = fetch_search_results(question, timeout=SEARCH_TIMEOUT)
            if results:
                parts = []
                for r in results:
                    title = (r.get("title") or "").strip()
                    snippet = (r.get("snippet") or "").strip()
                    if title and snippet:
                        parts.append(f"{title}: {snippet}")
                    elif snippet:
                        parts.append(snippet)
                web_ctx = " \n".join(parts)[:1600]
        except Exception as e:
            debug(f"search failed: {e}")

    # Build prompt WITHOUT JSON schema instructions
    prompt = build_prompt(question, clean_answers, web_ctx, None)

    # --- Try OpenAI API first ---
    use_openai = False
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        ok, resp_text = call_openai(prompt)
        if ok:
            debug(f"OpenAI returned {len(resp_text)} chars")
            parsed_obj, _ = extract_json_object(resp_text)
            if parsed_obj:
                debug("Using OpenAI result")
                use_openai = True
                # ---------- BINARY MARKET OUTPUT ----------
                if clean_answers and _answers_imply_binary(clean_answers):
                    bo = str(parsed_obj.get("best_option", "")).strip().lower()
                    try:
                        conf_val = float(parsed_obj.get("confidence", 0.0))
                    except:
                        conf_val = 0.0
                    yes_pct = 0
                    if bo in ("yes", "y", "true", "t", "1"):
                        yes_pct = int(conf_val * 100) if conf_val <= 1.01 else int(conf_val)
                    elif bo in ("no", "n", "false", "f", "0"):
                        p = conf_val if conf_val <= 1.01 else conf_val / 100.0
                        yes_pct = int(max(0.0, (1.0 - p) * 100))
                    else:
                        try:
                            raw_yes = parsed_obj.get("yes", None)
                            if raw_yes is not None:
                                y = float(raw_yes)
                                yes_pct = int(y if y > 1.01 else y * 100)
                            else:
                                yes_pct = int(conf_val * 100)
                        except:
                            yes_pct = int(conf_val * 100)
                    yes_pct = max(0, min(100, yes_pct))
                    return {"yes": yes_pct, "reasoning": parsed_obj.get("reasoning", "")}
                else:
                    best = parsed_obj.get("best_option", None)
                    conf = parsed_obj.get("confidence", 0.0)
                    reas = parsed_obj.get("reasoning", "")
                    try:
                        conf_f = float(conf)
                        if conf_f > 1.01:
                            conf_f = conf_f / 100.0
                    except:
                        conf_f = 0.0
                    return {"best_option": best, "confidence": float(conf_f), "reasoning": reas}
        else:
            debug("OpenAI failed or returned invalid JSON, falling back to normal pipeline")

    # --- Fall back to normal Ollama + Bing/Brave/Serp/Serper/DDG pipeline ---
    if not use_openai:
        ok, resp_text = call_ollama(prompt)
        debug(f"ollama returned {len(resp_text)} chars")


    if not ok:
        return {
            "best_option": None,
            "confidence": 0.0,
            "reasoning": "OLLAMA ERROR"
        }

    # Try to extract the FIRST valid JSON object from the raw response.
    # Use robust extractor to handle extra text, duplicate objects, etc.
    parsed_obj, cleaned_fragment = extract_json_object(resp_text)

    # If extractor failed, keep parsed_obj None and cleaned_fragment as the original raw text.
    if parsed_obj is None:
        cleaned_fragment = resp_text if resp_text else None

    # ---------- Final return (patched for smart_strategy compatibility) ----------
    if parsed_obj is not None:
        # Determine if this is binary (YES/NO) or multi-choice (MCQ)
        is_binary = False
        if clean_answers and _answers_imply_binary(clean_answers):
            is_binary = True

        # ---------- BINARY MARKET OUTPUT ----------
        if is_binary:
            # Expect LLM returned {"best_option": "YES", "confidence": 0.63, ...}
            bo = str(parsed_obj.get("best_option", "")).strip().lower()

            try:
                conf_val = float(parsed_obj.get("confidence", 0.0))
            except Exception:
                conf_val = 0.0

            if bo in ("yes", "y", "true", "t", "1"):
                yes_pct = int(conf_val * 100) if conf_val <= 1.01 else int(conf_val)
            elif bo in ("no", "n", "false", "f", "0"):
                # If model said NO with confidence, interpret yes_pct as 1 - conf
                p = conf_val if conf_val <= 1.01 else conf_val / 100.0
                yes_pct = int(max(0.0, (1.0 - p) * 100))
            else:
                # If no best_option or unknown token, try to use a 'yes' field if present
                try:
                    raw_yes = parsed_obj.get("yes", None)
                    if raw_yes is not None:
                        y = float(raw_yes)
                        yes_pct = int(y if y > 1.01 else y * 100)
                    else:
                        yes_pct = int(conf_val * 100)
                except Exception:
                    yes_pct = int(conf_val * 100)

            # Bound yes_pct
            yes_pct = max(0, min(100, yes_pct))

            # Return exactly in the legacy binary format:
            return {
                "yes": yes_pct,
                "reasoning": parsed_obj.get("reasoning", "")
            }

        # ---------- MULTI-CHOICE OUTPUT ----------
        # Smart strategy requires:
        # {"best_option": "<id>", "confidence": float, "reasoning": "..."}
        best = parsed_obj.get("best_option", None)
        conf = parsed_obj.get("confidence", 0.0)
        reas = parsed_obj.get("reasoning", "")

        # If confidence was provided as 0..100, normalize to 0..1 for clarity here
        try:
            conf_f = float(conf)
            if conf_f > 1.01:
                conf_f = conf_f / 100.0
        except Exception:
            conf_f = 0.0

        return {
            "best_option": best,
            "confidence": float(conf_f),
            "reasoning": reas
        }

    # If parsing failed earlier, return a small predictable fallback object.
    # This ensures downstream code always gets the expected keys.
    return {
        "best_option": None,
        "confidence": 0.0,
        "reasoning": "LLM did not return valid JSON."
    }




# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Robust Ollama LLM reasoner (llama_reasoner_llama_web)"
    )
    parser.add_argument("question", type=str, help="Question text to evaluate")
    parser.add_argument(
        "--answers", "-a", nargs="*", help="Optional list of answers (quoted). If present, MCQ style returned."
    )
    parser.add_argument(
        "--no-search", action="store_true", help="Disable web search (LLM-only)"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Print raw LLM response in debug (if DEBUG enabled)"
    )
    args = parser.parse_args()

    q = args.question
    answers = args.answers if args.answers else None

    # Call the reasoner
    res = reason(q, answers, use_search=not args.no_search)

    # ---------- Patch: print full evaluation ----------
    print("\n=== LLM EVALUATION ===")
    print(json.dumps(res, indent=2))
