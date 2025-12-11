#!/usr/bin/env python3
"""
llm_reasoner_local.py (cleaned output)

- Local-only LLM reasoning (no web/search APIs).
- Calls local Ollama at OLLAMA_HOST to get YES/NO probabilities for a question.
- Handles streamed JSON output from Ollama.
- Optionally provide: market_yes_prob and short local context/argument.
- Debug prints kept, but large prompt/model buffer outputs are hidden for readability.
- Prints concise LLM result + up to 400 chars of the LLM reasoning + final weighted decision.

USAGE:
    python3 llm_reasoner_local.py "Your question here" [market_yes_prob] [short_context_string]
"""

import os
import sys
import time
import json
import requests
from typing import Tuple, Optional

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://172.25.192.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")  # change if your model name differs

# LLM call/timeouts
LLM_TIMEOUT_SECONDS = 300  # 5 minutes

# Blend weights (LLM vs market). Must sum to 1.0
WEIGHT_LLM = 0.7
WEIGHT_MARKET = 0.3

# Debug print helper
def debug(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[DEBUG {ts}] {msg}")

# -----------------------------
# Build prompt for the local LLM
# -----------------------------
def build_prompt(question: str, context: Optional[str] = None) -> str:
    ctx_block = ""
    if context:
        ctx_block = f"\n\nAdditional context / facts (from user):\n{context}\n\n"
    prompt = f"""
You are an objective, careful evaluator for a binary question.

Question:
\"\"\"{question}\"\"\"

{ctx_block}Task:
1) Based on your knowledge and any provided context, estimate the probability (0-100) that the correct answer is YES.
2) Also provide the probability for NO.
3) Provide a concise (1-2 sentence) reasoning sentence.

Return ONLY a JSON object with keys:
{{"yes": NUMBER, "no": NUMBER, "reasoning": "short explanation"}}

If uncertain or lacking information, state uncertainty and be conservative.
"""
    return prompt.strip()

# -----------------------------
# Call local Ollama endpoint (streamed JSON)
# -----------------------------
def call_local_ollama(prompt: str, model: str = OLLAMA_MODEL, timeout: int = LLM_TIMEOUT_SECONDS) -> Tuple[float, float, str]:
    """
    Calls local Ollama generate endpoint and returns (yes_pct, no_pct, reasoning_text).
    Handles streamed JSON output.
    On any failure, returns neutral (50.0, 50.0, explanation).
    """
    url = f"{OLLAMA_HOST}/api/generate"
    debug(f"call_local_ollama: calling {url} with model={model}")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.0,
    }

    try:
        buffer = ""
        with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                # each line should be JSON chunk according to assumed Ollama streaming format
                try:
                    chunk = json.loads(line)
                except Exception:
                    # if it's not pure JSON, append raw text
                    try:
                        chunk_text = line.decode("utf-8", errors="ignore")
                        buffer += chunk_text
                    except Exception:
                        continue
                else:
                    # Collect both 'response' and 'thinking' if present
                    buffer += chunk.get("response", "") + chunk.get("thinking", "")
                    if chunk.get("done", False):
                        break

        # hide long model buffer in debug output for readability
        debug("call_local_ollama: model responded (details hidden).")

        # Try to parse strict JSON first
        try:
            parsed = json.loads(buffer)
            yes = float(parsed.get("yes", 50.0))
            no = float(parsed.get("no", 50.0))
            reason = parsed.get("reasoning", parsed.get("explanation", "")) or ""
            return yes, no, reason
        except Exception:
            # Heuristic extraction (robust fallback parsing)
            import re
            m_yes = re.search(
                r'["\']?yes["\']?\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)',
                buffer,
                flags=re.IGNORECASE,
            )
            m_no = re.search(
                r'["\']?no["\']?\s*[:\-]?\s*([0-9]{1,3}(?:\.[0-9]+)?)',
                buffer,
                flags=re.IGNORECASE,
            )
            if m_yes:
                try:
                    yes_v = float(m_yes.group(1))
                    no_v = float(m_no.group(1)) if m_no else max(0.0, 100.0 - yes_v)
                    reason = buffer.strip()
                    return yes_v, no_v, reason
                except Exception:
                    pass

            perc = re.findall(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%', buffer)
            if len(perc) >= 2:
                try:
                    yes_v = float(perc[0])
                    no_v = float(perc[1])
                    return yes_v, no_v, buffer.strip()
                except Exception:
                    pass

            debug("call_local_ollama: could not parse numeric probabilities; returning neutral 50/50")
            return 50.0, 50.0, buffer.strip()

    except Exception as e:
        debug(f"call_local_ollama: request failed: {e}")
        return 50.0, 50.0, f"Local Ollama call failed: {e}"

# -----------------------------
# Main evaluate flow
# -----------------------------
def evaluate_question_local(question: str, market_prob: Optional[float] = None, context: Optional[str] = None):
    """
    Calls the local LLM and returns a dictionary. Returned YES/NO are percentages (0..100).
    The structure matches previous behavior so caller code doesn't need changes.
    """
    debug("evaluate_question_local: start")
    debug(f"Ollama host: {OLLAMA_HOST}, model: {OLLAMA_MODEL}")

    prompt = build_prompt(question, context)
    # hide prompt contents for readability but keep debug trace
    debug("Prompt built (hidden for readability).")

    # Informational line (short)
    if market_prob is not None:
        try:
            mp_display = f"{market_prob:.4f}" if market_prob <= 1.01 else f"{float(market_prob):.2f}"
        except Exception:
            mp_display = str(market_prob)
        debug(f"Calling LLM with context: Market ML probability (0..1): {mp_display}")
    else:
        debug("Calling LLM with no market context")

    yes_raw, no_raw, reason = call_local_ollama(prompt, model=OLLAMA_MODEL)

    # Normalize probabilities (produce percentages 0..100)
    s = (yes_raw + no_raw)
    if s <= 0:
        debug("Raw probs sum <= 0; resetting to 50/50")
        yes_pct, no_pct = 50.0, 50.0
    else:
        yes_pct = (yes_raw / s) * 100.0
        no_pct = (no_raw / s) * 100.0

    debug(f"LLM-only normalized: YES={yes_pct:.2f}, NO={no_pct:.2f}")

    # Clean, readable LLM output (concise)
    print(f"[LLM] YES={yes_pct:.2f}% NO={no_pct:.2f}%")
    # Print up to 400 chars of reasoning so user still gets the explanation
    if reason:
        cleaned_reason = " ".join(reason.strip().split())
        print(f"[LLM Reasoning] {cleaned_reason[:400]}")
    else:
        print("[LLM Reasoning] (no reasoning returned)")

    # Combine with market probability if provided
    if market_prob is not None:
        if market_prob <= 1.01:
            market_yes_pct = market_prob * 100.0
        else:
            market_yes_pct = float(market_prob)
        debug(f"Combining with market probability {market_yes_pct:.2f}% (weights LLM={WEIGHT_LLM}, Market={WEIGHT_MARKET})")
        final_yes = WEIGHT_LLM * yes_pct + WEIGHT_MARKET * market_yes_pct
        final_no = 100.0 - final_yes
        final_yes = max(0.0, min(final_yes, 100.0))
        final_no = max(0.0, min(final_no, 100.0))

        side = "YES" if final_yes >= 50.0 else "NO"
        weight = final_yes if side == "YES" else final_no

        # concise final output
        print(f"[COMBINED] YES={final_yes:.2f}% NO={final_no:.2f}%")
        print(f"[FINAL] {side} ({weight:.2f}%)")
        return {"yes": final_yes, "no": final_no, "reasoning": reason, "side": side, "weight": weight}
    else:
        side = "YES" if yes_pct >= 50.0 else "NO"
        weight = yes_pct if side == "YES" else no_pct

        print(f"[FINAL LLM-ONLY] {side} ({weight:.2f}%)")
        return {"yes": yes_pct, "no": no_pct, "reasoning": reason, "side": side, "weight": weight}

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 llm_reasoner_local.py \"Your question here\" [market_yes_prob] [short_context]")
        sys.exit(1)

    question = sys.argv[1]
    market_prob = None
    context = None
    if len(sys.argv) >= 3:
        try:
            market_prob = float(sys.argv[2])
        except Exception:
            context = sys.argv[2]
    if len(sys.argv) >= 4:
        context = " ".join(sys.argv[3:])

    start = time.time()
    result = evaluate_question_local(question, market_prob=market_prob, context=context)
    debug(f"Total elapsed: {time.time() - start:.2f}s")
    print("\n=== RAW OUTPUT JSON ===")
    print(json.dumps(result, indent=2))
