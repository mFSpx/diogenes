#!/usr/bin/env python3
"""
GROQ GAUNTLET — Adversarial annealing pass over Darwin Hammer survivors.
Sends each gen1/2/3 hybrid to Groq 70B with a brutal adversarial prompt.
Scores 1-10. Survivors (>=6) saved to ALGOS/evolved/GAUNTLET_SURVIVORS/.
Fast: parallel API calls, no mercy.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVOLVED = ROOT / "ALGOS" / "evolved"
OUT = EVOLVED / "GAUNTLET_SURVIVORS"
LEADERBOARD = EVOLVED / "gauntlet_leaderboard.json"
OUT.mkdir(parents=True, exist_ok=True)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
PARALLEL = 20
PASS_THRESHOLD = 6
GENS = [1, 2, 3]

ADVERSARIAL_PROMPT = """\
You are a brutally honest computational mathematician and senior engineer.
You have zero patience for slop, fake math, or copy-paste jobs dressed up as research.

Read this hybrid algorithm code. Tear it apart. Then score it.

CODE:
{code}

EVALUATION CRITERIA (be savage on each):
1. MATHEMATICAL NOVELTY (1-10): Is there a genuine mathematical fusion, or did the LLM just paste two files together? Does a new equation or operator emerge?
2. IMPLEMENTATION CORRECTNESS (1-10): Does the code actually implement what the docstring claims? Are the formulas right?
3. FUSION DEPTH (1-10): Are the parent algorithms truly integrated at the mathematical level, or just coexisting in the same file?
4. UTILITY (1-10): Could this actually be useful for something? Or is it a mathematical curiosity with no application?

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no commentary:
{{"score": <integer 1-10, overall>, "novelty": <1-10>, "correctness": <1-10>, "fusion": <1-10>, "utility": <1-10>, "verdict": "<one brutal sentence>", "bright_spot": "<one thing that is actually interesting if anything, or null>", "is_survivor": <true if score>=6 else false>}}
"""


def load_key():
    k = os.environ.get("GROQ_API_KEY", "")
    if k:
        return k
    p = Path("/tmp/lucidota_groq_key")
    if p.exists():
        return p.read_text().strip()
    sys.exit("No Groq key")


def groq_call(key, code, timeout=60):
    prompt = ADVERSARIAL_PROMPT.format(code=code[:3000])
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "groq-python/0.28.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read().decode())["choices"][0]["message"]["content"]
            raw = raw.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
        return None
    except Exception:
        return None


def load_files():
    files = []
    for gen in GENS:
        d = EVOLVED / f"gen{gen}"
        if d.exists():
            for p in sorted(d.glob("*.py")):
                files.append((gen, p))
    return files


def process_one(key, gen, path):
    try:
        code = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    result = groq_call(key, code)
    if not result:
        return None

    result["file"] = str(path)
    result["gen"] = gen
    result["name"] = path.name
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat()

    if result.get("is_survivor") or result.get("score", 0) >= PASS_THRESHOLD:
        dst = OUT / f"gen{gen}_{path.name}"
        dst.write_text(code, encoding="utf-8")
        print(f"  [SURVIVOR gen{gen}] score={result.get('score')} {path.name[:50]}")
        print(f"    -> {result.get('verdict','')[:80]}")
        if result.get("bright_spot"):
            print(f"    BRIGHT: {result['bright_spot'][:80]}")
    else:
        print(f"  [dead gen{gen}] score={result.get('score')} {path.name[:40]} — {result.get('verdict','')[:60]}")

    return result


def main():
    key = load_key()
    files = load_files()
    print(f"\n=== GROQ GAUNTLET ===")
    print(f"Files to hammer: {len(files)} (gen1={sum(1 for g,_ in files if g==1)}, gen2={sum(1 for g,_ in files if g==2)}, gen3={sum(1 for g,_ in files if g==3)})")
    print(f"Model: {MODEL} | Parallel: {PARALLEL} | Pass threshold: {PASS_THRESHOLD}/10\n")

    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(process_one, key, gen, path): (gen, path) for gen, path in files}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)

    survivors = [r for r in results if r.get("score", 0) >= PASS_THRESHOLD]
    survivors.sort(key=lambda r: -r.get("score", 0))

    leaderboard = {
        "total_evaluated": len(results),
        "survivors": len(survivors),
        "survival_rate": f"{100*len(survivors)/max(len(results),1):.1f}%",
        "top_10": survivors[:10],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    LEADERBOARD.write_text(json.dumps(leaderboard, indent=2))

    print(f"\n=== GAUNTLET COMPLETE ===")
    print(f"Evaluated : {len(results)}")
    print(f"Survivors : {len(survivors)} ({leaderboard['survival_rate']})")
    print(f"Leaderboard: {LEADERBOARD}")
    print(f"\nTOP 10:")
    for r in survivors[:10]:
        print(f"  [{r['score']}/10] gen{r['gen']} {r['name'][:50]}")
        print(f"    {r.get('verdict','')[:90]}")
        if r.get("bright_spot"):
            print(f"    BRIGHT: {r['bright_spot'][:90]}")


if __name__ == "__main__":
    main()
