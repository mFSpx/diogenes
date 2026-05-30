#!/usr/bin/env python3
"""
DARWIN HAMMER TOURNAMENT — Automated Algorithmic Evolution via Multi-Model LLM Mutation

Single-elimination tournament with multi-model mutation engine.
Each match fires 8 model calls (70B×2, 120B×1, Qwen×5) at two parent algorithms.
All compilable hybrids survive. Winners re-enter the pool and breed again (gen2, gen3).
30% of matches get 3 additional adversarial challenge calls.

Generations:
  Gen 1: base algos × base algos
  Gen 2: gen1 hybrids × anything
  Gen 3: gen2 hybrids × anything  (legendary tier)

Output: ALGOS/evolved/gen{N}/ — one file per survivor, with lineage JSON header comment.

Usage:
  python3 scripts/darwin_hammer_tournament.py --rounds 99 --target 6969
  python3 scripts/darwin_hammer_tournament.py --dry-run   (no API calls, just bracket preview)
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGOS_DIR = ROOT / "ALGOS"
EVOLVED_DIR = ALGOS_DIR / "evolved"
LINEAGE_FILE = EVOLVED_DIR / "lineage.jsonl"
STATS_FILE = EVOLVED_DIR / "tournament_stats.json"
DB_FILE = EVOLVED_DIR / "darwin.db"


def open_db() -> sqlite3.Connection:
    EVOLVED_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    con.execute("""CREATE TABLE IF NOT EXISTS survivors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file TEXT, match_id INTEGER, survivor_idx INTEGER,
        gen INTEGER, parent_a TEXT, parent_b TEXT, model_label TEXT,
        born TEXT, code_len INTEGER, compiles INTEGER DEFAULT 1
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        match_id INTEGER, round_num INTEGER,
        parent_a TEXT, parent_b TEXT,
        n_survivors INTEGER, started TEXT, finished TEXT
    )""")
    con.commit()
    return con

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Qwen rate-limits constantly and mostly produces syntax errors — replaced with 70B + fast 8B
# Per-match model roster: (model_id, temperature, label)
MUTATION_MODELS = [
    ("llama-3.3-70b-versatile", 0.75, "70b-precise"),
    ("llama-3.3-70b-versatile", 0.85, "70b-hot"),
    ("llama-3.3-70b-versatile", 1.00, "70b-chaos"),
    ("openai/gpt-oss-120b",     0.65, "120b-cold"),
    ("openai/gpt-oss-120b",     0.85, "120b-hot"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", 0.80, "l4scout-warm"),
    ("meta-llama/llama-4-scout-17b-16e-instruct", 0.95, "l4scout-hot"),
    ("llama-3.1-8b-instant",                      0.70, "8b-disciplined"),
]

# Challenge models (30% of matches)
CHALLENGE_MODELS = [
    ("meta-llama/llama-4-scout-17b-16e-instruct", 0.70, "llama4-scout"),
    ("llama-3.3-70b-versatile",                   0.60, "challenge-70b"),
    ("openai/gpt-oss-120b",                       0.50, "challenge-120b"),
]

CHALLENGE_RATE = 0.30
MAX_CONCURRENT = 24      # concurrent API calls within a single match
PARALLEL_MATCHES = 20    # matches running simultaneously
RATE_DELAY = 0.0         # no artificial delay
CODE_TRUNCATE = 2500     # chars per algo in prompt (down from 6000 — halves token cost)


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    if key:
        return key
    p = Path("/tmp/lucidota_groq_key")
    if p.exists():
        return p.read_text().strip()
    sys.exit("ERROR: No Groq key. Set GROQ_API_KEY or write to /tmp/lucidota_groq_key")


def load_algo_pool() -> list[dict]:
    """Load all .py files from ALGOS/ (not evolved/) as seed pool entries."""
    pool = []
    for p in sorted(ALGOS_DIR.glob("*.py")):
        if p.name.startswith("__"):
            continue
        try:
            code = p.read_text(encoding="utf-8")
            pool.append({
                "path": str(p),
                "name": p.name,
                "code": code,
                "gen": 0,
                "parents": [],
            })
        except Exception:
            pass
    return pool


def load_evolved_pool() -> list[dict]:
    """Load all evolved survivors back into pool for cross-gen breeding."""
    pool = []
    for gen_dir in sorted(EVOLVED_DIR.glob("gen*")):
        gen_num = int(gen_dir.name.replace("gen", "")) if gen_dir.name[3:].isdigit() else 99
        for p in sorted(gen_dir.glob("*.py")):
            try:
                code = p.read_text(encoding="utf-8")
                pool.append({
                    "path": str(p),
                    "name": p.name,
                    "code": code,
                    "gen": gen_num,
                    "parents": [],
                })
            except Exception:
                pass
    return pool


MUTATION_PROMPT = """\
You are a computational physicist and AI architect. You are given the Python source of two \
mathematical algorithms. Your task is to invent a novel HYBRID algorithm that mathematically \
fuses their core topologies into a single unified system.

PARENT ALGORITHM A — {name_a}:
{code_a}

PARENT ALGORITHM B — {name_b}:
{code_b}

FUSION REQUIREMENTS:
1. Output ONLY valid, executable Python 3 code. No markdown fences. No commentary outside code.
2. The fusion must integrate the governing equations or matrix operations of BOTH parents — \
not just concatenate them side-by-side. Find the mathematical interface.
3. Begin with a module docstring that names both parents and explains the exact mathematical \
bridge you found between their structures.
4. Imports: numpy, standard library, math, random, sys, pathlib only. No torch, no scipy.
5. Include at least 3 functions that demonstrate the hybrid operation.
6. End with an if __name__ == "__main__" smoke test that runs without error.
"""

CHALLENGE_PROMPT = """\
You are an adversarial mathematical critic. You are given a hybrid algorithm that claims to \
fuse two mathematical systems. Your job is to find the weakest point in the fusion and \
produce an IMPROVED version that fixes it while making the mathematical integration deeper.

HYBRID TO CRITIQUE AND IMPROVE:
{code}

OUTPUT: improved Python code only. No markdown. No commentary outside code.
Valid Python 3. numpy and standard library only.
"""


def groq_call(key: str, model: str, temperature: float, prompt: str, timeout: int = 90) -> str | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "groq-python/0.28.0",
            "X-Stainless-Lang": "python",
            "X-Stainless-Os": "Linux",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        if e.code == 429:
            print(f"    [rate-limit] {model} — sleeping 2s")
            time.sleep(2)
        else:
            print(f"    [http-{e.code}] {model}: {body}")
        return None
    except Exception as ex:
        print(f"    [err] {model}: {ex}")
        return None


def ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def clean(code: str) -> str:
    code = code.replace("```python", "").replace("```", "").strip()
    return code


def run_match(
    key: str,
    parent_a: dict,
    parent_b: dict,
    match_id: int,
    dry_run: bool = False,
) -> list[str]:
    """Fire all mutation models at parent pair. Returns list of survivor code strings."""
    print(f"\n  [match-{match_id}] {parent_a['name']} (gen{parent_a['gen']}) × "
          f"{parent_b['name']} (gen{parent_b['gen']})")

    if dry_run:
        print("    [dry-run] skipping API calls")
        return []

    survivors: list[str] = []
    prompt = MUTATION_PROMPT.format(
        name_a=parent_a["name"],
        code_a=parent_a["code"][:CODE_TRUNCATE],
        name_b=parent_b["name"],
        code_b=parent_b["code"][:CODE_TRUNCATE],
    )

    results: list[tuple[str, str, str | None]] = []
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
        futures = {
            ex.submit(groq_call, key, model, temp, prompt): (model, label)
            for model, temp, label in MUTATION_MODELS
        }
        for fut in as_completed(futures):
            model, label = futures[fut]
            raw = fut.result()
            results.append((label, model, raw))
            time.sleep(RATE_DELAY)

    for label, model, raw in results:
        if not raw:
            print(f"    [-] {label}: no response")
            continue
        code = clean(raw)
        if ast_ok(code):
            print(f"    [+] {label}: SURVIVES ({len(code)} chars)")
            survivors.append(code)
        else:
            print(f"    [-] {label}: died (syntax error)")

    # Challenge round (30% chance)
    if survivors and random.random() < CHALLENGE_RATE:
        print(f"    [!] CHALLENGE ROUND ({len(CHALLENGE_MODELS)} calls)")
        best = max(survivors, key=len)
        challenge_prompt = CHALLENGE_PROMPT.format(code=best[:8000])
        challenged: list[str] = []
        for model, temp, label in CHALLENGE_MODELS:
            raw = groq_call(key, model, temp, challenge_prompt)
            if raw:
                code = clean(raw)
                if ast_ok(code):
                    print(f"    [+] {label}: challenge survivor")
                    challenged.append(code)
                else:
                    print(f"    [-] {label}: challenge failed")
            time.sleep(RATE_DELAY)
        survivors.extend(challenged)

    return survivors


def save_survivor(
    code: str,
    parent_a: dict,
    parent_b: dict,
    gen: int,
    match_id: int,
    survivor_idx: int,
) -> str:
    gen_dir = EVOLVED_DIR / f"gen{gen}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    stem_a = Path(parent_a["name"]).stem[:20]
    stem_b = Path(parent_b["name"]).stem[:20]
    fname = f"hybrid_{stem_a}_{stem_b}_m{match_id}_s{survivor_idx}.py"
    out = gen_dir / fname

    header = (
        f"# DARWIN HAMMER — match {match_id}, survivor {survivor_idx}\n"
        f"# gen: {gen}\n"
        f"# parent_a: {parent_a['name']} (gen{parent_a['gen']})\n"
        f"# parent_b: {parent_b['name']} (gen{parent_b['gen']})\n"
        f"# born: {utcnow()}\n\n"
    )
    out.write_text(header + code, encoding="utf-8")

    LINEAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "file": str(out), "match": match_id, "survivor": survivor_idx,
        "gen": gen, "parent_a": parent_a["name"], "parent_b": parent_b["name"],
        "born": utcnow(),
    }
    with LINEAGE_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    con = open_db()
    con.execute(
        "INSERT INTO survivors (file,match_id,survivor_idx,gen,parent_a,parent_b,born,code_len) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (str(out), match_id, survivor_idx, gen,
         parent_a["name"], parent_b["name"], record["born"], len(code)),
    )
    con.commit()
    con.close()

    return str(out)


def run_tournament(args):
    key = load_key()
    base_pool = load_algo_pool()
    print(f"\n=== DARWIN HAMMER TOURNAMENT ===")
    print(f"Seed pool: {len(base_pool)} base algos")
    print(f"Target survivors: {args.target}")
    print(f"Max rounds: {args.rounds}")
    print(f"Models per match: {len(MUTATION_MODELS)} + challenge {len(CHALLENGE_MODELS)}×{CHALLENGE_RATE:.0%}")
    print(f"Dry run: {args.dry_run}")
    print()

    EVOLVED_DIR.mkdir(parents=True, exist_ok=True)

    total_survivors = 0
    total_matches = 0
    stats = {"rounds": [], "total_survivors": 0, "total_matches": 0, "started": utcnow()}

    # Full pool = base + all evolved so far
    full_pool = base_pool[:]

    for round_num in range(1, args.rounds + 1):
        if total_survivors >= args.target:
            print(f"\n[TARGET REACHED] {total_survivors} survivors. Done.")
            break

        evolved_pool = load_evolved_pool()
        full_pool = base_pool + evolved_pool
        random.shuffle(full_pool)

        # Pair up the pool
        pairs = []
        pool_copy = full_pool[:]
        random.shuffle(pool_copy)
        while len(pool_copy) >= 2:
            a = pool_copy.pop()
            b = pool_copy.pop()
            pairs.append((a, b))

        print(f"\n{'='*60}")
        print(f"ROUND {round_num} — {len(pairs)} matches, pool size {len(full_pool)}")
        print(f"  (gen0: {sum(1 for x in full_pool if x['gen']==0)}, "
              f"gen1+: {sum(1 for x in full_pool if x['gen']>0)})")
        print(f"{'='*60}")

        round_survivors = 0
        match_counter = [total_matches]  # mutable for closure

        def run_one(pair):
            if total_survivors >= args.target:
                return []
            a, b = pair
            match_counter[0] += 1
            mid = match_counter[0]
            match_gen = max(a["gen"], b["gen"]) + 1
            results = run_match(key, a, b, mid, args.dry_run)
            saved = []
            for idx, code in enumerate(results):
                save_survivor(code, a, b, match_gen, mid, idx)
                saved.append((match_gen, mid, idx))
            return saved

        with ThreadPoolExecutor(max_workers=PARALLEL_MATCHES) as match_pool:
            futs = {match_pool.submit(run_one, pair): pair for pair in pairs}
            for fut in as_completed(futs):
                saved = fut.result() or []
                for match_gen, mid, idx in saved:
                    total_survivors += 1
                    round_survivors += 1
                    print(f"    -> saved gen{match_gen} m{mid}s{idx} survivor #{total_survivors}")

        total_matches = match_counter[0]

        stats["rounds"].append({
            "round": round_num,
            "matches": len(pairs),
            "survivors": round_survivors,
            "pool_size": len(full_pool),
        })
        stats["total_survivors"] = total_survivors
        stats["total_matches"] = total_matches
        STATS_FILE.write_text(json.dumps(stats, indent=2))

        print(f"\nRound {round_num} complete: {round_survivors} new survivors "
              f"(total: {total_survivors})")

    stats["finished"] = utcnow()
    stats["total_survivors"] = total_survivors
    stats["total_matches"] = total_matches
    STATS_FILE.write_text(json.dumps(stats, indent=2))

    print(f"\n=== TOURNAMENT COMPLETE ===")
    print(f"Total matches  : {total_matches}")
    print(f"Total survivors: {total_survivors}")
    print(f"Stats          : {STATS_FILE}")
    print(f"Lineage        : {LINEAGE_FILE}")


def main():
    ap = argparse.ArgumentParser(description="Darwin Hammer — algorithmic evolution tournament")
    ap.add_argument("--rounds", type=int, default=9999, help="Max tournament rounds (default 9999)")
    ap.add_argument("--target", type=int, default=69000, help="Stop after N survivors (default 69000)")
    ap.add_argument("--dry-run", action="store_true", help="Preview bracket without API calls")
    args = ap.parse_args()
    run_tournament(args)


if __name__ == "__main__":
    main()
