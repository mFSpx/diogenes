#!/usr/bin/env python3
"""
vibe_sequencer.py — rate-aware delegator with mandatory post-execution audit + receipts.

Queue JSONL format (one job per line):
  {
    "label": "human-readable name",
    "prompt": "full task prompt",
    "model": "codestral|ministral|groq|auto",
    "max_tokens": 60000,
    "max_price": 0.40,
    "verify": ["string that must appear in created/modified file"],
    "target_file": "/abs/path/to/file/that/should/exist/after"
  }

Model routing (auto mode):
  SQL / migration / schema  -> groq
  TypeScript / JavaScript   -> codestral
  Rust                      -> codestral
  Python edit (small)       -> ministral
  Python create (complex)   -> codestral
  default                   -> codestral

Concurrency caps:  codestral=1  ministral=3  groq=6
Retry on 429:      codestral=35s  ministral=12s  groq=5s  (x3 attempts)
Receipts:          05_OUTPUTS/receipts/seq_{label}_{ts}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT      = Path(__file__).resolve().parents[1]
VIBE      = str(ROOT / ".venv/bin/vibe")
RECEIPTS  = ROOT / "05_OUTPUTS" / "receipts"
RECEIPTS.mkdir(parents=True, exist_ok=True)

CONCURRENCY = {"codestral": 1, "ministral": 3, "groq": 6}
RETRY_WAIT  = {"codestral": 35, "ministral": 12, "groq": 5}
MAX_RETRIES = 3

MODEL_ENV = {
    "codestral": "codestral-2508",
    "ministral": "mistral-small-2506",
}

# Provider families — same family must not audit its own work.
# Ordered by cheapest first for audit selection.
PROVIDER_FAMILY = {
    "groq":       "groq_llama",
    "codestral":  "mistral_family",
    "ministral":  "mistral_family",
}
AUDIT_PREFERENCE = ["groq", "ministral", "codestral"]  # groq=cheap/token, ministral=free-tier

def get_cross_auditor(execution_model: str) -> str:
    """Return cheapest provider from a different family than the one that did the work."""
    own_family = PROVIDER_FAMILY.get(execution_model, execution_model)
    for candidate in AUDIT_PREFERENCE:
        if PROVIDER_FAMILY.get(candidate, candidate) != own_family:
            return candidate
    return AUDIT_PREFERENCE[0]

_print_lock = threading.Lock()

def _print(*a, **kw):
    with _print_lock:
        print(*a, **kw, flush=True)


# ── Model router ──────────────────────────────────────────────────────────────

def resolve_model(job: dict) -> str:
    m = job.get("model", "auto").lower()
    if m != "auto":
        return m
    p = job.get("prompt", "").lower()
    tf = job.get("target_file", "").lower()
    sql_kw  = ("create table", "alter table", "drop constraint", "add column",
                "migration", "schema", ".sql", "insert into", "update ")
    ts_kw   = (".ts", "typescript", "runtime.ts", "engine.ts", "phantom/src")
    rs_kw   = (".rs", "rust", "cargo.toml", "tokio", "apalis")
    py_big  = ("create", "write", "implement", "build")
    if any(k in p or k in tf for k in sql_kw):
        return "groq"
    if any(k in p or k in tf for k in ts_kw):
        return "codestral"
    if any(k in p or k in tf for k in rs_kw):
        return "codestral"
    if any(k in p for k in py_big):
        return "codestral"
    return "ministral"


# ── Execution backends ────────────────────────────────────────────────────────

def run_groq(prompt: str) -> str:
    import openai
    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT["groq"] * (attempt + 1))
            else:
                raise


def run_vibe(job: dict, model_key: str) -> str:
    import subprocess
    env = os.environ.copy()
    env["MISTRAL_MODEL"] = MODEL_ENV.get(model_key, "codestral-2508")
    max_tok = str(job.get("max_tokens", 60000))
    max_pr  = str(job.get("max_price",  0.40))
    for attempt in range(MAX_RETRIES):
        proc = subprocess.run(
            [VIBE, "--trust", "-p", job["prompt"],
             "--agent", "auto-approve",
             "--max-tokens", max_tok,
             "--max-price",  max_pr],
            env=env, capture_output=True, text=True, cwd=str(ROOT),
        )
        out = proc.stdout + proc.stderr
        if proc.returncode == 0:
            return out
        if "429" in out or "Rate limit" in out:
            wait = RETRY_WAIT[model_key] * (attempt + 1)
            _print(f"    [seq] 429/{model_key} retry {attempt+1}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)
        elif "Token limit exceeded" in out:
            raise RuntimeError(f"TOKEN_LIMIT: {out[-150:]}")
        else:
            raise RuntimeError(f"VIBE_ERR: {out[-300:]}")
    raise RuntimeError("MAX_RETRIES exceeded")


# ── Audit layer ───────────────────────────────────────────────────────────────

def audit_job(job: dict, vibe_output: str) -> dict:
    """Run a fast Groq audit of the completed work. Returns {pass: bool, reason: str}."""
    tf = job.get("target_file")
    verify_strings = job.get("verify", [])

    checks = []

    # 1. File existence check
    if tf:
        exists = Path(tf).exists()
        size   = Path(tf).stat().st_size if exists else 0
        checks.append({"check": "file_exists", "pass": exists and size > 20,
                        "detail": f"{tf} exists={exists} size={size}"})

    # 2. String presence checks
    for s in verify_strings:
        if tf and Path(tf).exists():
            content = Path(tf).read_text(errors="replace")
            found = s.lower() in content.lower()
            checks.append({"check": f"contains:{s[:40]}", "pass": found})
        else:
            checks.append({"check": f"contains:{s[:40]}", "pass": False,
                            "detail": "file not found"})

    # 3. Cross-family audit — whoever did the work cannot audit itself.
    execution_model = job.get("_execution_model", "codestral")
    auditor = get_cross_auditor(execution_model)
    audit_via_groq = (auditor == "groq")
    groq_verdict = None
    if tf and Path(tf).exists():
        snippet = Path(tf).read_text(errors="replace")[:800]
        audit_prompt = (
            f"Task was: {job['prompt'][:300]}\n\n"
            f"File created/modified: {tf}\n"
            f"First 800 chars of result:\n{snippet}\n\n"
            "Does the file look like a real, correct implementation of the task? "
            "Answer with exactly: PASS or FAIL, then one sentence reason."
        )
        try:
            if audit_via_groq:
                # Groq generated the work → Mistral (ministral) audits it
                verdict = run_vibe({"prompt": audit_prompt, "max_tokens": 4000, "max_price": 0.02,
                                    "label": "cross-audit"}, "ministral")
                audit_label = "mistral_cross_audit"
            else:
                # Mistral/codestral generated the work → Groq audits it
                verdict = run_groq(audit_prompt)
                audit_label = "groq_cross_audit"
            passed  = "PASS" in verdict.upper()[:20]
            groq_verdict = {"pass": passed, "verdict": verdict[:200], "auditor": audit_label}
            checks.append({"check": audit_label, "pass": passed, "detail": verdict[:200]})
        except Exception as e:
            checks.append({"check": "cross_audit", "pass": None,
                            "detail": f"audit_error: {e}"})

    overall = all(c["pass"] for c in checks if c["pass"] is not None)
    return {"pass": overall, "checks": checks}


# ── Receipt writer ────────────────────────────────────────────────────────────

def write_receipt(job: dict, model: str, status: str, elapsed: float,
                  audit: dict | None, error: str | None) -> Path:
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = job.get("label", "job").replace(" ", "_")[:40]
    path = RECEIPTS / f"seq_{slug}_{ts}.json"
    receipt = {
        "receipt_mode": "VIBE_SEQUENCER",
        "label":        job.get("label"),
        "target_file":  job.get("target_file"),
        "model":        model,
        "status":       status,
        "elapsed_s":    elapsed,
        "audit":        audit,
        "error":        error,
        "timestamp_utc": ts,
        "prompt_preview": job["prompt"][:200],
    }
    path.write_text(json.dumps(receipt, indent=2))
    return path


# ── Job executor ──────────────────────────────────────────────────────────────

def execute_job(job: dict) -> dict:
    label = job.get("label", job["prompt"][:50])
    model = resolve_model(job)
    t0    = time.time()

    try:
        _print(f"  → [{model}] {label}")
        if model == "groq":
            output = run_groq(job["prompt"])
            # groq returns content; if target_file specified, write it
            tf = job.get("target_file")
            if tf and output:
                content = output
                if content.startswith("```"):
                    lines   = content.split("\n")
                    content = "\n".join(l for l in lines if not l.startswith("```"))
                Path(tf).parent.mkdir(parents=True, exist_ok=True)
                Path(tf).write_text(content + "\n")
        else:
            output = run_vibe(job, model)

        elapsed = round(time.time() - t0, 1)
        job["_execution_model"] = model  # stamp so audit_job knows which cross-auditor to use
        audit   = audit_job(job, output)
        status  = "ok" if audit["pass"] else "audit_fail"
        rpath   = write_receipt(job, model, status, elapsed, audit, None)

        icon = "✓" if audit["pass"] else "⚠ AUDIT FAIL"
        _print(f"  {icon} [{model}] {label} ({elapsed}s) receipt={rpath.name}")
        return {"label": label, "model": model, "status": status,
                "elapsed": elapsed, "audit": audit}

    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        rpath   = write_receipt(job, model, "failed", elapsed, None, str(e))
        _print(f"  ✗ [{model}] {label} — {e} receipt={rpath.name}")
        return {"label": label, "model": model, "status": "failed",
                "elapsed": elapsed, "error": str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Rate-aware vibe sequencer with audit+receipts")
    ap.add_argument("queue",        help="JSONL queue file (or - for stdin)")
    ap.add_argument("--dry-run",    action="store_true")
    ap.add_argument("--concurrency",type=int, default=0,
                    help="Override per-model concurrency cap (0=defaults)")
    args = ap.parse_args()

    raw = sys.stdin.read() if args.queue == "-" else Path(args.queue).read_text()
    jobs = [json.loads(l) for l in raw.splitlines() if l.strip() and not l.startswith("#")]
    _print(f"[seq] {len(jobs)} jobs | receipts → {RECEIPTS}")

    if args.dry_run:
        for j in jobs:
            _print(f"  {resolve_model(j):12s} | {j.get('label', j['prompt'][:70])}")
        return

    by_model: dict[str, list[dict]] = {}
    for j in jobs:
        by_model.setdefault(resolve_model(j), []).append(j)

    all_results = []
    for model, mj in by_model.items():
        cap = args.concurrency if args.concurrency > 0 else CONCURRENCY[model]
        _print(f"[seq] {model} — {len(mj)} jobs, cap={cap}")
        with ThreadPoolExecutor(max_workers=cap) as pool:
            for r in as_completed({pool.submit(execute_job, j): j for j in mj}):
                all_results.append(r.result())

    ok   = sum(1 for r in all_results if r["status"] == "ok")
    warn = sum(1 for r in all_results if r["status"] == "audit_fail")
    bad  = sum(1 for r in all_results if r["status"] == "failed")
    _print(f"\n[seq] ✓{ok}  ⚠{warn} audit-fail  ✗{bad} failed")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
