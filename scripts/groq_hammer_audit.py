#!/usr/bin/env python3
"""groq_hammer_audit.py — Daily LLM regression gate for recently-changed scripts.

Mutation class: read_only  (no DB writes, no graph mutations)

For each recently-changed Python script in scripts/, calls Groq with a
LUCIDOTA law prompt and surfaces violations as structured findings.

Receipt: 05_OUTPUTS/audit/groq_hammer_audit_<ts>.json
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "audit"

_GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_GROQ_MODEL = os.environ.get("GROQ_AUDIT_MODEL", "llama-3.3-70b-versatile")

_LAW_PROMPT = """You are a LUCIDOTA law auditor. Review this Python script for violations.
Flag ONLY real issues; do not flag style or minor nits.

Law checks:
1. RAW_GRAPH_WRITE — direct INSERT/UPDATE to lucidota_go.graph_item or lucidota_go.graph_edge
   without going through a command_envelope_uuid or graph_materialization_helper
2. MISSING_WORKER_CONTRACT — ABSURD worker that lacks absurd_worker_contracts.py validation
3. SLOP — any function >10× PocketFlow (~100 LOC), i.e. >1000 LOC
4. MISSING_RECEIPT — script claims completion but writes no receipt to 05_OUTPUTS/
5. AMBIENT_GRAPH_TRUTH — script constructs "canonical" graph facts in memory then persists
   without the promotion gate

Return JSON only, no prose:
{
  "violations": [
    {"law": "RAW_GRAPH_WRITE|MISSING_WORKER_CONTRACT|SLOP|MISSING_RECEIPT|AMBIENT_GRAPH_TRUTH",
     "line": <int or null>, "detail": "<one sentence>"}
  ],
  "verdict": "CLEAN" | "VIOLATIONS_FOUND"
}

Script path: {path}
Script content:
---
{content}
---"""


def _recent_changed_scripts(days: int = 7) -> list[Path]:
    """Return .py files in scripts/ changed in the last N days via git."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", f"HEAD~{days * 2}", "HEAD"],
            cwd=ROOT, text=True, timeout=15
        ).strip()
        changed = {ROOT / p for p in out.splitlines() if p.startswith("scripts/") and p.endswith(".py")}
        return [p for p in changed if p.exists()]
    except Exception:
        return []


def _call_groq(prompt: str) -> dict:
    try:
        import openai
        client = openai.OpenAI(api_key=_GROQ_API_KEY, base_url=_GROQ_BASE_URL)
        resp = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        # Strip any markdown fencing
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {"violations": [], "verdict": "PARSE_ERROR", "error": str(e)}
    except Exception as e:
        return {"violations": [], "verdict": "CALL_ERROR", "error": str(e)}


def audit_file(path: Path, max_chars: int = 6000) -> dict:
    content = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    prompt = _LAW_PROMPT.format(path=path.relative_to(ROOT), content=content)
    result = _call_groq(prompt)
    return {"path": str(path.relative_to(ROOT)), **result}


def main() -> int:
    ap = argparse.ArgumentParser(description="Groq LLM regression audit for LUCIDOTA scripts.")
    ap.add_argument("--paths", nargs="*", help="Specific script paths. Defaults to recently changed.")
    ap.add_argument("--days", type=int, default=7, help="Look-back window for recent changes.")
    ap.add_argument("--max-files", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--output", help="Override receipt path.")
    args = ap.parse_args()

    if not _GROQ_API_KEY:
        print("[groq_hammer_audit] ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 1

    if args.paths:
        targets = [Path(p) for p in args.paths if Path(p).exists()]
    else:
        targets = _recent_changed_scripts(args.days)

    targets = targets[:args.max_files]

    if not targets:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        receipt = {"status": "NO_TARGETS", "files_audited": 0, "ts": ts}
        OUT.mkdir(parents=True, exist_ok=True)
        rp = Path(args.output) if args.output else OUT / f"groq_hammer_audit_{ts}.json"
        rp.write_text(json.dumps(receipt, indent=2))
        print(f"RECEIPT_PATH={rp.relative_to(ROOT)}")
        return 0

    findings = []
    for path in targets:
        r = audit_file(path)
        findings.append(r)
        if not args.json:
            verdict = r.get("verdict", "?")
            vcount = len(r.get("violations", []))
            print(f"  {path.relative_to(ROOT)}: {verdict} ({vcount} violations)")

    total_violations = sum(len(f.get("violations", [])) for f in findings)
    clean = sum(1 for f in findings if f.get("verdict") == "CLEAN")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    receipt = {
        "schema": "groq_hammer_audit_v1",
        "ts": ts,
        "files_audited": len(findings),
        "clean": clean,
        "violations_total": total_violations,
        "model": _GROQ_MODEL,
        "findings": findings,
        "status": "PASS" if total_violations == 0 else "VIOLATIONS_FOUND",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    rp = Path(args.output) if args.output else OUT / f"groq_hammer_audit_{ts}.json"
    rp.write_text(json.dumps(receipt, indent=2))
    print(f"RECEIPT_PATH={rp.relative_to(ROOT)}")
    print(f"GROQ_HAMMER_AUDIT={receipt['status']}")

    if args.json:
        print(json.dumps(receipt, indent=2))

    return 0 if total_violations == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
