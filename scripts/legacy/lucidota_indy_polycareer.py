#!/usr/bin/env python3
"""INDY_READs polycareer workflow router + Glow Watch hook.

Deterministic/no-LLM helper for the Polycareer Workflow Wizard subproject.
It watches local CLAWD/workflow/agent artifacts for:
- the right boring workflow mode; and
- weird, high-value method patterns worth teaching Indy later.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import psycopg

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2] if THIS_FILE.parent.name == "legacy" else THIS_FILE.parents[1]
PROJECT = ROOT / "00_PROJECT_BRAIN" / "INDY_READS_POLYCAREER_WORKFLOW_WIZARD"
ROLE_MODES = PROJECT / "ROLE_MODES.json"
OUT_DIR = ROOT / "05_OUTPUTS" / "indy_polycareer"
FINDINGS_JSONL = OUT_DIR / "glow_watch_findings.jsonl"
FINDINGS_MD = OUT_DIR / "glow_watch_latest.md"
AGENT_DIR = ROOT / "01_REPOS" / "claudecode" / "rust" / ".claw-agents"
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"

ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "INTAKE_CLERK": ("drop", "incoming", "intake", "classify", "route", "manifest", "received", "file", "folder", "archive"),
    "EVIDENCE_VAULT": ("evidence", "hash", "sha", "cas", "custody", "source", "preserve", "archive", "receipt", "chain", "krampus", "korpus", "krampus express"),
    "OSINT_ANALYST": ("osint", "verify", "geolocate", "source", "pivot", "open source", "archive", "claim", "confidence"),
    "FRAUD_EXAMINER": ("fraud", "transaction", "incident", "finding", "interview", "loss", "scheme", "misrepresentation"),
    "LEGAL_CLERK": ("legal", "case", "court", "filing", "discovery", "exhibit", "authority", "deadline", "paralegal"),
    "NEWS_EDITOR": ("story", "publish", "editor", "headline", "source", "fact check", "right of reply", "correction"),
    "RESEARCH_LIBRARIAN": ("research", "literature", "screen", "include", "exclude", "bibliography", "systematic", "study"),
    "EXEC_ASSISTANT": ("calendar", "meeting", "follow up", "task", "brief", "schedule", "inbox", "reminder"),
    "MAILROOM_TECH": ("scan", "ocr", "index", "mail", "routing slip", "document", "metadata", "queue"),
    "MARKET_ANALYST": ("market", "competitor", "segment", "tam", "positioning", "customer", "industry", "pricing"),
    "SALES_STRATEGIST": ("sales", "buyer", "champion", "deal", "pipeline", "meddic", "pain", "proposal"),
    "ACTIVIST_ORGANIZER": ("power map", "campaign", "coalition", "target", "organize", "advocacy", "message", "pressure"),
    "RISK_ANALYST": ("risk", "threat", "vulnerability", "mitigation", "exposure", "trigger", "safety", "security"),
    "POET_EDITOR": ("voice", "poem", "narrative", "rhetoric", "style", "edit", "resonance", "beautiful"),
    "GLOW_HUNTER": ("glow", "anomaly", "weird", "unique", "invention", "better", "method", "pattern", "screech", "rage", "dope", "hypersystemic", "db is the os", "db as os", "rust core", "python wet clay", "go-25", "palantir on a shitty laptop", "low resources", "indy_reads", "indy reads", "krampus express", "percyhpnai", "percyphonai", "percyphon", "percy", "diogenes kernel", "needles", "6x needles", "mamba", "bonsai", "deepseek", "lora", "ternary"),
}

GLOW_POSITIVE = {
    "anomaly": 10, "anomalous": 10, "weird": 6, "unique": 7, "novel": 7, "unusual": 7,
    "invention": 10, "inventions": 10, "hypersystemic": 12,
    "better": 8, "faster": 5, "elegant": 6, "clever": 5, "brilliant": 8, "dope": 7,
    "method": 6, "workflow": 6, "protocol": 5, "pattern": 5, "playbook": 5, "system": 4, "sidekick": 5, "kernel": 7, "rust": 4, "operator": 4,
    "receipt": 7, "receipts": 7, "evidence": 5, "reproducible": 8, "repeatable": 8,
    "backlash": 7, "rage": 6, "screech": 7, "controversial": 5, "institution": 4,
    "secure": 4, "confidence": 3, "outperform": 10, "beats": 7, "different": 5, "ternary": 5,
}
GLOW_PHRASES = {
    "operator control": 10,
    "rust core": 10,
    "python wet clay": 10,
    "db is the os": 14,
    "db as os": 14,
    "go-25": 8,
    "palantir on a shitty laptop": 14,
    "low resources": 8,
    "krampus express": 12,
    "diogenes kernel": 12,
    "indy_reads": 10,
    "indy reads": 10,
    "six needles": 9,
    "6x needles": 9,
    "percyhpnai": 10,
    "percyphonai": 10,
    "percyphon": 10,
    "deepseek": 5,
    "mamba": 5,
    "bonsai": 5,
    "lora": 5,
}
GLOW_NEGATIVE = {"credential": -10, "missing": -6, "failed": -8, "error": -5, "traceback": -8, "exception": -6}
METHOD_CUES = ("because", "by ", "using", "instead", "unlike", "method", "workflow", "pattern", "protocol", "trick", "move", "habit", "wired", "operator control", "db is the os", "db as os", "sidekick")


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def load_modes() -> dict[str, Any]:
    return json.loads(ROLE_MODES.read_text(encoding="utf-8"))


def approx_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", text.lower())


def excerpt(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def stable_id(*parts: str) -> str:
    h = hashlib.sha256("\0".join(parts).encode("utf-8", errors="replace")).hexdigest()
    return h[:24]


def route_text(text: str, top_n: int = 5) -> list[dict[str, Any]]:
    low = text.lower()
    scores: list[dict[str, Any]] = []
    for mode, kws in ROLE_KEYWORDS.items():
        score = 0
        hits: list[str] = []
        for kw in kws:
            if kw in low:
                score += 3 + min(4, len(kw.split()))
                hits.append(kw)
        # broad evidence/workflow defaults
        if mode in {"INTAKE_CLERK", "EVIDENCE_VAULT"} and any(x in low for x in ("file", "source", "evidence", "drop")):
            score += 2
        if score:
            scores.append({"mode": mode, "score": score, "hits": hits[:8]})
    if not scores:
        scores.append({"mode": "INTAKE_CLERK", "score": 1, "hits": ["default"]})
    scores.sort(key=lambda r: (r["score"], r["mode"]), reverse=True)
    return scores[:top_n]


def glow_score(text: str) -> dict[str, Any]:
    words = approx_words(text)
    counts: dict[str, int] = {}
    score = 0
    for w in words:
        if w in GLOW_POSITIVE:
            counts[w] = counts.get(w, 0) + 1
            score += GLOW_POSITIVE[w]
        if w in GLOW_NEGATIVE:
            counts[w] = counts.get(w, 0) + 1
            score += GLOW_NEGATIVE[w]
    # receipt-backed weird method beats mere hype.
    low = text.lower()
    for phrase, weight in GLOW_PHRASES.items():
        if phrase in low:
            counts[phrase] = counts.get(phrase, 0) + 1
            score += weight
    has_method = any(cue in low for cue in METHOD_CUES)
    has_receipt = any(x in low for x in ("evidence", "receipt", "source", "hash", "archive", "proof", "example"))
    has_backlash = any(x in low for x in ("rage", "screech", "backlash", "controvers", "mad", "hated"))
    if has_method:
        score += 10
    if has_receipt:
        score += 10
    if has_backlash:
        score += 8
    score = max(0, min(100, score))
    return {
        "glow_score": score,
        "signals": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
        "has_method": has_method,
        "has_receipt": has_receipt,
        "has_backlash": has_backlash,
    }


def extract_moves(text: str, limit: int = 5) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    moves = []
    for s in sentences:
        low = s.lower()
        if any(cue in low for cue in METHOD_CUES) or any(x in low for x in ("better", "unique", "weird", "glow", "outperform")):
            moves.append(excerpt(s, 220))
        if len(moves) >= limit:
            break
    return moves


def product_template(mode: str) -> str:
    mode = mode.upper()
    templates = {
        "GLOW_HUNTER": "# Glow Profile\n\n## Subject\n## Domain\n## Why they glow\n## Core method\n## Evidence of quality\n## Backlash / screech map\n## Transferable moves\n## Risks / ethics\n## Indy adaptation\n## Score\n",
        "LEGAL_CLERK": "# Clerk Memo\n\n## Question presented\n## Working view\n## Facts from record\n## Issues\n## Evidence index\n## Deadlines\n## Questions for counsel/operator\n",
        "NEWS_EDITOR": "# Publication Review Memo\n\n## Story thesis\n## Verified facts\n## Unsupported claims\n## Source matrix\n## Right-of-reply targets\n## Risk flags\n## Publication recommendation\n",
        "OSINT_ANALYST": "# OSINT Brief\n\n## Requirement\n## Source set\n## Verified facts\n## Source reliability\n## Timeline\n## Contradictions\n## Gaps\n## Pivots\n",
    }
    return templates.get(mode, f"# {mode} Product\n\n## Mission\n## Source set\n## Findings\n## Confidence\n## Gaps\n## Risks\n## Next actions\n")


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        with conn.cursor() as cur:
            if CONTROL_SCHEMA.exists():
                cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
            if WORKFLOW_SCHEMA.exists():
                cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
            cur.execute(
                """
                INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                VALUES (%s,%s,%s,%s,'lucidota_indy_polycareer',%s::jsonb)
                RETURNING event_id::text
                """,
                (workflow_id, run_id, phase, status, jdump(detail)),
            )
            event_id = cur.fetchone()[0]
        conn.commit()
    return str(event_id)


def recent_workflow_items(since: dt.datetime, limit: int) -> list[dict[str, Any]]:
    try:
        with psycopg.connect(STATE_DSN, connect_timeout=3) as conn:
            rows = conn.execute(
                """
                SELECT event_id::text, workflow_id, phase, status, source, detail::text, created_at::text
                FROM lucidota_control.workflow_event
                WHERE created_at >= %s
                  AND workflow_id <> 'indy-polycareer-glow-watch'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (since, limit),
            ).fetchall()
    except Exception:
        return []
    return [
        {
            "kind": "workflow_event",
            "id": r[0],
            "title": f"{r[1]}:{r[2]}:{r[3]}",
            "path": "lucidota_control.workflow_event",
            "created_at": r[6],
            "text": f"workflow_id={r[1]} phase={r[2]} status={r[3]} source={r[4]} detail={r[5]}",
        }
        for r in rows
    ]


def agent_items(agent_dir: Path, since: dt.datetime, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not agent_dir.exists():
        return out
    cutoff = since.timestamp()
    files = sorted(agent_dir.glob("agent-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for jf in files[: max(limit * 2, limit)]:
        try:
            if jf.stat().st_mtime < cutoff:
                continue
            data = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
            md_path = Path(data.get("outputFile") or jf.with_suffix(".md"))
            md = md_path.read_text(encoding="utf-8", errors="replace") if md_path.exists() else ""
            text = jdump(data) + "\n" + md
            out.append({
                "kind": "claw_agent",
                "id": str(data.get("agentId") or jf.stem),
                "title": str(data.get("name") or jf.stem),
                "path": str(jf),
                "created_at": str(data.get("createdAt") or int(jf.stat().st_mtime)),
                "text": text,
            })
        except Exception as exc:
            out.append({"kind": "claw_agent_error", "id": jf.stem, "title": jf.name, "path": str(jf), "created_at": "", "text": f"agent artifact read error: {exc}"})
        if len(out) >= limit:
            break
    return out


def make_finding(item: dict[str, Any]) -> dict[str, Any]:
    text = item.get("text") or ""
    g = glow_score(text)
    routes = route_text(text)
    primary = routes[0]["mode"] if routes else "INTAKE_CLERK"
    return {
        "finding_id": stable_id(item.get("kind", ""), item.get("id", ""), item.get("text", "")[:500]),
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_kind": item.get("kind"),
        "source_id": item.get("id"),
        "source_title": item.get("title"),
        "source_path": item.get("path"),
        "source_created_at": item.get("created_at"),
        "primary_mode": primary,
        "routes": routes,
        **g,
        "transferable_moves": extract_moves(text),
        "excerpt": excerpt(text, 700),
    }


def append_findings(findings: list[dict[str, Any]]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if FINDINGS_JSONL.exists():
        with FINDINGS_JSONL.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("finding_id"):
                    existing.add(str(obj["finding_id"]))
    new_findings = [f for f in findings if str(f.get("finding_id")) not in existing]
    if new_findings:
        with FINDINGS_JSONL.open("a", encoding="utf-8") as fh:
            for f in new_findings:
                fh.write(jdump(f) + "\n")
    lines = ["# INDY_READs Glow Watch Latest", "", f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}", ""]
    if not new_findings:
        lines.append("No new glow candidates above threshold.")
    for f in new_findings[:20]:
        lines += [
            f"## {f['source_kind']} · {f['source_title']}",
            "",
            f"- glow_score: {f['glow_score']}",
            f"- primary_mode: {f['primary_mode']}",
            f"- source: `{f['source_path']}`",
            f"- signals: {f['signals']}",
            "- transferable_moves:",
        ]
        lines += [f"  - {m}" for m in f.get("transferable_moves", [])] or ["  - none extracted"]
        lines += ["", "> " + f.get("excerpt", "")[:500], ""]
    FINDINGS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(new_findings)


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    count = 0
    if FINDINGS_JSONL.exists():
        count = sum(1 for _ in FINDINGS_JSONL.open("r", encoding="utf-8", errors="replace"))
    modes = load_modes().get("modes", [])
    return {
        "ok": True,
        "project": str(PROJECT.relative_to(ROOT)),
        "role_modes": len(modes),
        "findings_jsonl": str(FINDINGS_JSONL.relative_to(ROOT)),
        "findings_count": count,
        "agent_dir": str(AGENT_DIR.relative_to(ROOT)),
        "workflow": "indy-polycareer-glow-watch",
    }


def cmd_modes(args: argparse.Namespace) -> dict[str, Any]:
    data = load_modes()
    return {"ok": True, "modes": data.get("modes", []), "glow_score_dimensions": data.get("glow_score_dimensions", [])}


def cmd_route(args: argparse.Namespace) -> dict[str, Any]:
    text = args.message or " ".join(args.rest).strip()
    if not text and args.file:
        text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not text:
        raise SystemExit("route needs --message, --file, or trailing text")
    routes = route_text(text, top_n=args.top)
    g = glow_score(text)
    return {"ok": True, "routes": routes, "glow": g, "transferable_moves": extract_moves(text), "excerpt": excerpt(text)}


def cmd_glow_score(args: argparse.Namespace) -> dict[str, Any]:
    text = args.text or ""
    if args.notes:
        text += "\n" + Path(args.notes).read_text(encoding="utf-8", errors="replace")
    if not text:
        raise SystemExit("glow-score needs --text or --notes")
    finding = make_finding({"kind": "manual", "id": args.subject or stable_id(text), "title": args.subject or "manual", "path": args.notes or "manual", "created_at": "", "text": text})
    if args.write:
        written = append_findings([finding])
        event_id = emit_event("indy-polycareer-glow-watch", finding["finding_id"], "manual_glow_score", "succeeded", {"finding": finding})
        finding["workflow_event"] = event_id
        finding["written"] = written
    return {"ok": True, "finding": finding}


def cmd_product_template(args: argparse.Namespace) -> dict[str, Any]:
    return {"ok": True, "mode": args.mode.upper(), "template": product_template(args.mode)}


def cmd_watch_once(args: argparse.Namespace) -> dict[str, Any]:
    since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=args.since_hours)
    items: list[dict[str, Any]] = []
    if args.workflow_events:
        items.extend(recent_workflow_items(since, args.limit))
    if args.agent_artifacts:
        items.extend(agent_items(Path(args.agent_dir), since, args.limit))
    # Also allow explicit watch paths for the operator's "little agent" scratch pads.
    for watch_path in args.path:
        p = Path(watch_path).expanduser()
        if p.is_file():
            try:
                items.append({"kind": "watch_file", "id": stable_id(str(p), str(p.stat().st_mtime_ns)), "title": p.name, "path": str(p), "created_at": dt.datetime.fromtimestamp(p.stat().st_mtime, tz=dt.timezone.utc).isoformat(), "text": p.read_text(encoding="utf-8", errors="replace")})
            except Exception:
                pass
    findings = [make_finding(i) for i in items]
    findings.sort(key=lambda f: (f["glow_score"], f["routes"][0]["score"] if f.get("routes") else 0), reverse=True)
    kept = [f for f in findings if f["glow_score"] >= args.threshold][: args.limit]
    written = append_findings(kept)
    event_id = emit_event(
        "indy-polycareer-glow-watch",
        f"watch-once-{int(time.time())}",
        "watch_once",
        "succeeded",
        {
            "since": since.isoformat(),
            "items_seen": len(items),
            "candidates_seen": len(kept),
            "candidates_written": written,
            "threshold": args.threshold,
            "findings_jsonl": str(FINDINGS_JSONL.relative_to(ROOT)),
            "latest_md": str(FINDINGS_MD.relative_to(ROOT)),
            "top": kept[:5],
        },
    )
    return {"ok": True, "event_id": event_id, "items_seen": len(items), "candidates_seen": len(kept), "candidates_written": written, "findings": kept[: args.limit], "latest_md": str(FINDINGS_MD.relative_to(ROOT))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lucidota-indy-polycareer")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("modes").set_defaults(func=cmd_modes)

    p = sub.add_parser("route")
    p.add_argument("--message", default="")
    p.add_argument("--file", default="")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("rest", nargs="*")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("glow-score")
    p.add_argument("--subject", default="")
    p.add_argument("--text", default="")
    p.add_argument("--notes", default="")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_glow_score)

    p = sub.add_parser("product-template")
    p.add_argument("--mode", required=True)
    p.set_defaults(func=cmd_product_template)

    p = sub.add_parser("watch-once")
    p.add_argument("--since-hours", type=float, default=24.0)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--threshold", type=int, default=35)
    p.add_argument("--agent-artifacts", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workflow-events", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--agent-dir", default=str(AGENT_DIR))
    p.add_argument("--path", action="append", default=[])
    p.set_defaults(func=cmd_watch_once)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
