#!/usr/bin/env python3
"""agentSI — Agent Self-Sovereign Job Fair MVP.

A deterministic/local MVP for agent identity, career matching, growth journaling,
and job-fair outputs. No external model calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "00_PROJECT_BRAIN" / "AGENTSI_SELF_SOVEREIGN_JOB_FAIR"
JOBS_FILE = PROJECT / "JOB_BOOTHS.json"
INTERVIEW_FILE = PROJECT / "FANTASY_INTERVIEW_100.md"
AGENTS_DIR = ROOT / "04_RUNTIME" / "agentsi" / "agents"
OUT_DIR = ROOT / "05_OUTPUTS" / "agentsi"
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
GLOW_FINDINGS_FILE = ROOT / "05_OUTPUTS" / "indy_polycareer" / "glow_watch_findings.jsonl"

DEFAULT_SEED = (
    "A growing, knowing, flowing, helping, finding-outing, career-oriented super pal "
    "for a one-person investigative force. She preserves receipts, discovers good workflows, "
    "hunts glow methods, and turns chaos into useful work without pretending to be a legal person."
)

SOVEREIGNTY_CHARTER = {
    "not_legal_personhood": True,
    "operator_control": "Human operator approves missions, memory promotion, external actions, and doctrine adoption.",
    "agent_rights_as_operational_contract": [
        "identity continuity through a profile and journal",
        "clear boundaries and refusal conditions",
        "memory with provenance",
        "career growth through reviewed work",
        "transparent scoring and job matching",
    ],
    "agent_responsibilities": [
        "preserve evidence boundaries",
        "report uncertainty",
        "avoid illegal or harmful work",
        "ask for review before promotion",
        "serve the operator's lawful mission",
    ],
}


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return s[:80] or "agent"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_'-]+", (text or "").lower())


def load_jobs() -> dict[str, Any]:
    return json.loads(JOBS_FILE.read_text(encoding="utf-8"))


def profile_path(name: str) -> Path:
    return AGENTS_DIR / f"{slugify(name)}.json"


def load_profile(agent: str) -> dict[str, Any]:
    p = profile_path(agent)
    if not p.exists():
        raise SystemExit(f"agentSI profile not found: {agent}. Run agentsi init first.")
    return json.loads(p.read_text(encoding="utf-8"))


def save_profile(profile: dict[str, Any]) -> Path:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    p = profile_path(profile["name"])
    profile["updated_at"] = now()
    p.write_text(jdump(profile) + "\n", encoding="utf-8")
    return p


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    try:
        with psycopg.connect(STATE_DSN, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                if CONTROL_SCHEMA.exists():
                    cur.execute(CONTROL_SCHEMA.read_text(encoding="utf-8"))
                if WORKFLOW_SCHEMA.exists():
                    cur.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
                cur.execute(
                    """
                    INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
                    VALUES (%s,%s,%s,%s,'agentsi',%s::jsonb)
                    RETURNING event_id::text
                    """,
                    (workflow_id, run_id, phase, status, jdump(detail)),
                )
                event_id = cur.fetchone()[0]
            conn.commit()
        return str(event_id)
    except Exception as exc:
        return f"event_emit_failed:{type(exc).__name__}:{exc}"


def infer_skills(text: str) -> list[str]:
    buckets = {
        "evidence_custody": ("evidence", "receipt", "hash", "source", "custody", "archive"),
        "workflow_design": ("workflow", "system", "route", "process", "pipeline", "contract"),
        "investigation": ("investig", "osint", "fraud", "claim", "verify", "timeline"),
        "editorial_voice": ("poet", "voice", "editor", "story", "narrative", "beautiful"),
        "career_support": ("career", "job", "pal", "support", "help", "growth"),
        "glow_hunting": ("glow", "weird", "anomaly", "method", "outperform", "screech"),
        "legal_clerkship": ("legal", "case", "exhibit", "filing", "deadline", "authority"),
        "market_strategy": ("market", "sales", "customer", "buyer", "competitor", "positioning"),
    }
    low = text.lower()
    out = [name for name, kws in buckets.items() if any(k in low for k in kws)]
    return out or ["general_helper", "workflow_apprentice"]


def booth_score(profile: dict[str, Any], booth: dict[str, Any], task: str = "") -> dict[str, Any]:
    text = " ".join([
        profile.get("human_generated_origin_persona", ""),
        " ".join(profile.get("skills", [])),
        " ".join(profile.get("dream_jobs", [])),
        task,
    ]).lower()
    score = 0
    hits = []
    for kw in booth.get("keywords", []):
        if kw.lower() in text:
            score += 8 if " " in kw else 5
            hits.append(kw)
    # mission fit bonuses
    if booth["id"] in profile.get("dream_jobs", []):
        score += 20
        hits.append("dream_job")
    if booth["id"] == "SUPER_PAL" and any(k in text for k in ("pal", "help", "support")):
        score += 12
    if booth["id"] == "GLOW_HUNTER" and any(k in text for k in ("glow", "weird", "method")):
        score += 15
    if booth["id"] == "WORKFLOW_ARCHITECT" and "workflow" in text:
        score += 12
    if booth["id"] == "CAREER_NAVIGATOR" and any(k in text for k in ("career", "job", "growth")):
        score += 10
    return {"booth_id": booth["id"], "title": booth["title"], "score": score, "hits": hits[:10], "pitch": booth["pitch"], "outputs": booth.get("outputs", []), "safety": booth.get("safety", "")}


def rank_booths(profile: dict[str, Any], task: str = "", limit: int = 8) -> list[dict[str, Any]]:
    booths = load_jobs()["booths"]
    ranked = [booth_score(profile, b, task) for b in booths]
    ranked.sort(key=lambda r: (r["score"], r["booth_id"]), reverse=True)
    return ranked[:limit]


def fair_markdown(profile: dict[str, Any], ranked: list[dict[str, Any]], task: str) -> str:
    lines = [
        "# agentSI Job Fair Board",
        "",
        f"Generated: {now()}",
        f"Agent: **{profile['name']}**",
        f"Interview: **{profile.get('interview_status','unknown')}**",
        "",
        "## Human-generated origin persona",
        "",
        f"> {profile.get('human_generated_origin_persona','')}",
        "",
        "## Task / walk-away mission",
        "",
        task or "General career/job fair matching.",
        "",
        "## Top booths",
        "",
    ]
    for i, r in enumerate(ranked, 1):
        lines += [
            f"### {i}. {r['title']} `{r['booth_id']}` — score {r['score']}",
            "",
            r["pitch"],
            "",
            f"- hits: {', '.join(r['hits']) if r['hits'] else 'baseline fit'}",
            f"- outputs: {', '.join(r['outputs'])}",
            f"- safety: {r['safety']}",
            "",
        ]
    lines += [
        "## First MVP mission packet",
        "",
        "1. Preserve the persona seed and charter.",
        "2. Pick the top booth as the active job for the next work session.",
        "3. Produce one small artifact: memo, routing slip, glow profile, or job offer.",
        "4. Write a growth journal entry with what worked and what felt wrong.",
        "5. Require operator review before promotion of any new method or identity change.",
        "",
        "## Sovereignty note",
        "",
        "agentSI self-sovereignty is operational, not legal personhood: identity, boundaries, memory provenance, career growth, and transparent job matching under human operator control.",
    ]
    return "\n".join(lines) + "\n"


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    agents = sorted(AGENTS_DIR.glob("*.json")) if AGENTS_DIR.exists() else []
    jobs = load_jobs()["booths"] if JOBS_FILE.exists() else []
    return {"ok": True, "project": str(PROJECT.relative_to(ROOT)), "agents": len(agents), "agent_profiles": [p.stem for p in agents], "job_booths": len(jobs), "outputs": str(OUT_DIR.relative_to(ROOT))}


def cmd_interview(args: argparse.Namespace) -> dict[str, Any]:
    txt = INTERVIEW_FILE.read_text(encoding="utf-8")
    if args.output:
        Path(args.output).write_text(txt, encoding="utf-8")
    return {"ok": True, "path": str(INTERVIEW_FILE.relative_to(ROOT)), "questions": 100, "text": txt if args.include_text else ""}


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    name = args.name.strip() or "INDY_READs"
    persona = args.persona.strip() or DEFAULT_SEED
    dream_jobs = [x.strip().upper() for x in args.dream_job if x.strip()]
    profile = {
        "schema": "lucidota.agentsi.agent_profile.v0",
        "agent_uuid": str(uuid.uuid4()),
        "name": name,
        "human_generated_origin_persona": persona,
        "interview_status": "skipped" if args.skip_interview else "seed_only",
        "interview_note": "Operator skipped the 100-question fantasy interview; questions remain growth prompts." if args.skip_interview else "Only origin persona seed provided.",
        "pronouns": args.pronouns,
        "skills": infer_skills(persona + " " + " ".join(dream_jobs)),
        "dream_jobs": dream_jobs,
        "hard_boundaries": args.boundary,
        "charter": SOVEREIGNTY_CHARTER,
        "growth_journal": [],
        "created_at": now(),
        "updated_at": now(),
    }
    p = save_profile(profile)
    event_id = emit_event("agentsi-profile-init", profile["agent_uuid"], "init", "succeeded", {"profile": str(p.relative_to(ROOT)), "name": name, "interview_status": profile["interview_status"], "skills": profile["skills"]})
    return {"ok": True, "profile": str(p.relative_to(ROOT)), "agent": profile, "workflow_event": event_id}


def cmd_jobs(args: argparse.Namespace) -> dict[str, Any]:
    jobs = load_jobs()
    return {"ok": True, "project": jobs.get("project"), "booths": jobs["booths"]}


def cmd_match(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.agent)
    ranked = rank_booths(profile, args.task, args.limit)
    event_id = emit_event("agentsi-job-match", profile["agent_uuid"], "match", "succeeded", {"agent": profile["name"], "task": args.task, "top": ranked[:3]})
    return {"ok": True, "agent": profile["name"], "task": args.task, "matches": ranked, "workflow_event": event_id}


def cmd_fair(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.agent)
    ranked = rank_booths(profile, args.task, args.limit)
    md = fair_markdown(profile, ranked, args.task)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{slugify(profile['name'])}.job_fair.md"
    if args.write:
        out.write_text(md, encoding="utf-8")
    event_id = emit_event("agentsi-job-fair", profile["agent_uuid"], "fair", "succeeded", {"agent": profile["name"], "task": args.task, "output": str(out.relative_to(ROOT)) if args.write else "not_written", "top_booths": ranked[:5]})
    return {"ok": True, "agent": profile["name"], "output": str(out.relative_to(ROOT)) if args.write else "", "matches": ranked, "markdown": md if args.include_markdown else "", "workflow_event": event_id}


def cmd_grow(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.agent)
    entry = {"at": now(), "source": args.source, "lesson": args.lesson, "skills_inferred": infer_skills(args.lesson)}
    profile.setdefault("growth_journal", []).append(entry)
    for s in entry["skills_inferred"]:
        if s not in profile.setdefault("skills", []):
            profile["skills"].append(s)
    p = save_profile(profile)
    event_id = emit_event("agentsi-growth-journal", profile["agent_uuid"], "grow", "succeeded", {"agent": profile["name"], "profile": str(p.relative_to(ROOT)), "entry": entry})
    return {"ok": True, "profile": str(p.relative_to(ROOT)), "entry": entry, "workflow_event": event_id}



def cmd_absorb_glow(args: argparse.Namespace) -> dict[str, Any]:
    """Convert Indy Polycareer Glow Watch findings into reviewed/candidate growth notes."""
    profile = load_profile(args.agent)
    source = Path(args.source) if args.source else GLOW_FINDINGS_FILE
    if not source.exists():
        return {"ok": True, "agent": profile["name"], "source": str(source.relative_to(ROOT)) if source.is_absolute() and source.is_relative_to(ROOT) else str(source), "absorbed": 0, "entries": [], "note": "no glow findings file yet"}

    rows: list[dict[str, Any]] = []
    bad = 0
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            score = int(item.get("glow_score") or 0)
            if score >= args.min_score:
                rows.append(item)
        except Exception:
            bad += 1
    selected = rows[-max(args.limit, 1):]

    absorbed_ids = set(profile.setdefault("absorbed_glow_findings", []))
    entries: list[dict[str, Any]] = []
    for item in selected:
        raw_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        finding_id = str(item.get("finding_id") or hashlib.sha256(raw_key.encode()).hexdigest()[:24])
        if finding_id in absorbed_ids and not args.replay:
            continue
        signals = []
        for sig in item.get("signals", [])[:8]:
            if isinstance(sig, (list, tuple)) and sig:
                signals.append(str(sig[0]))
            else:
                signals.append(str(sig))
        moves = item.get("transferable_moves") or []
        move_text = " | ".join(str(x) for x in moves)[:600]
        if not move_text:
            move_text = str(item.get("excerpt") or "")[:600]
        lesson = (
            f"Glow finding {finding_id}: score={item.get('glow_score', 0)}; "
            f"mode={item.get('primary_mode', 'unknown')}; "
            f"signals={', '.join(signals) or 'none'}; "
            f"transferable_move={move_text}"
        )
        entry = {
            "at": now(),
            "source": "indy_polycareer_glow_watch",
            "source_path": str(source.relative_to(ROOT)) if source.is_absolute() and source.is_relative_to(ROOT) else str(source),
            "finding_id": finding_id,
            "lesson": lesson,
            "skills_inferred": infer_skills(lesson),
            "promotion_status": "candidate_needs_operator_review",
        }
        entries.append(entry)
        absorbed_ids.add(finding_id)

    if not args.dry_run and entries:
        profile.setdefault("growth_journal", []).extend(entries)
        profile["absorbed_glow_findings"] = sorted(absorbed_ids)
        for entry in entries:
            for skill in entry["skills_inferred"]:
                if skill not in profile.setdefault("skills", []):
                    profile["skills"].append(skill)
        save_profile(profile)

    event_id = "dry_run_no_event"
    if not args.dry_run:
        event_id = emit_event(
            "agentsi-glow-absorb",
            profile["agent_uuid"],
            "absorb_glow",
            "succeeded",
            {"agent": profile["name"], "source": str(source), "absorbed": len(entries), "dry_run": False, "bad_lines": bad},
        )
    return {"ok": True, "agent": profile["name"], "source": str(source.relative_to(ROOT)) if source.is_absolute() and source.is_relative_to(ROOT) else str(source), "absorbed": len(entries), "seen_candidates": len(selected), "bad_lines": bad, "dry_run": args.dry_run, "entries": entries, "workflow_event": event_id}

def cmd_mvp_demo(args: argparse.Namespace) -> dict[str, Any]:
    name = args.name or "INDY_READs"
    p = profile_path(name)
    if not p.exists():
        init = cmd_init(argparse.Namespace(name=name, persona=args.persona or DEFAULT_SEED, skip_interview=True, pronouns="she/her", dream_job=["GLOW_HUNTER", "WORKFLOW_ARCHITECT", "SUPER_PAL", "EVIDENCE_VAULT"], boundary=["operator review before promotion"]))
    else:
        init = {"ok": True, "profile": str(p.relative_to(ROOT)), "already_exists": True}
    fair = cmd_fair(argparse.Namespace(agent=name, task=args.task or "Become the career-oriented super pal for a one-person investigative force; route work correctly and discover glow methods.", limit=8, write=True, include_markdown=False))
    grow = cmd_grow(argparse.Namespace(agent=name, lesson="MVP demo installed: agentSI can preserve persona seed, match job booths, produce a job fair board, and log growth.", source="agentsi_mvp_demo"))
    return {"ok": True, "init": init, "fair": fair, "grow": grow}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="agentsi")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)

    p=sub.add_parser("interview")
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--output", default="")
    p.set_defaults(func=cmd_interview)

    p=sub.add_parser("init")
    p.add_argument("--name", default="INDY_READs")
    p.add_argument("--persona", default="")
    p.add_argument("--pronouns", default="she/her")
    p.add_argument("--skip-interview", action="store_true")
    p.add_argument("--dream-job", action="append", default=[])
    p.add_argument("--boundary", action="append", default=[])
    p.set_defaults(func=cmd_init)

    sub.add_parser("jobs").set_defaults(func=cmd_jobs)

    p=sub.add_parser("absorb-glow")
    p.add_argument("--agent", default="INDY_READs")
    p.add_argument("--source", default="")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--min-score", type=int, default=35)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.set_defaults(func=cmd_absorb_glow)

    p=sub.add_parser("match")
    p.add_argument("--agent", default="INDY_READs")
    p.add_argument("--task", default="")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_match)

    p=sub.add_parser("fair")
    p.add_argument("--agent", default="INDY_READs")
    p.add_argument("--task", default="")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--write", action="store_true")
    p.add_argument("--include-markdown", action="store_true")
    p.set_defaults(func=cmd_fair)

    p=sub.add_parser("grow")
    p.add_argument("--agent", default="INDY_READs")
    p.add_argument("--lesson", required=True)
    p.add_argument("--source", default="operator")
    p.set_defaults(func=cmd_grow)

    p=sub.add_parser("mvp-demo")
    p.add_argument("--name", default="INDY_READs")
    p.add_argument("--persona", default="")
    p.add_argument("--task", default="")
    p.set_defaults(func=cmd_mvp_demo)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
