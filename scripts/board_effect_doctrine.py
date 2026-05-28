#!/usr/bin/env python3
"""Deterministic Santa/Krampus board-effect doctrine.

Santa and Krampus are LUCIDOTA-local route personas/metaphors, not external
technical terms. They gate how much board position a move must touch and how
negative/audit claims are allowed to form.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "board_effect_doctrine"
OBS = ROOT / "04_RUNTIME" / "observation_center" / "board_effect_doctrine_latest.json"
BIG_BOARD = ROOT / "05_OUTPUTS" / "big_board.json"

EFFECT_PATTERNS: dict[str, tuple[str, ...]] = {
    "proof": ("proof", "verify", "verified", "evidence", "test"),
    "receipt": ("receipt", "ledger", "log", "hash", "05_outputs"),
    "graph": ("graph", "node", "edge", "promotion", "materialize", "stage"),
    "routing": ("route", "routing", "lane", "gate", "treelite", "river"),
    "model": ("model", "groq", "cohere", "local", "llm", "vram"),
    "krampuschewing": ("krampus", "krampuschewing", "corpse", "quarantine", "archive"),
    "corpus": ("corpus", "book", "chunk", "indy", "reads", "hunch"),
    "test": ("test", "pytest", "smoke", "regression"),
    "ontology": ("ontology", "working reality", "hypothesis", "truth", "abduction"),
    "status": ("status", "handoff", "watch", "observation", "metric", "board"),
    "db": ("postgres", "database", "db", "sql", "row", "table"),
    "latency": ("fast", "latency", "stream", "async", "queue", "bytewax"),
}

KRAMPUS_WORDS = {
    "audit",
    "slop",
    "naughty",
    "corpse",
    "quarantine",
    "kill",
    "dead",
    "krampus",
    "graph",
    "canonical",
    "promotion",
    "evidence",
    "contradiction",
}
SANTA_WORDS = {"glow", "nice", "santa", "assume", "find", "green", "help"}
NAUGHTY_WORDS = {"naughty", "slop", "corpse", "quarantine", "kill", "dead", "bad", "fail", "broken", "poison"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _norm_effect(effect: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", effect.strip().lower()).strip("_")


def extract_board_effects(text: str, explicit_effects: Iterable[str] | None = None) -> list[str]:
    """Return deterministic, deduped board-effect categories touched by a move."""
    low = text.lower()
    effects: set[str] = set()
    for effect, needles in EFFECT_PATTERNS.items():
        if any(needle in low for needle in needles):
            effects.add(effect)
    for effect in explicit_effects or []:
        norm = _norm_effect(str(effect))
        if norm:
            effects.add(norm)
    return sorted(effects)


def detect_naughty_claims(text: str) -> list[str]:
    low = text.lower()
    return sorted(word for word in NAUGHTY_WORDS if re.search(rf"\b{re.escape(word)}\b", low))


def select_board_persona(text: str) -> str:
    low = text.lower()
    if any(re.search(rf"\b{re.escape(word)}\b", low) for word in KRAMPUS_WORDS):
        return "krampus"
    if any(re.search(rf"\b{re.escape(word)}\b", low) for word in SANTA_WORDS):
        return "santa"
    return "santa"


def evaluate_board_effect(
    *,
    text: str,
    persona: str | None = None,
    evidence_refs: Iterable[str] | None = None,
    explicit_effects: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate whether a move satisfies the local Santa/Krampus board law."""
    persona = (persona or select_board_persona(text)).strip().lower()
    if persona not in {"krampus", "santa"}:
        persona = select_board_persona(text)
    effects = extract_board_effects(text, explicit_effects)
    evidence = [str(ref) for ref in (evidence_refs or []) if str(ref).strip()]
    naughty_claims = detect_naughty_claims(text)
    minimum = 2 if persona == "krampus" else 1
    blockers: list[str] = []
    if len(effects) < minimum:
        blockers.append("krampus_requires_at_least_two_board_effects" if persona == "krampus" else "santa_requires_at_least_one_board_effect")
    if persona == "krampus" and naughty_claims and not evidence:
        blockers.append("naughty_claim_without_evidence")
    preferred = len(effects) >= 4 or len(effects) == 2 if persona == "krampus" else len(effects) >= 1
    verdict = "PASS" if not blockers else "FAIL"
    return {
        "schema": "lucidota.board_effect_doctrine.v1",
        "persona": persona,
        "local_metaphor": True,
        "effect_count": len(effects),
        "effects": effects,
        "minimum_effect_count": minimum,
        "preferred_effect_shape": "2_or_4_plus" if persona == "krampus" else "1_plus",
        "preferred_shape_met": bool(preferred),
        "fairness": "evidence_bound" if persona == "krampus" else "glow_seeking",
        "assumption_policy": "must_not_invent_naughty" if persona == "krampus" else "may_seek_glow_but_must_label_assumptions",
        "naughty_claims": naughty_claims,
        "evidence_refs": evidence,
        "blockers": blockers,
        "verdict": verdict,
        "canonical_graph_writes_performed": False,
    }


def write_receipt(result: dict[str, Any], *, text: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lucidota.board_effect_doctrine.receipt.v1",
        "generated_at": now(),
        "text_hash": sha256_text(text),
        "result": result,
        "canonical_graph_writes_performed": False,
    }
    path = OUT / f"board_effect_doctrine_{stamp()}.json"
    payload["receipt_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    OBS.parent.mkdir(parents=True, exist_ok=True)
    OBS.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    update_big_board(payload)
    print("RECEIPT_PATH=" + rel(path))
    return path


def update_big_board(payload: dict[str, Any]) -> None:
    if BIG_BOARD.exists():
        try:
            board = json.loads(BIG_BOARD.read_text(encoding="utf-8"))
        except Exception:
            board = {}
    else:
        board = {}
    obs = board.setdefault("observation_center", {})
    obs["board_effect_doctrine"] = {
        "source_receipt": payload.get("receipt_path"),
        "verdict": payload["result"].get("verdict"),
        "persona": payload["result"].get("persona"),
        "effect_count": payload["result"].get("effect_count"),
        "canonical_graph_writes_performed": False,
    }
    counters = board.setdefault("counters", {})
    counters["board_effect_doctrine_last_effect_count"] = payload["result"].get("effect_count", 0)
    BIG_BOARD.parent.mkdir(parents=True, exist_ok=True)
    BIG_BOARD.write_text(json.dumps(board, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LUCIDOTA local Santa/Krampus board-effect doctrine.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--persona", choices=["krampus", "santa"])
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--effect", action="append", default=[])
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = evaluate_board_effect(text=args.text, persona=args.persona, evidence_refs=args.evidence_ref, explicit_effects=args.effect)
    if args.write_receipt:
        write_receipt(result, text=args.text)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    print("BOARD_EFFECT_DOCTRINE=" + result["verdict"])
    return 0 if result["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
