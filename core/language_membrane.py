#!/usr/bin/env python3
"""The Language Membrane: deterministic intake + 4-lane output weaving."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from ALGOS.minhash import shingles, signature, similarity
from services.fairyfuse.fairyfuse_backend import route_command as fairyfuse_route_command
from pypeline.math.semantic_neighbors import register_document, semantic_neighbors

ROOT = Path(__file__).resolve().parents[2]
RETE_REGEXES = [
    re.compile(r"\bFOR UPDATE SKIP LOCKED\b", re.I),
    re.compile(r"\b(?:draft_only|FACT|PROBABLE|POSSIBLE|BULLSHIT|SURE_MAYBE)\b"),
    re.compile(r"\b(?:mouse_delta_sum|keystroke_burst|psyche_wrath_velocity|psyche_forensic_shield_ratio)\b", re.I),
]


@dataclass(frozen=True)
class MembraneRoute:
    lane: str
    reason: str
    matches: list[str]
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def route_inbound_text(text: str, *, enclave_id: str = "language_membrane", k: int = 5) -> dict[str, Any]:
    text = text or ""
    hits = [rx.pattern for rx in RETE_REGEXES if rx.search(text)]
    if hits:
        return MembraneRoute(
            lane="rete_regex",
            reason="fast structural match",
            matches=hits,
            payload={"text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest(), "length": len(text)},
        ).as_dict()

    register_document(enclave_id, f"query:{abs(hash(text))}", text)
    neighbors = semantic_neighbors(text, enclave_id=enclave_id, k=k, backend="memory")
    return MembraneRoute(
        lane="cpu_minilm_like",
        reason="ambiguous text -> CPU semantic neighbor search",
        matches=[],
        payload={"neighbors": [n.__dict__ for n in neighbors], "length": len(text), "backend": "memory_minilm_shim"},
    ).as_dict()


def rag_exact_quotes(query: str, vault_texts: dict[str, str], *, k: int = 3) -> list[dict[str, Any]]:
    qsig = signature(shingles(query, width=4), k=64)
    rows: list[tuple[float, str, str]] = []
    for doc_id, text in vault_texts.items():
        dsig = signature(shingles(text, width=4), k=64)
        score = similarity(qsig, dsig)
        rows.append((score, doc_id, text))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [{"doc_id": doc_id, "score": round(score, 4), "quote": text[:400]} for score, doc_id, text in rows[:k]]


def weave_output(
    *,
    deterministic_template: str,
    rag_quotes: list[dict[str, Any]],
    deepseek_synthesis: str,
    fairyfuse_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fairyfuse_context = fairyfuse_context or {}
    lanes = [
        {"lane": "tera_template", "text": deterministic_template},
        {"lane": "rag_quotes", "text": "\n".join(q.get("quote", "") for q in rag_quotes)},
        {"lane": "deepseek_q4", "text": deepseek_synthesis},
    ]
    smoothing = fairyfuse_route_command(
        deterministic_template[:512],
        "language_membrane_smoothing",
        {"rag_quotes": rag_quotes, "context": fairyfuse_context},
    ).to_dict()
    lanes.append({"lane": "fairyfuse_smoothing", "text": smoothing["ternary_vector"][:12]})
    return {
        "schema": "lucidota.language_membrane.weave_output.v1",
        "lanes": lanes,
        "smoothing": smoothing,
        "outbound_state": "draft_only",
    }
