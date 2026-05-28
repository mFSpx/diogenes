#!/usr/bin/env python3
"""Sovereign GLiNER zero-shot extraction instrument for the proof hoard.

Purpose:
- Accept raw text plus target labels.
- Return exact character-offset spans: start, end, text, label, score.
- Stay decoupled from production ABSURD/runtime imports.

Dependency:
- Optional: pip install gliner
- Model loading is explicit. No remote model is downloaded unless the operator passes
  --allow-remote-model with a non-local --model name.

Fallback:
- If GLiNER/model is unavailable, this tool can perform literal label matching so
  offset plumbing remains testable. The backend is then clearly marked
  literal_fallback_no_gliner, not GLiNER.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INSTALL_COMMAND = "pip install gliner"
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def parse_labels(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_LABELS)
    p = Path(raw)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x) for x in labels if str(x).strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def gliner_available() -> tuple[bool, str]:
    try:
        import gliner  # noqa: F401
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def model_is_local(model: str | None) -> bool:
    return bool(model) and Path(str(model)).exists()


def literal_fallback(text: str, labels: list[str], *, case_sensitive: bool = False) -> list[Span]:
    flags = 0 if case_sensitive else re.IGNORECASE
    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for label in labels:
        candidates = {label}
        candidates.add(label.replace(" / ", " "))
        candidates.add(label.replace("-", " "))
        for phrase in sorted(candidates, key=len, reverse=True):
            if not phrase.strip():
                continue
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", flags)
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(Span(match.start(), match.end(), match.group(0), label, 1.0, "literal_fallback_no_gliner"))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))


def run_gliner(text: str, labels: list[str], model_name: str, threshold: float, *, allow_remote_model: bool) -> tuple[list[Span], dict[str, Any]]:
    if not model_is_local(model_name) and not allow_remote_model:
        return [], {
            "backend": "gliner_not_loaded_remote_model_blocked",
            "reason": "model is not a local path and --allow-remote-model was not set",
            "install_command": INSTALL_COMMAND,
        }
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained(model_name)
        entities = model.predict_entities(text, labels, threshold=threshold)
    except Exception as exc:
        return [], {
            "backend": "gliner_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "install_command": INSTALL_COMMAND,
        }
    spans: list[Span] = []
    for ent in entities:
        start = int(ent.get("start", ent.get("start_pos", 0)))
        end = int(ent.get("end", ent.get("end_pos", start)))
        matched = str(ent.get("text", text[start:end]))
        label = str(ent.get("label", ent.get("class", "")))
        score = float(ent.get("score", ent.get("confidence", 0.0)))
        if 0 <= start <= end <= len(text) and label:
            spans.append(Span(start, end, matched, label, score, "gliner"))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label)), {"backend": "gliner", "model": model_name, "threshold": threshold}


def extract(text: str, labels: list[str], *, model: str | None = None, threshold: float = 0.35, allow_remote_model: bool = False, no_fallback: bool = False) -> dict[str, Any]:
    available, availability = gliner_available()
    backend_detail: dict[str, Any]
    spans: list[Span] = []
    if available and model:
        spans, backend_detail = run_gliner(text, labels, model, threshold, allow_remote_model=allow_remote_model)
        if spans or backend_detail.get("backend") == "gliner":
            backend = backend_detail.get("backend", "gliner")
        elif no_fallback:
            backend = backend_detail.get("backend", "gliner_error")
        else:
            fallback = literal_fallback(text, labels)
            spans = fallback
            backend = "literal_fallback_after_gliner_unavailable"
    elif no_fallback:
        backend_detail = {"backend": "gliner_missing_or_model_unspecified", "availability": availability, "install_command": INSTALL_COMMAND}
        backend = backend_detail["backend"]
    else:
        backend_detail = {"backend": "literal_fallback_no_gliner", "availability": availability, "install_command": INSTALL_COMMAND}
        spans = literal_fallback(text, labels)
        backend = "literal_fallback_no_gliner"
    return {
        "schema": "lucidota.proof_hoard.gliner_zero_shot_extractor.v1",
        "generated_at": now_iso(),
        "text_sha256": sha256_text(text),
        "text_length": len(text),
        "labels": labels,
        "backend": backend,
        "backend_detail": backend_detail,
        "install_instruction": INSTALL_COMMAND if not available else "gliner package importable",
        "spans": [asdict(s) for s in spans],
        "span_count": len(spans),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Proof-hoard GLiNER zero-shot extraction instrument")
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--labels", help="Comma-separated labels or path to JSON fixture. Defaults to Operator ontology labels.")
    ap.add_argument("--model", default=os.environ.get("GLINER_MODEL_PATH"), help="Local GLiNER model path/name. Remote names require --allow-remote-model.")
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--allow-remote-model", action="store_true", help="Allow GLiNER.from_pretrained to resolve a non-local model name; may download externally.")
    ap.add_argument("--no-fallback", action="store_true", help="Do not use literal offset fallback when GLiNER/model is unavailable.")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()
    text = load_text(args)
    labels = parse_labels(args.labels)
    result = extract(text, labels, model=args.model, threshold=args.threshold, allow_remote_model=args.allow_remote_model, no_fallback=args.no_fallback)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=not args.pretty, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
