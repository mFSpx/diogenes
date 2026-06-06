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

from ALGOS.runtime_caps import MAX_LABELS, MAX_SPANS, MAX_TEXT_CHARS, cap_text

INSTALL_COMMAND = "pip install gliner"
DEFAULT_LABELS = [
    "Operator", "Rainmaker", "Paladin / God-Mode", "Psyche / State-Collapse",
    "Forensic Shield", "Infinite Sink", "Anchor Weight", "Server Wipe",
    "API Rate Limiting", "Environment Migration", "Cruelty Protocols",
    "Master’s Eye", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS",
    "DIOGENES", "FairyFuse", "Job Fair Allocator", "Darwinian Surfaces",
    "Command Envelope Protocol",
]

# ---------------------------------------------------------------------------
# Code-specific entity labels for GLiNER code extraction
# ---------------------------------------------------------------------------
CODE_ENTITY_LABELS = [
    # Structure
    "CLASS_DEFINITION",
    "FUNCTION_DEFINITION",
    "IMPORT_STATEMENT",
    "DECORATOR",
    # Data layer
    "DB_TABLE_NAME",
    "DB_SCHEMA_NAME",
    "API_ENDPOINT_DEFINITION",
    "SQL_QUERY",
    # Config & env
    "CONFIG_KEY",
    "ENVIRONMENT_VARIABLE",
    # Domain
    "ONTOLOGY_TERM",
    "ERROR_EXCEPTION_TYPE",
    "PROTOCOL_OR_INTERFACE",
    "ALGORITHM_NAME",
    "SYSTEM_COMPONENT_NAME",
    "QUEUE_OR_WORKFLOW_NAME",
    "SCHEMA_OR_CONTRACT",
]

# Regex patterns for literal code-entity fallback detection.
# Each entry maps a label to a list of compiled patterns.
_CODE_ENTITY_PATTERNS: dict[str, list[re.Pattern]] = {}

def _compile_code_patterns() -> dict[str, list[re.Pattern]]:
    """Lazy-compile code entity regex patterns on first use."""
    if _CODE_ENTITY_PATTERNS:
        return _CODE_ENTITY_PATTERNS
    patterns: dict[str, list[re.Pattern]] = {
        "CLASS_DEFINITION": [
            re.compile(r"(?<!\w)class\s+([A-Za-z_]\w*)\s*(?:\(|:)", re.MULTILINE),
        ],
        "FUNCTION_DEFINITION": [
            re.compile(r"(?<!\w)(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE),
        ],
        "IMPORT_STATEMENT": [
            re.compile(r"^(?:from\s+([A-Za-z_.][\w.]*)\s+)?import\s+(.+)$", re.MULTILINE),
        ],
        "DECORATOR": [
            re.compile(r"^@([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", re.MULTILINE),
        ],
        "DB_TABLE_NAME": [
            re.compile(r"CREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:lucidota_\w+\.)?(\w+)", re.IGNORECASE | re.MULTILINE),
            re.compile(r'(?:FROM|INTO|TABLE|FROM\s+ONLY)\s+(?:lucidota_\w+\.)?(\w+)', re.IGNORECASE),
        ],
        "DB_SCHEMA_NAME": [
            re.compile(r"CREATE\s+SCHEMA\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.IGNORECASE | re.MULTILINE),
            re.compile(r'SET\s+search_path\s+TO\s+(\w+)', re.IGNORECASE),
        ],
        "API_ENDPOINT_DEFINITION": [
            re.compile(r"@\w+\.(?:get|post|put|patch|delete|options|head|route)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE | re.MULTILINE),
            re.compile(r"(?:\.add_route|\.add_api_route)\s*\(\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
        ],
        "ENVIRONMENT_VARIABLE": [
            re.compile(r"(?:os\.environ|os\.getenv|environ)\s*(?:\.\w+|[(\[])\s*['\"]?([A-Z_][A-Z0-9_]*)['\"]?"),
            re.compile(r'(?:os\.environ|os\.getenv|environ)\s*(?:\.\w+|[(\[])\s*["\']?([A-Z_][A-Z0-9_]*)["\']?'),
        ],
        "ONTOLOGY_TERM": [
            re.compile(r"\b(ENTITY|ATTRIBUTE|RELATIONSHIP|FRICTION|LEVERAGE|VISIBILITY|ACTION|EVENT|TIME|PATTERN|HYPOTHESIS|CLAIM|EVIDENCE|ATOMIC_ID|SIGNAL|GLOW|TERM|TOOL|ALGORITHM|NAUGHTY|NICE|GROUP|OPERATOR|MODE|COMMENT)\b"),
        ],
        "ERROR_EXCEPTION_TYPE": [
            re.compile(r"class\s+(\w+Error|Error\w+|Exception\w*|Fault\w*)\s*\(", re.MULTILINE),
            re.compile(r"class\s+(\w+)\s*\(.*(?:Exception|Error|BaseException)"),
        ],
        "PROTOCOL_OR_INTERFACE": [
            re.compile(r"class\s+(\w+)\s*\(.*Protocol\)"),
            re.compile(r"class\s+(\w+)\s*\(.*ABC\)"),
            re.compile(r"class\s+(\w+)\s*\(.*Interface\)"),
        ],
        "ALGORITHM_NAME": [
            re.compile(r'\b(?:class|def|async\s+def)\s+(\w+)(?:Bandit|Router|Ranker|Gate|Filter|Tree|Solver|Optimizer|Update|Delta|Coefficient|Entropy|Kernel|Hash|Dedupe|Partition|Motif|Attribution|Surrogate|Scheduler|Cipher|Schoolfield|Righting|Ambush|Leader|Election|Pruning|Avoidance|Sink|Rete|Estimate|Fold|Pheromone|HDC|NLP|LMS|SSIM|Minhash|Voronoi|RBF|KAN|LTC|VFE)\b', re.MULTILINE),
            re.compile(r"\b(?:Bandit|Rete|Hoeffding|GA|Tropical|Sheaf|Koopman|Caputo|Path\s*Signature|Diffusion\s*Forcing|Ternary|Pheromone|Infotaxis|Physarum|Capybara|Serpentina|Chelydrid|Poikilotherm|Doomsday|Fisher|Thanatosis|Possum|Honeybee|Dendritic|Rectified\s*Flow|Mistletoe|JEPA|Percyphon|Omni|Chaotic|Sprint|Liquid\s*Time|State\s*Space|Duality|Variational|Free\s*Energy|Entropic|Bayes|Damper|Tropical\s*Map|Belief|Rotor)\b"),
        ],
        "SYSTEM_COMPONENT_NAME": [
            re.compile(r"\b(?:KRAMPUSCHEWING|KORPUS|DIOGENES|FairyFuse|Rainmaker|Chrono[_-]?Ledger|Indy[_-]?[Rr]eads|Treelite|Marrow|Rive[rrs]|Absurd|Chrono|Catch[ _-]?Me|Master[ _-]?Eye|Cruelty[ _-]?Protocols|Darwinian[ _-]?[Ss]urfaces|Body[ _-]?Capture|Claw|IronClaw|Bonsai|Obelisk|Manticore|Hydra|Cerberus|Sphinx|Gryphon|Phoenix|Basilisk|Chimera|Pegasus|Centaur|Minotaur|Cyclops|Golem|LUCIDOTA|LLXPRT|ALPHASLOP|Project[ _-]?2501)\b"),
        ],
        "QUEUE_OR_WORKFLOW_NAME": [
            re.compile(r'QUEUE_NAME\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'WORKFLOW_NAME\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'JOB_KIND\s*=\s*["\']([^"\']+)["\']'),
            re.compile(r'WORKER_KEY\s*=\s*["\']([^"\']+)["\']'),
        ],
        "SCHEMA_OR_CONTRACT": [
            re.compile(r'"schema"\s*:\s*["\']([^"\']+)["\']'),
            re.compile(r'__all__\s*=\s*\[([^\]]+)\]'),
        ],
        "CONFIG_KEY": [
            re.compile(r'config\.([A-Za-z_]\w*)', re.IGNORECASE),
            re.compile(r'settings\.([A-Za-z_]\w*)', re.IGNORECASE),
            re.compile(r'cfg\[["\']([^"\']+)["\']\]'),
            re.compile(r'(?:get_config|get_setting)\s*\(\s*["\']([^"\']+)["\']'),
        ],
        "SQL_QUERY": [
            re.compile(r'(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|TRUNCATE)\s+[\s\S]*?(?:;|$)', re.IGNORECASE),
        ],
    }
    _CODE_ENTITY_PATTERNS.update(patterns)
    return _CODE_ENTITY_PATTERNS


def code_entity_fallback(text: str) -> list[Span]:
    """Regex-based code entity extraction for when GLiNER is unavailable.

    Uses compiled patterns to detect code constructs in source text and
    returns Span objects labeled by entity type.
    """
    patterns = _compile_code_patterns()
    spans: list[Span] = []
    seen: set[tuple[int, int, str]] = set()
    for label, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in pattern.finditer(text):
                # Determine the entity text:
                #   Try the first participating capture group; fall back to full match.
                #   (A group that didn't match returns None or has start=-1.)
                entity_text = None
                if match.lastindex and match.lastindex >= 1:
                    for gi in range(1, match.lastindex + 1):
                        g = match.group(gi)
                        if g is not None:
                            entity_text = g
                            break
                if entity_text is None or not entity_text.strip():
                    candidate = match.group(0)
                    if candidate and candidate.strip():
                        entity_text = candidate.strip()
                    else:
                        continue
                # For IMPORT_STATEMENT, reconstruct the full import line
                if label == "IMPORT_STATEMENT":
                    from_part = None
                    import_part = None
                    for gi in range(1, match.lastindex + 1):
                        g = match.group(gi)
                        if g is not None:
                            if from_part is None:
                                from_part = g
                            else:
                                import_part = g
                    if import_part:
                        entity_text = f"from {from_part} import {import_part}" if from_part else f"import {import_part}"
                start = match.start()
                end = match.end()
                # For patterns with a participating capture group, use its position
                for gi in range(1, (match.lastindex or 0) + 1):
                    if match.start(gi) >= 0:
                        start = match.start(gi)
                        end = match.end(gi)
                        break
                key = (start, end, label)
                if key not in seen:
                    seen.add(key)
                    spans.append(Span(
                        start=start,
                        end=end,
                        text=entity_text.strip(),
                        label=label,
                        score=1.0,
                        backend="code_entity_regex_fallback",
                    ))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label))


def code_entity_extract(text: str, labels: list[str] | None = None, *,
                        model: str | None = None,
                        threshold: float = 0.35,
                        allow_remote_model: bool = False,
                        no_fallback: bool = False) -> dict[str, Any]:
    """Extract code entities from source text.

    Tries GLiNER first (if model path is given and GLiNER is available),
    then falls back to regex-based code entity extraction.

    Returns the same schema as extract() but labeled with code entity types.
    """
    if labels is None:
        labels = list(CODE_ENTITY_LABELS)
    text, text_truncated = cap_text(text, limit=MAX_TEXT_CHARS)
    labels = [str(x) for x in labels if str(x).strip()][:MAX_LABELS]

    available, availability = gliner_available()
    backend: str
    backend_detail: dict[str, Any]
    spans: list[Span] = []

    label_set = frozenset(labels)
    if available and model:
        spans, backend_detail = run_gliner(text, labels, model, threshold, allow_remote_model=allow_remote_model)
        if spans or backend_detail.get("backend") == "gliner":
            backend = backend_detail.get("backend", "gliner")
        elif no_fallback:
            backend = backend_detail.get("backend", "gliner_error")
        else:
            spans = [s for s in code_entity_fallback(text) if s.label in label_set]
            backend = "code_entity_regex_after_gliner_unavailable"
    elif no_fallback:
        backend_detail = {"backend": "code_entity_gliner_missing_or_unspecified", "availability": availability, "install_command": INSTALL_COMMAND}
        backend = backend_detail["backend"]
    else:
        backend_detail = {"backend": "code_entity_regex_fallback", "availability": availability, "install_command": INSTALL_COMMAND}
        spans = [s for s in code_entity_fallback(text) if s.label in label_set]
        backend = "code_entity_regex_fallback"

    spans = spans[:MAX_SPANS]
    return {
        "schema": "lucidota.proof_hoard.gliner_code_entity_extractor.v1",
        "generated_at": now_iso(),
        "text_sha256": sha256_text(text),
        "text_length": len(text),
        "labels": labels[:MAX_LABELS],
        "backend": backend,
        "backend_detail": backend_detail,
        "install_instruction": INSTALL_COMMAND if not available else "gliner package importable",
        "text_truncated": text_truncated,
        "spans": [asdict(s) for s in spans],
        "span_count": len(spans),
    }


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
        return list(DEFAULT_LABELS)[:MAX_LABELS]
    p = Path(raw)
    try:
        is_file = p.exists() and p.is_file()
    except OSError:
        is_file = False
    if is_file:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            labels = data.get("required_exact_labels") or data.get("labels") or []
        else:
            labels = data
        return [str(x) for x in labels if str(x).strip()][:MAX_LABELS]
    return [part.strip() for part in raw.split(",") if part.strip()][:MAX_LABELS]


_MODEL_CACHE: dict[str, Any] = {}


def _get_cached_model(model_name: str, allow_remote_model: bool):
    cache_key = f"{model_name}|remote={bool(allow_remote_model)}"
    return _MODEL_CACHE.get(cache_key)


def _set_cached_model(model_name: str, allow_remote_model: bool, model: Any) -> Any:
    cache_key = f"{model_name}|remote={bool(allow_remote_model)}"
    _MODEL_CACHE[cache_key] = model
    return model


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
        model = _get_cached_model(model_name, allow_remote_model)
        if model is None:
            model = _set_cached_model(model_name, allow_remote_model, GLiNER.from_pretrained(model_name))
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
        if start < 0:
            start = 0
        if end < start:
            end = start
        matched = str(ent.get("text", text[start:end]))
        label = str(ent.get("label", ent.get("class", "")))
        score = float(ent.get("score", ent.get("confidence", 0.0)))
        if label:
            spans.append(Span(start, end, matched, label, score, "gliner"))
    return sorted(spans, key=lambda s: (s.start, s.end, s.label)), {"backend": "gliner", "model": model_name, "threshold": threshold}


def extract(text: str, labels: list[str], *, model: str | None = None, threshold: float = 0.35, allow_remote_model: bool = False, no_fallback: bool = False) -> dict[str, Any]:
    text, text_truncated = cap_text(text, limit=MAX_TEXT_CHARS)
    if not labels:
        labels = parse_labels(None)
    else:
        labels = [str(x) for x in labels if str(x).strip()][:MAX_LABELS]
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
    spans = spans[:MAX_SPANS]
    return {
        "schema": "lucidota.proof_hoard.gliner_zero_shot_extractor.v1",
        "generated_at": now_iso(),
        "text_sha256": sha256_text(text),
        "text_length": len(text),
        "labels": labels[:MAX_LABELS],
        "backend": backend,
        "backend_detail": backend_detail,
        "install_instruction": INSTALL_COMMAND if not available else "gliner package importable",
        "text_truncated": text_truncated,
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
