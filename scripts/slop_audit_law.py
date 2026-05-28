#!/usr/bin/env python3
"""Blueprint-First / PocketFlow simplicity audit.

Deterministic hygiene check: workflow path should live in code blueprints; models are
bounded tools. PocketFlow's ~100 line core is the local simplicity yardstick.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from spine_common import ROOT, now, receipt, rel

DEFAULT_POCKETFLOW = ROOT / "01_REPOS" / "PocketFlow" / "pocketflow" / "__init__.py"
OUT_ROOT = "05_OUTPUTS/slop_audit"
SCHEMA = "lucidota.slop_audit_law.v1"
ARXIV_REF = {
    "url": "https://arxiv.org/abs/2508.02721",
    "html_url": "https://arxiv.org/html/2508.02721",
    "title": "Blueprint First, Model Second: A Framework for Deterministic LLM Workflow",
    "submitted": "2025-08-01",
    "law_summary": [
        "workflow control path is an execution blueprint in source code",
        "deterministic engine executes the blueprint",
        "LLM calls are bounded tools, never the workflow path authority",
        "validation/double-check gates are code, not vibes",
        "telemetry/receipts are part of execution, not post-hoc prose",
    ],
}
TABOO_PATTERNS = [
    ("model_decides_workflow_path", re.compile(r"\b(llm|model|agent)\b.{0,60}\b(decides?|chooses?|selects?)\b.{0,60}\b(workflow|route|path|next step)", re.I | re.S)),
    ("prompt_only_control_flow", re.compile(r"\b(prompt|system prompt)\b.{0,80}\b(control|govern|enforce)\b.{0,80}\b(workflow|route|procedure)", re.I | re.S)),
]
NEGATING_RE = re.compile(r"\b(no|not|never|do not|don't|forbid|reject|prevent|bounded|deterministic)\b", re.I)


@dataclass(frozen=True)
class Span:
    kind: str
    name: str
    start: int
    end: int

    @property
    def loc(self) -> int:
        return max(0, self.end - self.start + 1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path.parent.parent), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return proc.stdout.strip() or None
    except Exception:
        return None


def reference_metrics(pocketflow_file: Path) -> dict[str, Any]:
    if not pocketflow_file.exists():
        return {
            "path": rel(pocketflow_file),
            "present": False,
            "lines": 100,
            "nonblank_lines": 88,
            "sha256": None,
            "commit": None,
            "fallback_used": True,
        }
    lines = pocketflow_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    return {
        "path": rel(pocketflow_file),
        "present": True,
        "lines": len(lines),
        "nonblank_lines": sum(1 for line in lines if line.strip()),
        "sha256": sha256(pocketflow_file),
        "commit": git_commit(pocketflow_file),
        "fallback_used": False,
    }


def iter_files(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            out.append(p)
            continue
        if p.is_file():
            out.append(p)
            continue
        for child in sorted(p.rglob("*")):
            if child.is_file() and child.suffix in {".py", ".rs", ".ts", ".js"}:
                if any(part in {".git", "node_modules", ".venv", "target", "__pycache__"} for part in child.parts):
                    continue
                out.append(child)
    return out


def python_spans(path: Path) -> list[Span]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    spans: list[Span] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and hasattr(node, "end_lineno"):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            spans.append(Span(kind, node.name, int(node.lineno), int(node.end_lineno or node.lineno)))
    return sorted(spans, key=lambda s: (s.start, s.end, s.name))


def rust_fn_spans(path: Path) -> list[Span]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    starts: list[tuple[str, int]] = []
    for idx, line in enumerate(lines, start=1):
        m = re.match(r"\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)\s*", line)
        if m:
            starts.append((m.group(1), idx))
    spans: list[Span] = []
    for name, start in starts:
        brace_depth = 0
        seen_open = False
        end = start
        for idx in range(start, len(lines) + 1):
            raw = lines[idx - 1]
            brace_depth += raw.count("{")
            if "{" in raw:
                seen_open = True
            brace_depth -= raw.count("}")
            if seen_open and brace_depth <= 0:
                end = idx
                break
        spans.append(Span("function", name, start, end))
    return spans


def spans_for(path: Path) -> list[Span]:
    if path.suffix == ".py":
        return python_spans(path)
    if path.suffix == ".rs":
        return rust_fn_spans(path)
    return []


def tier_for(ratio: float) -> str | None:
    if ratio >= 40:
        return "over_40x_pocketflow"
    if ratio >= 20:
        return "over_20x_pocketflow"
    if ratio >= 10:
        return "over_10x_pocketflow"
    if ratio >= 5:
        return "over_5x_pocketflow"
    return None


def taboo_hits(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    hits: list[dict[str, Any]] = []
    for kind, rx in TABOO_PATTERNS:
        for match in rx.finditer(text):
            snippet = " ".join(text[max(0, match.start() - 80): match.end() + 80].split())
            if "TABOO_PATTERNS" in snippet or kind in snippet:
                continue
            if NEGATING_RE.search(snippet):
                continue
            hits.append({"kind": kind, "path": rel(path), "snippet": snippet[:240]})
            break
    return hits


def audit(paths: list[str], *, pocketflow_file: Path, strict: bool) -> dict[str, Any]:
    ref = reference_metrics(pocketflow_file)
    baseline = max(1, int(ref.get("lines") or 100))
    warnings: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for path in iter_files(paths):
        if not path.exists():
            blockers.append({"kind": "path_missing", "path": rel(path)})
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            blockers.append({"kind": "path_unreadable", "path": rel(path), "error": str(exc)})
            continue
        lines = text.splitlines()
        file_ratio = len(lines) / baseline
        file_tier = tier_for(file_ratio)
        spans = spans_for(path)
        span_rows: list[dict[str, Any]] = []
        for span in spans:
            ratio = span.loc / baseline
            tier = tier_for(ratio)
            row = {"span_kind": span.kind, "name": span.name, "start": span.start, "end": span.end, "loc": span.loc, "ratio_to_pocketflow": round(ratio, 2)}
            span_rows.append(row)
            if tier:
                item = {"kind": "code_span_complexity_review", "tier": tier, "path": rel(path), **row}
                if strict or tier == "over_40x_pocketflow":
                    blockers.append(item)
                else:
                    warnings.append(item)
        if file_tier:
            item = {"kind": "file_size_complexity_review", "tier": file_tier, "path": rel(path), "loc": len(lines), "ratio_to_pocketflow": round(file_ratio, 2)}
            if strict or file_tier == "over_40x_pocketflow":
                blockers.append(item)
            else:
                warnings.append(item)
        for hit in taboo_hits(path):
            blockers.append(hit)
        files.append({
            "path": rel(path),
            "suffix": path.suffix,
            "loc": len(lines),
            "nonblank_lines": sum(1 for line in lines if line.strip()),
            "ratio_to_pocketflow": round(file_ratio, 2),
            "span_count": len(spans),
            "largest_spans": sorted(span_rows, key=lambda row: row["loc"], reverse=True)[:10],
        })
    verdict = "FAIL" if blockers else ("REVIEW" if warnings else "PASS")
    return {
        "schema": SCHEMA,
        "generated_at": now(),
        "verdict": verdict,
        "strict": strict,
        "blueprint_first_model_second": ARXIV_REF,
        "pocketflow_reference": ref,
        "local_law": {
            "workflow_path_authority": "source_code_blueprint",
            "model_role": "bounded_tool_only",
            "audit_question": "Are we doing PocketFlow-grade work, or are we inflating slop?",
            "review_thresholds": ["5x", "10x", "20x", "40x"],
            "direct_truth_promotion_performed": False,
        },
        "files_scanned": len(files),
        "files": files,
        "warnings": warnings,
        "blockers": blockers,
        "model_calls_performed": False,
        "network_calls_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit code against Blueprint-First / PocketFlow simplicity law.")
    ap.add_argument("--paths", nargs="+", default=["scripts"], help="Files/directories to audit; defaults to scripts/.")
    ap.add_argument("--pocketflow-file", default=str(DEFAULT_POCKETFLOW), help="PocketFlow reference __init__.py path.")
    ap.add_argument("--strict", action="store_true", help="Treat all >=5x PocketFlow complexity reviews as blockers.")
    ap.add_argument("--output", help="Optional receipt path.")
    ap.add_argument("--json", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit(args.paths, pocketflow_file=Path(args.pocketflow_file), strict=args.strict)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload["receipt_path"] = rel(out)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print("RECEIPT_PATH=" + rel(out))
    else:
        receipt("slop_audit_law", payload, root=OUT_ROOT)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("SLOP_AUDIT_LAW=" + payload["verdict"])
    return 0 if payload["verdict"] in {"PASS", "REVIEW"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
