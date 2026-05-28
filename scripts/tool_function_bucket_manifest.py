#!/usr/bin/env python3
"""Build a repeatable, reusable function-level bucket manifest for tools and scripts.

What this does:
- Reads the canonical TICKLETRUNK manifest for non-destructive tool inventory.
- Enriches with SCRIPT_AUDIT_MANIFEST function/role metadata when available.
- Buckets each tool by inferred FUNCTION (domain+purpose), not by one-off guesswork.
- Adds repeatability annotations so one-time-like tooling can be identified for reuse planning.
- Writes versioned JSON + Markdown summaries under 05_OUTPUTS/tool_function_buckets.

No files are moved, copied, deleted, or modified.
"""
from __future__ import annotations

import argparse
import json
import csv
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKLETRUNK = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
DEFAULT_SCRIPT_AUDIT = ROOT / "scripts" / "SCRIPT_AUDIT_MANIFEST.jsonl"
OUT_ROOT = ROOT / "05_OUTPUTS" / "tool_function_buckets"


FUNCTION_BUCKETS = [
    "DATA_INGEST",
    "DOCUMENT_PARSE",
    "GRAPH_WORKFLOW",
    "QUEUE_OR_WORKER",
    "GOVERNANCE_AUDIT",
    "WORKFLOW_ORCHESTRATION",
    "MODEL_RUNTIME",
    "SCHEMA_KNOWLEDGE",
    "STORAGE_INTERFACE",
    "INTERFACE_CLI",
    "REPORTING_INSIGHT",
    "SCRAPING_CAPTURE",
    "SERVICE_INFRA",
    "REPO_TOOLING",
    "REPO_ASSET",
    "OTHER",
]

REPORT_BUCKET_ORDER = [
    "WORKFLOW_ORCHESTRATION",
    "QUEUE_OR_WORKER",
    "SCHEMA_KNOWLEDGE",
    "GRAPH_WORKFLOW",
    "MODEL_RUNTIME",
    "REPO_TOOLING",
    "REPO_ASSET",
    "OTHER",
    "GOVERNANCE_AUDIT",
    "STORAGE_INTERFACE",
    "DATA_INGEST",
    "DOCUMENT_PARSE",
    "INTERFACE_CLI",
    "REPORTING_INSIGHT",
    "SERVICE_INFRA",
    "SCRAPING_CAPTURE",
]


ROLE_BUCKET_HINTS: dict[str, str] = {
    "legacy_script_corpse": "QUEUE_OR_WORKER",
    "absurd_queue_worker_or_helper": "QUEUE_OR_WORKER",
    "graph_promotion_or_materialization_helper": "GRAPH_WORKFLOW",
    "chrono_ledger_helper": "GRAPH_WORKFLOW",
    "korpus_krampus_ingestion_helper": "DATA_INGEST",
    "document_ingestion_helper": "DOCUMENT_PARSE",
    "model_runtime_helper": "MODEL_RUNTIME",
    "ahoy_strategy_dataset_builder_cli": "WORKFLOW_ORCHESTRATION",
    "ahoy_strategy_model_explanation_cli": "MODEL_RUNTIME",
    "ahoy_million_replay_orchestrator": "WORKFLOW_ORCHESTRATION",
    "queue_worker_auditor_component": "GOVERNANCE_AUDIT",
    "generic_queue_worker_executor": "QUEUE_OR_WORKER",
    "safe_ops_runtime_contract": "SERVICE_INFRA",
    "active_compatibility_entrypoint": "INTERFACE_CLI",
    "test_checker_auditor": "GOVERNANCE_AUDIT",
    "case_hash_hypertimeline_or_glow_runtime": "GRAPH_WORKFLOW",
    "pruned_optional_diagnostic_or_batch_tool": "REPORTING_INSIGHT",
    "sovereign_toolbox_prior": "REPO_TOOLING",
    "script_helper": "WORKFLOW_ORCHESTRATION",
}


PATTERN_BUCKETS: list[tuple[str, str]] = [
    ("DOCUMENT_PARSE", r"\b(parse|parser|ocr|quote|markdown_ingest|chunk|chunking|text_extract|pdf|extract)\b"),
    ("DATA_INGEST", r"\b(intake|ingest|capture|harvest|body_capture|browser_capture|gather|source_import|drive_import)\b"),
    ("SCRAPING_CAPTURE", r"\b(scrape|crawler|crawl|browser_capture|capture|url|http|website)\b"),
    ("QUEUE_OR_WORKER", r"\b(worker|daemon|runner|consume|retry|watch|watcher|spine|queue|service_loop|loop)\b"),
    ("GRAPH_WORKFLOW", r"\b(graph|timeline|chrono|claim|edge|promotion|river|meta|ontology|temporal|evidence|case|ledger)\b"),
    ("GOVERNANCE_AUDIT", r"\b(gate|audit|quarantine|validator|validation|contract|check|verify|snapshot|fault|security|hard\s*truth)\b"),
    ("WORKFLOW_ORCHESTRATION", r"\b(workflow|orchestr|pipeline|synthesis|dispatch|router|scheduler|scheduler|deck|govern|scheduler|policy|rule|state_machine)\b"),
    ("MODEL_RUNTIME", r"\b(model|llm|embed|inference|rerun|predict|runtime|xgboost|treelite|shap|reranker|bert|router)\b"),
    ("REPORTING_INSIGHT", r"\b(report|summary|dashboard|status|audit_report|fact|metric|inventory|manifest|ledger_report)\b"),
    ("INTERFACE_CLI", r"\b(cli|surface|command|api|endpoint|cmd|shell|export|import|bridge)\b"),
    ("STORAGE_INTERFACE", r"\b(store|storage|cache|runtime_asset|artifact|bundle|archive|drive|vault|file|dataset|snapshot)\b"),
    ("REPO_ASSET", r"\b(repo_tool|repo|repo_toolbox|repository|llama\.cpp|cybercrafter|lumen|doggystyle|sharksnail|pytorch|prismml|lucidota_etl|ncnn)\b"),
]


FUNCTION_NAME_BUCKET_HINTS: list[tuple[str, str]] = [
    ("DOCUMENT_PARSE", r"\b(parse|chunk|ingest|ocr|extract|markdown|pdf|quote|normalize|text|claim)\b"),
    ("DATA_INGEST", r"\b(collect|capture|harvest|import|body_capture|source_import|drive_import)\b"),
    ("SCRAPING_CAPTURE", r"\b(scrape|crawler|crawl|url|website|http|browser|crawl)\b"),
    ("QUEUE_OR_WORKER", r"\b(consume|worker|runner|daemon|retry|watch|watcher|spine|queue|service_loop|loop|claim)\b"),
    ("GRAPH_WORKFLOW", r"\b(graph|timeline|chrono|promotion|river|evidence|ledger|edge|material)\b"),
    ("GOVERNANCE_AUDIT", r"\b(audit|gate|validate|validation|contract|check|snapshot|security|hard_truth)\b"),
    ("WORKFLOW_ORCHESTRATION", r"\b(workflow|orchestr|pipeline|dispatch|route|router|policy|state_machine|synthesis)\b"),
    ("MODEL_RUNTIME", r"\b(model|llm|embed|inference|predict|runtime|router|rerank|reranker|xgboost|shap)\b"),
    ("REPORTING_INSIGHT", r"\b(report|summary|dashboard|status|metric|fact|inventory|ledger_report)\b"),
    ("INTERFACE_CLI", r"\b(cli|surface|command|api|endpoint|cmd|shell|export|import|bridge)\b"),
    ("STORAGE_INTERFACE", r"\b(store|storage|archive|snapshot|artifact|bundle|drive|vault|file|dataset)\b"),
    ("REPO_TOOLING", r"\b(manifest|registry|inventory|register|catalog)\b"),
    ("SERVICE_INFRA", r"\b(service|daemon|health|monitor|watchdog|runtime)\b"),
]

REPEATABILITY_BUCKETS = [
    "REPEATABLE_STANDARD",
    "REPEATABLE_LIMITED",
    "ONE_TIME_LIKELY",
    "UNKNOWN_PRACTICAL_SCOPE",
]


CATEGORY_BUCKET_MAP = {
    "SCHEMAS": "SCHEMA_KNOWLEDGE",
    "BOOKS": "SCHEMA_KNOWLEDGE",
    "MODELS": "MODEL_RUNTIME",
    "LORAS": "MODEL_RUNTIME",
    "VAULT": "STORAGE_INTERFACE",
    "RUNTIME": "STORAGE_INTERFACE",
    "SERVICES": "SERVICE_INFRA",
    "SKILLS": "REPO_TOOLING",
    "PLUGINS": "REPO_TOOLING",
    "SURFACES": "INTERFACE_CLI",
    "WORKFLOWS": "WORKFLOW_ORCHESTRATION",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


@dataclass
class ToolRow:
    path: str
    source: str
    category: str
    kind: str
    name: str
    proof_hoard_role: str | None = None
    status: str | None = None
    what_it_does: str | None = None
    what_it_is: str | None = None
    one_line_description: str | None = None
    role: str | None = None
    verdict: str | None = None
    functions: list[str] = field(default_factory=list)
    known_uses: list[str] = field(default_factory=list)
    known_use_count: int | None = None
    bucket: str = "OTHER"
    bucket_signal: str | None = None
    repeatability: str = "UNKNOWN_PRACTICAL_SCOPE"
    repeatability_signal: str | None = None
    function_match_signals: list[str] = field(default_factory=list)
    tool_source: str | None = None
    size_bytes: int | None = None
    imports: list[str] = field(default_factory=list)
    entrypoints: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    script_audit_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "source": self.source,
            "category": self.category,
            "kind": self.kind,
            "name": self.name,
            "proof_hoard_role": self.proof_hoard_role,
            "status": self.status,
            "what_it_does": self.what_it_does,
            "what_it_is": self.what_it_is,
            "one_line_description": self.one_line_description,
            "role": self.role,
            "verdict": self.verdict,
            "function_bucket": self.bucket,
            "bucket_signal": self.bucket_signal,
            "repeatability": self.repeatability,
            "repeatability_signal": self.repeatability_signal,
            "function_match_signals": self.function_match_signals,
            "size_bytes": self.size_bytes,
            "known_use_count": self.known_use_count,
            "known_uses": self.known_uses,
            "functions": self.functions,
            "imports": self.imports,
            "entrypoints": self.entrypoints,
            "tags": self.tags,
            "tool_source": self.tool_source,
            "script_audit_path": self.script_audit_path,
        }


def _coerce_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    toolboxes = data.get("toolboxes", {})
    out: list[dict[str, Any]] = []
    if not isinstance(toolboxes, dict):
        return out
    for _, toolbox in toolboxes.items():
        if isinstance(toolbox, dict):
            items = toolbox.get("items")
            if isinstance(items, list):
                out.extend(items)
            continue
        if isinstance(toolbox, list):
            out.extend(toolbox)
    return out


def load_tickletrunk(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    return _coerce_json(payload)


def load_script_audit(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            p = rec.get("path") or rec.get("facts", {}).get("path")
            if not p:
                continue
            p = rel(p)
            row = {
                "source": "SCRIPT_AUDIT",
                "role": rec.get("role"),
                "verdict": rec.get("verdict"),
                "functions": (rec.get("facts", {}).get("functions") if isinstance(rec.get("facts", {}).get("functions"), list) else []),
                "tool_source": "SCRIPT_AUDIT_MANIFEST",
                "size_bytes": rec.get("facts", {}).get("size_bytes") or rec.get("size_bytes"),
                "what_it_does": rec.get("what_it_does") or rec.get("facts", {}).get("what_it_does"),
                "what_it_is": rec.get("what_it_is") or rec.get("facts", {}).get("what_it_is"),
                "one_line_description": rec.get("one_line_description") or rec.get("facts", {}).get("one_line_description"),
                "imports": rec.get("facts", {}).get("imports") or [],
                "entrypoints": rec.get("facts", {}).get("entrypoints") or [],
            }
            out[p] = row
    return out


def infer_bucket_from_functions(functions: list[str]) -> tuple[str | None, str | None]:
    if not functions:
        return None, None
    lowered = [fn.lower() for fn in functions]
    for bucket, pattern in FUNCTION_NAME_BUCKET_HINTS:
        for fn in lowered:
            if re.search(pattern, fn):
                return bucket, f"function:{fn}"
    return None, None


def infer_repeatability(row: ToolRow) -> tuple[str, str | None]:
    uses = row.known_use_count
    if isinstance(uses, int):
        if uses >= 5:
            return "REPEATABLE_STANDARD", f"known_use_count:{uses}"
        if uses >= 2:
            return "REPEATABLE_LIMITED", f"known_use_count:{uses}"
        if uses == 1:
            lower_path = row.path.lower()
            # Single observed use is often evidence of a single test or incomplete
            # telemetry capture, not necessarily a one-shot tool. Keep standard as
            # repeatable unless path/function explicitly signals one-off intent.
            if re.search(r"\b(one[-_]?time|tmp|temp|oneoff|ephemeral|bootstrap|migration|migrate|one.?off)\b", lower_path):
                return "ONE_TIME_LIKELY", f"known_use_count:1|path_hint:{lower_path}"
            return "REPEATABLE_LIMITED", "known_use_count:1"
        return "UNKNOWN_PRACTICAL_SCOPE", f"known_use_count:{uses}"

    if row.category == "SCRIPTS" and row.bucket in {"INTERFACE_CLI", "QUEUE_OR_WORKER", "WORKFLOW_ORCHESTRATION", "DOCUMENT_PARSE"}:
        return "REPEATABLE_LIMITED", "inferred_active_script_category"
    if "oneoff" in row.path.lower() or "one-off" in row.path.lower():
        return "ONE_TIME_LIKELY", "path_hint:oneoff"
    return "UNKNOWN_PRACTICAL_SCOPE", "no_known_use_count"


def infer_bucket(row: ToolRow) -> tuple[str, str | None]:
    function_bucket, function_signal = infer_bucket_from_functions(row.functions)
    if function_bucket:
        return function_bucket, function_signal

    if row.role:
        role_key = row.role.lower()
        bucket = ROLE_BUCKET_HINTS.get(role_key)
        if bucket:
            return bucket, f"role:{row.role}"

    category = (row.category or "").upper()
    if category in CATEGORY_BUCKET_MAP:
        return CATEGORY_BUCKET_MAP[category], f"category:{category}"

    kind = (row.kind or "").lower()
    if kind in {"schema", "book"}:
        return "SCHEMA_KNOWLEDGE", f"kind:{row.kind}"
    if kind in {"model", "lora"}:
        return "MODEL_RUNTIME", f"kind:{row.kind}"
    if kind in {"plugin", "skill"}:
        return "REPO_TOOLING", f"kind:{row.kind}"
    if kind in {"runtime_asset"}:
        return "STORAGE_INTERFACE", f"kind:{row.kind}"
    if row.source.startswith("01_REPOS"):
        return "REPO_ASSET", "path:repo"

    haystack = [
        row.path.lower(),
        (row.name or "").lower(),
        (row.what_it_does or "").lower(),
        (row.role or "").lower(),
        " ".join(row.tags).lower(),
        " ".join(row.functions).lower(),
        " ".join(row.known_uses).lower(),
    ]
    text = " ".join(haystack)

    for bucket, pattern in PATTERN_BUCKETS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return bucket, f"pattern:{pattern}"
    return "OTHER", None


def build_manifest(tickletrunk_path: Path, script_audit_path: Path) -> dict[str, Any]:
    rows: dict[str, ToolRow] = {}

    # seed with TICKLETRUNK entries (baseline universe)
    for item in load_tickletrunk(tickletrunk_path):
        path = rel(item.get("relative_path") or item.get("path") or "")
        if not path:
            continue
        rows[path] = ToolRow(
            path=path,
            source="TICKLETRUNK",
            category=item.get("category", "") or "",
            kind=item.get("kind", "") or "",
            name=item.get("name") or Path(path).name,
            proof_hoard_role=item.get("proof_hoard_role") or item.get("status") or None,
            status=item.get("status"),
            what_it_does=item.get("what_it_does") or item.get("description"),
            what_it_is=item.get("what_it_is"),
            one_line_description=item.get("one_line_description"),
            tags=list(item.get("tags") or []),
            known_uses=list(item.get("known_uses") or []),
            known_use_count=item.get("known_use_count"),
            imports=list(item.get("imports") or []),
            entrypoints=list(item.get("entrypoints") or []),
            size_bytes=item.get("size_bytes"),
            tool_source="TICKLETRUNK",
        )

    # enrich/update from script audit (deeper functional view)
    for path, audit in load_script_audit(script_audit_path).items():
        existing = rows.get(path)
        if existing is None:
            existing = ToolRow(
                path=path,
                source="SCRIPT_AUDIT",
                category="SCRIPTS",
                kind="script",
                name=Path(path).name,
                tool_source="SCRIPT_AUDIT_MANIFEST",
            )
            rows[path] = existing

        for k, v in audit.items():
            if v is None:
                continue
            if k in {"source"}:
                continue
            if k == "functions":
                existing.functions = list(v)
            elif k == "imports":
                existing.imports = list(v)
            elif k == "entrypoints":
                existing.entrypoints = list(v)
            elif k == "script_audit_path":
                existing.script_audit_path = str(v)
            elif k == "role":
                existing.role = str(v)
            elif k == "verdict":
                existing.verdict = str(v)
            elif k == "size_bytes":
                try:
                    existing.size_bytes = int(v)
                except Exception:
                    pass
            elif k in {"what_it_does", "what_it_is", "one_line_description"}:
                setattr(existing, k, v)
            elif k == "tool_source":
                existing.tool_source = str(v)
            elif k == "source":
                pass
            else:
                # preserve any future fields if needed
                pass

    for row in rows.values():
        bucket, signal = infer_bucket(row)
        row.bucket = bucket
        row.bucket_signal = signal
        row.repeatability, row.repeatability_signal = infer_repeatability(row)
        if not row.function_match_signals and row.functions:
            row.function_match_signals.append("functions_present_without_strong_match")

    bucket_counts = Counter(row.bucket for row in rows.values())
    repeatability_counts = Counter(row.repeatability for row in rows.values())
    source_counts = Counter(row.source for row in rows.values())
    category_counts = Counter(row.category or "UNKNOWN" for row in rows.values())
    role_counts = Counter(row.role or "_no_role_" for row in rows.values())

    items = [row.to_dict() for row in sorted(rows.values(), key=lambda x: (x.bucket, x.path))]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["function_bucket"]].append(item)

    return {
        "schema": "lucidota.tool_function_bucket_manifest.v1",
        "generated_at": utc_now(),
        "non_destructive": True,
        "tickletrunk_input": rel(tickletrunk_path),
        "script_audit_input": rel(script_audit_path),
        "scope": {
            "tool_rows": len(items),
            "function_buckets": len(FUNCTION_BUCKETS),
            "categories_in_scope": len(category_counts),
        },
        "counts": {
            "function_bucket": {bucket: int(bucket_counts.get(bucket, 0)) for bucket in FUNCTION_BUCKETS},
            "repeatability": {bucket: int(repeatability_counts.get(bucket, 0)) for bucket in REPEATABILITY_BUCKETS},
            "source": dict(source_counts),
            "category": dict(sorted(category_counts.items(), key=lambda kv: kv[0])),
            "role": dict(sorted(role_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "provenance": {
                "TICKLETRUNK": int(sum(1 for row in items if "TICKLETRUNK" in (row.get("source") or ""))),
                "SCRIPT_AUDIT": int(sum(1 for row in items if "SCRIPT_AUDIT" in (row.get("source") or ""))),
            },
        },
        "groups": {bucket: [item["path"] for item in rows_list[:400]] for bucket, rows_list in grouped.items()},
        "items": items,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LUCIDOTA tool/function bucket manifest",
        "",
        f"Generated: {payload['generated_at']}",
        f"Function buckets available: {payload['scope']['function_buckets']}",
        "",
        "No files were moved, copied, renamed, or deleted. Output is intended to be rerun identically.",
        "",
        "## Manifest-wide counts",
        f"- total_items: {payload['scope']['tool_rows']}",
        "",
        "## Counts by Function Bucket",
    ]
    for bucket in FUNCTION_BUCKETS:
        lines.append(f"- {bucket}: {payload['counts']['function_bucket'].get(bucket, 0)}")
    lines.append("")
    lines.append("## Repeatability buckets")
    for bucket in REPEATABILITY_BUCKETS:
        lines.append(f"- {bucket}: {payload['counts']['repeatability'].get(bucket, 0)}")

    lines.extend([
        "",
        "## Bucketed tool samples (top 35 each)",
        "",
    ])
    for bucket in FUNCTION_BUCKETS:
        lines.append(f"### {bucket}")
        limit = 0
        for row in payload["items"]:
            if row["function_bucket"] != bucket:
                continue
            if limit >= 35:
                lines.append("- *(truncated)*")
                break
            signal = row.get("bucket_signal") or "no signal"
            details = []
            if row.get("category"):
                details.append(row["category"])
            if row.get("role"):
                details.append(f"role={row['role']}")
            if row.get("verdict"):
                details.append(f"verdict={row['verdict']}")
            if row.get("tool_source"):
                details.append(f"source={row['tool_source']}")
            details.append(f"repeatability={row.get('repeatability', 'UNKNOWN_PRACTICAL_SCOPE')}")
            desc = " ; ".join(details)
            lines.append(f"- `{row['path']}` — {signal} — {desc}")
            limit += 1
        lines.append("")
    lines.append("## One-time-likely candidates (by inferred practical repeatability)")
    one_time_count = 0
    for row in payload["items"]:
        if row.get("repeatability") != "ONE_TIME_LIKELY":
            continue
        if one_time_count >= 120:
            lines.append("- *(truncated)*")
            break
        lines.append(f"- `{row['path']}` — {row['function_bucket']} — {row.get('repeatability_signal', '')}")
        one_time_count += 1
    if one_time_count == 0:
        lines.append("- (none)")
    return "\n".join(lines)


def render_bucket_reports(payload: dict[str, Any], bucket_order: list[str] | None = None) -> tuple[str, str, str, dict[str, str]]:
    order = bucket_order or FUNCTION_BUCKETS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in payload["items"]:
        grouped[item["function_bucket"]].append(item)

    full_lines = [
        "# Function bucket manifest (full by bucket)",
        f"Generated: {payload['generated_at']}",
        "",
    ]
    for bucket in order:
        rows = sorted(grouped.get(bucket, []), key=lambda row: row["path"])
        full_lines.append(f"## {bucket} ({len(rows)})")
        full_lines.extend([
            "| path | repeatability | bucket_signal | known_use_count | source |",
            "|---|---|---|---:|---|",
        ])
        for row in rows:
            full_lines.append(
                f"| {row['path']} | {row.get('repeatability', '')} | {row.get('bucket_signal', '')} | "
                f"{row.get('known_use_count', 0)} | {row.get('source', '')} |"
            )
        full_lines.append("")

    one_time_rows = [row for row in payload["items"] if row.get("repeatability") == "ONE_TIME_LIKELY"]
    one_time_lines = [
        "# One-time-likely candidates",
        "",
        f"Generated: {payload['generated_at']}",
        f"count: {len(one_time_rows)}",
        "",
        "| path | function_bucket | repeatability_signal | known_use_count |",
        "|---|---|---|---:|",
    ]
    if one_time_rows:
        for row in sorted(one_time_rows, key=lambda row: row["path"]):
            one_time_lines.append(
                f"| {row['path']} | {row['function_bucket']} | {row.get('repeatability_signal', '')} | "
                f"{row.get('known_use_count', '')} |"
            )
    else:
        one_time_lines.append("| (none) | | | |")

    index_lines = [
        "# Tool Function Buckets (Standardized run)",
        "",
        f"Generated from: {rel(OUT_ROOT / 'tool_function_bucket_manifest_latest.json')}",
        f"Total tools: {payload['scope']['tool_rows']}",
        f"Generated at: {payload['generated_at']}",
        "",
        "## Counts",
    ]
    for bucket in order:
        index_lines.append(f"- {bucket}: {len(grouped.get(bucket, []))}")

    index_lines.extend(["", "## Per-bucket artifacts", ""])
    for bucket in order:
        index_lines.append(f"- [{bucket}](by_bucket/{bucket}.md) -> {len(grouped.get(bucket, []))}")

    by_bucket_markdowns: dict[str, str] = {}
    for bucket in order:
        rows = sorted(grouped.get(bucket, []), key=lambda row: row["path"])
        lines = [
            f"# {bucket} ({len(rows)})",
            "",
            "| path | repeatability | use_count | bucket_signal | source |",
            "|---|---|---:|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| `{row['path']}` | {row.get('repeatability', '')} | "
                f"{row.get('known_use_count', 0)} | {row.get('bucket_signal', '')} | {row.get('source', '')} |"
            )
        by_bucket_markdowns[bucket] = "\n".join(lines)

    return (
        "\n".join(full_lines),
        "\n".join(one_time_lines),
        "\n".join(index_lines),
        by_bucket_markdowns,
    )


def render_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": item.get("path", ""),
            "function_bucket": item.get("function_bucket", ""),
            "bucket_signal": item.get("bucket_signal", ""),
            "repeatability": item.get("repeatability", ""),
            "repeatability_signal": item.get("repeatability_signal", ""),
            "category": item.get("category", ""),
            "kind": item.get("kind", ""),
            "source": item.get("source", ""),
            "tool_source": item.get("tool_source", ""),
            "role": item.get("role") or "",
            "verdict": item.get("verdict") or "",
            "known_use_count": item.get("known_use_count", ""),
            "size_bytes": item.get("size_bytes", ""),
            "entrypoints": ";".join(item.get("entrypoints", [])),
            "tags": ";".join(item.get("tags", [])),
            "functions": ";".join((item.get("functions") or [])[:16]),
            "proof_hoard_role": item.get("proof_hoard_role") or "",
        }
        for item in payload["items"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify tools by function with a repeatable, non-destructive manifest.")
    parser.add_argument("--tickletrunk", default=str(DEFAULT_TICKLETRUNK), help="Path to TICKLETRUNK.json")
    parser.add_argument("--script-audit", default=str(DEFAULT_SCRIPT_AUDIT), help="Path to SCRIPT_AUDIT_MANIFEST.jsonl")
    parser.add_argument("--output-dir", default=str(OUT_ROOT), help="Directory to write JSON/MD outputs")
    parser.add_argument("--no-write", action="store_true", help="Do not write output files")
    parser.add_argument("--no-latest", action="store_true", help="Do not refresh latest manifest pointers")
    parser.add_argument("--csv", action="store_true", help="Also write CSV manifest output")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    args = parser.parse_args()

    payload = build_manifest(Path(args.tickletrunk), Path(args.script_audit))
    if not args.no_write:
        out = Path(args.output_dir)
        if not out.is_absolute():
            out = ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        base = out / f"tool_function_bucket_manifest_{stamp()}"
        json_path = base.with_suffix(".json")
        md_path = base.with_suffix(".md")
        csv_path = base.with_suffix(".csv")
        latest_json = out / "tool_function_bucket_manifest_latest.json"
        latest_md = out / "tool_function_bucket_manifest_latest.md"
        latest_csv = out / "tool_function_bucket_manifest_latest.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")
        full_by_bucket, one_time_md, bucket_index, by_bucket = render_bucket_reports(payload, bucket_order=REPORT_BUCKET_ORDER)
        (out / "tool_function_bucket_manifest_latest_full_by_bucket.md").write_text(full_by_bucket, encoding="utf-8")
        (out / "tool_function_one_time_likely.md").write_text(one_time_md, encoding="utf-8")
        (out / "bucket_by_bucket_index.md").write_text(bucket_index, encoding="utf-8")
        by_bucket_dir = out / "by_bucket"
        by_bucket_dir.mkdir(parents=True, exist_ok=True)
        for bucket_name, text in by_bucket.items():
            (by_bucket_dir / f"{bucket_name}.md").write_text(text, encoding="utf-8")

        if args.csv:
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "path",
                        "function_bucket",
                        "bucket_signal",
                        "repeatability",
                        "repeatability_signal",
                        "category",
                        "kind",
                        "source",
                        "tool_source",
                        "role",
                        "verdict",
                        "known_use_count",
                        "size_bytes",
                        "entrypoints",
                        "tags",
                        "functions",
                        "proof_hoard_role",
                    ],
                )
                writer.writeheader()
                writer.writerows(render_csv_rows(payload))
        if not args.no_latest:
            shutil.copy2(json_path, latest_json)
            shutil.copy2(md_path, latest_md)
            if args.csv:
                shutil.copy2(csv_path, latest_csv)
        payload["report_json"] = rel(json_path)
        payload["report_markdown"] = rel(md_path)
        if args.csv:
            payload["report_csv"] = rel(csv_path)
        print("REPORT_JSON=" + rel(json_path))
        print("REPORT_MD=" + rel(md_path))
        if args.csv:
            print("REPORT_CSV=" + rel(csv_path))
        if not args.no_latest:
            print("REPORT_JSON_LATEST=" + rel(latest_json))
            print("REPORT_MD_LATEST=" + rel(latest_md))
            if args.csv:
                print("REPORT_CSV_LATEST=" + rel(latest_csv))

    if args.json:
        print(json.dumps(payload, sort_keys=True))

    print("TOOL_FUNCTION_BUCKET_MANIFEST=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
