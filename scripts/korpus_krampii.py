#!/usr/bin/env python3
"""KORPUS KRAMPII mass-ingestion workflow.

Fast deterministic reclamation of legacy sprawl:
  scan -> sha256 -> UUID -> CAS lock -> exact dedupe -> componentize ->
  entity/concept extraction -> GO routing -> Postgres indexes -> ABSURD event.

No LLM calls. No prompt-thinking lane. Algorithms only.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import datetime as dt
import hashlib
import math
import pickle
import json
import mimetypes
import os
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import psycopg

ROOT = Path(__file__).resolve().parents[1]
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")
STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state"))
KORPUS_SCHEMA = ROOT / "06_SCHEMA" / "019_korpus_krampii.sql"
INVESTIGATION_SCHEMA = ROOT / "06_SCHEMA" / "018_investigation_artifact.sql"
GO_SCHEMA = ROOT / "06_SCHEMA" / "016_go_graph_core.sql"
VAULT_SCHEMA = ROOT / "06_SCHEMA" / "005_cas_manifest.sql"
WORKFLOW_SCHEMA = ROOT / "06_SCHEMA" / "006_workflow_registry.sql"
CONTROL_SCHEMA = ROOT / "06_SCHEMA" / "001_lucidota_control.sql"
DERIVED_QUEUE_SCHEMA = ROOT / "06_SCHEMA" / "020_korpus_derived_compute_queue.sql"
DROP_ROOT = ROOT / "03_VAULT" / "korpus_krampii" / "DROP"
OUTPUT_ROOT = ROOT / "05_OUTPUTS" / "korpus_krampii"
NAMESPACE = uuid.UUID("00000000-0000-4000-8000-000000000414")
MAX_COMPONENT_TOKENS = 500
EMBED_MODEL = "ckdog1.kernel.hash_quantized_embedding.v1"
RIVER_MODEL_KEY = "korpus_component_riverml_stats_v1"
RIVER_STATE_VERSION = "OperatorVibes+CyberneticTelemetry+HalfSpaceTrees64Scaled+DBSTREAM+HoeffdingTreeClassifier+MeanVar.v4"
RIVER_BULK_LIGHT_ENV = "LUCIDOTA_KORPUS_RIVER_MODE"
NEAR_DUP_MODE_ENV = "LUCIDOTA_KORPUS_NEAR_DUP_MODE"
GRAPH_MODE_ENV = "LUCIDOTA_KORPUS_GRAPH_MODE"
LEGACY_DIRECT_GRAPH_ENV = "LUCIDOTA_KORPUS_ALLOW_LEGACY_DIRECT_GRAPH"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "legacy"))
sys.path.insert(0, str(ROOT / "01_REPOS" / "doggystyle"))

from ALGOS.minhash import similarity as minhash_similarity  # type: ignore  # noqa: E402
from ALGOS.korpus_text import entropy_for_text, minhash_for_text, vector_literal  # type: ignore  # noqa: E402
from ALGOS.krampus_chrono import (  # type: ignore  # noqa: E402
    chrono_file_date,
    chronological_key,
    parse_loose_datetime,
)
from ALGOS.krampus_stickers import (  # type: ignore  # noqa: E402
    component_features,
    dbstream_features,
    heuristic_river_label,
    hst_features,
    operator_cluster_hint,
    vibe_tags_for,
)
from river import anomaly as river_anomaly, cluster as river_cluster, stats as river_stats, tree as river_tree  # type: ignore  # noqa: E402
from lucidota_artifact_ingest import (  # type: ignore  # noqa: E402
    classify_kind,
    detect_mime,
    ensure_case,
    ensure_state,
    entity_graph_location,
    entity_graph_payload,
    entity_graph_term,
    extract_entities,
    insert_graph_edge,
    insert_graph_item,
    normalize_ws,
    sha256_file,
    should_promote_entity_to_graph,
    store_locked_cas,
)
from lucidota_clawd_runtime import route_go25  # type: ignore  # noqa: E402

TEXT_SUFFIXES = {
    ".txt", ".md", ".mdx", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".sql", ".xml", ".html", ".htm", ".css",
}
CODE_SUFFIXES = {
    ".py", ".rs", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".php", ".rb", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".psd1", ".lua",
    ".r", ".jl", ".swift", ".kt", ".scala", ".pl", ".pm", ".dart", ".vue", ".svelte",
}
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "target", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".cache", ".next", "dist", "build", ".tox", ".ruff_cache", "04_RUNTIME", "05_OUTPUTS",
}
DEFAULT_EXCLUDE_REL_PREFIXES = {
    "03_VAULT/cas/", "03_VAULT/korpus_krampii/", "03_VAULT/cases/", "snap/", ".cache/",
}
CONCEPT_PATTERNS = [
    ("goal", "ACTION", re.compile(r"\b(?:GOAL|OBJECTIVE|MISSION)\b\s*[:\-]?\s*(.{3,220})", re.I), 69),
    ("action_item", "ACTION", re.compile(r"\b(?:" + "TO" + r"DO|FIXME|NEXT|ACTION ITEM)\b\s*[:\-]?\s*(.{3,220})", re.I), 69),
    ("idea", "HYPOTHESIS", re.compile(r"\b(?:IDEA|HUNCH|MAYBE|PROPOSAL)\b\s*[:\-]?\s*(.{3,220})", re.I), 50),
    ("note", "COMMENT", re.compile(r"\b(?:NOTE|OBSERVATION|REMEMBER)\b\s*[:\-]?\s*(.{3,220})", re.I), 50),
    ("bug", "FRICTION", re.compile(r"\b(?:BUG|BROKEN|FAILS?|ERROR|ISSUE)\b\s*[:\-]?\s*(.{3,220})", re.I), 50),
    ("tool", "TOOL", re.compile(r"\b(?:tool|script|command|cli|workflow)\b\s*[:=\-]?\s*([A-Za-z0-9_./:@\-]{2,160})", re.I), 50),
    ("relationship", "RELATIONSHIP", re.compile(r"([^\n]{2,100}?)\s*(?:->|=>|depends on|uses|calls|imports|links to)\s*([^\n]{2,120})", re.I), 50),
]
CODE_SYMBOL_REGEX = {
    ".py": re.compile(r"^\s*(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M),
    ".rs": re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|impl)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M),
    ".js": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+([A-Za-z_$][A-Za-z0-9_$]*)|^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", re.M),
    ".ts": re.compile(r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type)\s+([A-Za-z_$][A-Za-z0-9_$]*)|^\s*(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=", re.M),
    ".sql": re.compile(r"^\s*(?:CREATE|ALTER)\s+(?:TABLE|VIEW|FUNCTION|SCHEMA|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_.\"]+)", re.I | re.M),
    ".sh": re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{", re.M),
    ".ps1": re.compile(r"^\s*function\s+([A-Za-z0-9_\-]+)", re.I | re.M),
}


@dataclass
class ComponentDraft:
    component_kind: str
    language: str
    title: str
    symbol: str
    start_line: int
    end_line: int
    content: str


@dataclass
class FileResult:
    ok: bool
    path: str
    root: str
    relative_path: str
    sha256: str = ""
    file_uuid: str = ""
    size_bytes: int = 0
    mime: str = ""
    file_kind: str = ""
    suffix: str = ""
    cas_uri: str = ""
    locked_relative_path: str = ""
    mtime: str = ""
    components: list[dict[str, Any]] | None = None
    file_minhash: list[int] | None = None
    error: str = ""
    skipped_reason: str = ""
    detail: dict[str, Any] | None = None


def pg_clean_text(value: str) -> str:
    """Postgres text/jsonb cannot store NUL bytes; preserve position with U+FFFD."""
    return str(value).replace("\x00", "�")


def pg_clean(obj: Any) -> Any:
    if isinstance(obj, str):
        return pg_clean_text(obj)
    if isinstance(obj, dict):
        return {pg_clean_text(str(k)): pg_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [pg_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [pg_clean(v) for v in obj]
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(pg_clean(obj), ensure_ascii=False, sort_keys=True, default=str)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def utc_from_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def file_uuid_for_sha(sha: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"korpus-file:{sha}"))


def component_uuid_for(file_uuid: str, component_sha: str, index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"korpus-component:{file_uuid}:{index}:{component_sha}"))


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def read_text(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pg_clean_text(data.decode(enc, errors="replace"))
        except (LookupError, UnicodeError):
            continue
    return pg_clean_text(data.decode("utf-8", errors="replace"))


def ext_language(suffix: str) -> str:
    mapping = {
        ".py": "python", ".rs": "rust", ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
        ".md": "markdown", ".mdx": "markdown", ".sql": "sql", ".sh": "shell", ".bash": "shell", ".ps1": "powershell",
        ".json": "json", ".jsonl": "jsonl", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".html": "html", ".css": "css",
    }
    return mapping.get(suffix, suffix.removeprefix("."))


def split_long_component(kind: str, language: str, title: str, symbol: str, start_line: int, lines: list[str]) -> list[ComponentDraft]:
    out: list[ComponentDraft] = []
    buf: list[str] = []
    buf_start = start_line
    for offset, line in enumerate(lines):
        tentative = "\n".join(buf + [line])
        if buf and token_count(tentative) > MAX_COMPONENT_TOKENS:
            content = normalize_ws("\n".join(buf))
            out.append(ComponentDraft(kind, language, title, symbol, buf_start, start_line + offset - 1, content))
            buf = [line]
            buf_start = start_line + offset
        else:
            buf.append(line)
    if buf:
        content = normalize_ws("\n".join(buf))
        if content:
            out.append(ComponentDraft(kind, language, title, symbol, buf_start, start_line + len(lines) - 1, content))
    return out


def markdown_components(text: str, language: str) -> list[ComponentDraft]:
    """Split Markdown into parent sections plus fenced code-block child artifacts."""
    lines = text.splitlines()
    out: list[ComponentDraft] = []
    starts = [i for i, line in enumerate(lines) if re.match(r"^#{1,6}\s+", line)]
    if starts:
        starts.append(len(lines))
        for pos, start in enumerate(starts[:-1]):
            end = starts[pos + 1]
            title = re.sub(r"^#{1,6}\s+", "", lines[start]).strip()[:180]
            out.extend(split_long_component("markdown_section", language, title, "", start + 1, lines[start:end]))
    else:
        out.extend(split_long_component("markdown_document", language, "document", "", 1, lines))

    fence_re = re.compile(r"^```\s*([A-Za-z0-9_+.-]*)\s*$")
    i = 0
    block_num = 0
    while i < len(lines):
        m = fence_re.match(lines[i])
        if not m:
            i += 1
            continue
        lang = (m.group(1) or "code").lower()
        start = i + 1
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```"):
            j += 1
        code_lines = lines[i + 1:j]
        content = normalize_ws("\n".join(code_lines))
        if content:
            block_num += 1
            out.extend(split_long_component("markdown_code_block", lang, f"fenced code block {block_num} ({lang})", "", start + 1, code_lines))
        i = j + 1
    return sorted(out, key=lambda c: (c.start_line, c.component_kind, c.title))


def python_ast_components(text: str) -> list[ComponentDraft]:
    """Use Python's native AST: functions/classes/methods/docstrings become components."""
    lines = text.splitlines()
    out: list[ComponentDraft] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return split_long_component("code_chunk", "python", "syntax_error_chunk", "", 1, lines)

    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc:
        doc_lines = module_doc.splitlines()
        out.extend(split_long_component("python_docstring", "python", "module docstring", "__doc__", 1, doc_lines))

    class_stack: list[str] = []

    def emit_node(node: ast.AST, kind: str, symbol: str) -> None:
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line))
        chunk = lines[start_line - 1:end_line]
        out.extend(split_long_component(kind, "python", symbol, symbol, start_line, chunk))
        doc = ast.get_docstring(node, clean=False) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else None
        if doc:
            out.extend(split_long_component(f"{kind}_docstring", "python", f"{symbol} docstring", symbol, start_line, doc.splitlines()))

    def visit_body(body: list[ast.stmt], parent: str = "") -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                symbol = f"{parent}.{node.name}" if parent else node.name
                emit_node(node, "python_class", symbol)
                visit_body(node.body, symbol)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = f"{parent}.{node.name}" if parent else node.name
                emit_node(node, "python_method" if parent else "python_function", symbol)

    # prelude/imports before first def/class
    first_symbol = min((getattr(n, "lineno", 999999) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))), default=0)
    if first_symbol and first_symbol > 1:
        pre = lines[: first_symbol - 1]
        if normalize_ws("\n".join(pre)):
            out.extend(split_long_component("code_prelude", "python", "imports/prelude", "", 1, pre))
    visit_body(tree.body)
    return sorted(out or split_long_component("code_chunk", "python", "module", "", 1, lines), key=lambda c: (c.start_line, c.component_kind, c.symbol))


def code_components(text: str, suffix: str, language: str) -> list[ComponentDraft]:
    if suffix == ".py":
        comps = python_ast_components(text)
        if len(comps) == 1 and comps[0].component_kind == "code_chunk":
            comps[0].component_kind = "python_syntax_error_chunk"
        return comps
    rx = CODE_SYMBOL_REGEX.get(suffix)
    if not rx and suffix in {".jsx", ".tsx"}:
        rx = CODE_SYMBOL_REGEX.get(".js")
    lines = text.splitlines()
    out: list[ComponentDraft] = []
    matches = list(rx.finditer(text)) if rx else []
    if matches:
        line_starts = [0]
        for m in re.finditer("\n", text):
            line_starts.append(m.end())
        def line_no(pos: int) -> int:
            n = 1
            for s0 in line_starts:
                if s0 <= pos:
                    n += 1 if s0 else 0
                else:
                    break
            return max(1, n)
        spans = []
        for idx, m in enumerate(matches):
            start_pos = m.start()
            end_pos = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            symbol = next((g for g in m.groups() if g), "") if m.groups() else m.group(1)
            spans.append((line_no(start_pos), line_no(end_pos), symbol, text[start_pos:end_pos].splitlines()))
        if matches[0].start() > 0:
            pre = text[: matches[0].start()].splitlines()
            if normalize_ws("\n".join(pre)):
                out.extend(split_long_component("code_prelude", language, "prelude", "", 1, pre))
        for start_line, end_line, symbol, chunk_lines in spans:
            out.extend(split_long_component("code_symbol", language, symbol, symbol, start_line, chunk_lines))
    else:
        out.extend(split_long_component("code_chunk", language, "chunk", "", 1, lines))
    return out


def generic_text_components(text: str, suffix: str, language: str) -> list[ComponentDraft]:
    if suffix in {".md", ".mdx", ".rst"}:
        return markdown_components(text, language)
    if suffix in CODE_SUFFIXES or suffix in CODE_SYMBOL_REGEX:
        return code_components(text, suffix, language)
    return split_long_component("text_chunk", language, "chunk", "", 1, text.splitlines())


def load_river_state(conn: psycopg.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT payload, sample_count FROM lucidota_korpus.river_model_blob WHERE model_key=%s", (RIVER_MODEL_KEY,)).fetchone()
    state: dict[str, Any] | None = None
    if row and os.environ.get("LUCIDOTA_DISABLE_DB_PICKLE_STATE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        try:
            loaded = pickle.loads(bytes(row[0]))
            if isinstance(loaded, dict):
                state = loaded
                state["sample_count"] = int(row[1] or state.get("sample_count", 0))
        except (pickle.PickleError, EOFError, TypeError, ValueError):
            state = None
    if state is None:
        state = {"sample_count": 0}
    state.setdefault("means", {})
    state.setdefault("vars", {})
    # Actual RiverML online models: anomaly scoring plus incremental route learning.
    # Keep this self-healing so older pickled Mean/Var-only state upgrades in place.
    if "hst" not in state or state.get("river_version") != RIVER_STATE_VERSION:
        state["hst"] = river_anomaly.HalfSpaceTrees(n_trees=16, height=6, window_size=64, seed=414)
        state["dbstream"] = river_cluster.DBSTREAM(clustering_threshold=0.55, fading_factor=0.01, cleanup_interval=64, minimum_weight=1.0)
    if "classifier" not in state or state.get("river_version") != RIVER_STATE_VERSION:
        state["classifier"] = river_tree.HoeffdingTreeClassifier(grace_period=20, delta=1e-5)
    if "dbstream" not in state:
        state["dbstream"] = river_cluster.DBSTREAM(clustering_threshold=0.55, fading_factor=0.01, cleanup_interval=64, minimum_weight=1.0)
    state["river_version"] = RIVER_STATE_VERSION
    return state


def save_river_state(conn: psycopg.Connection, state: dict[str, Any]) -> None:
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    conn.execute(
        """
        INSERT INTO lucidota_korpus.river_model_blob(model_key, model_kind, payload, sample_count, detail)
        VALUES (%s,'river.OperatorVibes+HalfSpaceTrees+DBSTREAM+HoeffdingTreeClassifier+MeanVar',%s,%s,%s::jsonb)
        ON CONFLICT(model_key) DO UPDATE SET model_kind=EXCLUDED.model_kind, payload=EXCLUDED.payload, sample_count=EXCLUDED.sample_count, detail=EXCLUDED.detail, updated_at=now()
        """,
        (
            RIVER_MODEL_KEY,
            payload,
            int(state.get("sample_count", 0)),
            jdump(
                {
                    "library": "river",
                    "mode": "online_one_component_at_a_time",
                    "models": ["anomaly.HalfSpaceTrees", "cluster.DBSTREAM", "tree.HoeffdingTreeClassifier", "stats.Mean", "stats.Var"],
                    "learn_order": "score_then_learn",
                }
            ),
        ),
    )


def river_decide_component(conn: psycopg.Connection, batch_uuid: str, file_uuid: str, component_uuid: str, comp: dict[str, Any]) -> dict[str, Any]:
    features = component_features(comp)
    river_mode = os.environ.get(RIVER_BULK_LIGHT_ENV, "full").strip().lower()
    if river_mode in {"light", "bulk-light", "off", "disabled"}:
        route = heuristic_river_label(comp, features)
        score = 0
        vibe_spike = False
        detail = {
            "pre_update_sample_count": None,
            "river_package": "bypassed_for_bulk_ingest",
            "models": [],
            "bulk_light_mode": True,
            "reason": f"{RIVER_BULK_LIGHT_ENV}={river_mode}",
            "operator_cluster_hint": operator_cluster_hint(features),
            "heuristic_label": route,
            "score_parts": {"z_bps": 0, "anomaly_bps": 0},
            "dbstream_features": dbstream_features({key: float(value) for key, value in features.items()}),
        }
        conn.execute(
            """
            UPDATE lucidota_korpus.component
            SET river_decision=%s, river_score=%s, vibe_spike=%s, river_features=%s::jsonb
            WHERE component_uuid=%s::uuid
            """,
            (route, score, vibe_spike, jdump(features), component_uuid),
        )
        conn.execute(
            """
            INSERT INTO lucidota_korpus.river_decision(batch_uuid, file_uuid, component_uuid, model_key, decision_route, score, vibe_spike, features, detail)
            VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT(component_uuid, model_key) DO UPDATE SET decision_route=EXCLUDED.decision_route, score=EXCLUDED.score, vibe_spike=EXCLUDED.vibe_spike, features=EXCLUDED.features, detail=EXCLUDED.detail
            """,
            (batch_uuid, file_uuid, component_uuid, RIVER_MODEL_KEY, route, score, vibe_spike, jdump(features), jdump(detail)),
        )
        return {"decision_route": route, "score": score, "vibe_spike": vibe_spike, "features": features, "detail": detail}
    state = load_river_state(conn)
    means: dict[str, Any] = state.setdefault("means", {})
    variances: dict[str, Any] = state.setdefault("vars", {})
    hst = state["hst"]
    classifier = state["classifier"]
    dbstream = state["dbstream"]
    sample_count = int(state.get("sample_count", 0))
    max_z = 0.0
    z_scores: dict[str, float] = {}
    numeric_features = {key: float(value) for key, value in features.items()}
    bounded_features = hst_features(numeric_features)
    stream_features = dbstream_features(numeric_features)
    for key, value in features.items():
        mean = means.get(key)
        var = variances.get(key)
        if mean is None:
            mean = means[key] = river_stats.Mean()
        if var is None:
            var = variances[key] = river_stats.Var()
        if sample_count >= 5:
            std = math.sqrt(max(float(var.get()), 1e-9))
            z = abs(float(value) - float(mean.get())) / std
            z_scores[key] = z
            max_z = max(max_z, z)
    try:
        anomaly_score = float(hst.score_one(bounded_features) or 0.0)
    except Exception as exc:
        print(f"warning: River HalfSpaceTrees score_one failed: {exc}", file=sys.stderr)
        anomaly_score = 0.0
    try:
        classifier_pred = classifier.predict_one(numeric_features)
    except Exception as exc:
        print(f"warning: River classifier predict_one failed: {exc}", file=sys.stderr)
        classifier_pred = None
    try:
        proba = classifier.predict_proba_one(numeric_features) or {}
    except Exception as exc:
        print(f"warning: River classifier predict_proba_one failed: {exc}", file=sys.stderr)
        proba = {}
    try:
        dbstream_cluster = dbstream.predict_one(stream_features)
    except Exception as exc:
        print(f"warning: River DBSTREAM predict_one failed: {exc}", file=sys.stderr)
        dbstream_cluster = None
    classifier_conf = float(max(proba.values())) if proba else 0.0
    anomaly_bps = int(round(max(0.0, min(1.0, anomaly_score)) * 10000))
    z_bps = int(round(max_z * 1000))
    score = max(0, min(10000, max(z_bps, anomaly_bps)))
    vibe_spike = sample_count >= 5 and (score >= 3500 or anomaly_score >= 0.75)
    heuristic_label = heuristic_river_label(comp, features)
    if vibe_spike:
        route = "vibe_spike"
    elif classifier_pred and classifier_conf >= 0.55:
        route = str(classifier_pred)
    else:
        route = heuristic_label
    # River online learn_one: score/predict first, then learn exactly one component.
    for key, value in features.items():
        means[key].update(float(value))
        variances[key].update(float(value))
    try:
        hst.learn_one(bounded_features)
    except Exception as exc:
        print(f"warning: River HalfSpaceTrees learn_one failed: {exc}", file=sys.stderr)
    try:
        dbstream.learn_one(stream_features)
    except Exception as exc:
        print(f"warning: River DBSTREAM learn_one failed: {exc}", file=sys.stderr)
    try:
        classifier.learn_one(numeric_features, heuristic_label)
    except Exception as exc:
        print(f"warning: River classifier learn_one failed: {exc}", file=sys.stderr)
    state["sample_count"] = sample_count + 1
    save_river_state(conn, state)
    detail = {
        "z_scores": {k: round(v, 4) for k, v in z_scores.items()},
        "pre_update_sample_count": sample_count,
        "river_package": "river",
        "models": ["anomaly.HalfSpaceTrees", "cluster.DBSTREAM", "tree.HoeffdingTreeClassifier", "stats.Mean", "stats.Var"],
        "anomaly_score": round(anomaly_score, 6),
        "hst_features": {k: round(v, 6) for k, v in bounded_features.items()},
        "dbstream_features": {k: round(v, 6) for k, v in stream_features.items()},
        "dbstream_cluster": dbstream_cluster,
        "operator_cluster_hint": operator_cluster_hint(features),
        "classifier_pred": classifier_pred,
        "classifier_conf": round(classifier_conf, 6),
        "classifier_proba": {str(k): round(float(v), 6) for k, v in proba.items()},
        "heuristic_label": heuristic_label,
        "score_parts": {"z_bps": z_bps, "anomaly_bps": anomaly_bps},
    }
    conn.execute(
        """
        UPDATE lucidota_korpus.component
        SET river_decision=%s, river_score=%s, vibe_spike=%s, river_features=%s::jsonb
        WHERE component_uuid=%s::uuid
        """,
        (route, score, vibe_spike, jdump(features), component_uuid),
    )
    conn.execute(
        """
        INSERT INTO lucidota_korpus.river_decision(batch_uuid, file_uuid, component_uuid, model_key, decision_route, score, vibe_spike, features, detail)
        VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
        ON CONFLICT(component_uuid, model_key) DO UPDATE SET decision_route=EXCLUDED.decision_route, score=EXCLUDED.score, vibe_spike=EXCLUDED.vibe_spike, features=EXCLUDED.features, detail=EXCLUDED.detail
        """,
        (batch_uuid, file_uuid, component_uuid, RIVER_MODEL_KEY, route, score, vibe_spike, jdump(features), jdump(detail)),
    )
    return {"decision_route": route, "score": score, "vibe_spike": vibe_spike, "features": features, "detail": detail}


def concepts_from_component(component: ComponentDraft, go_terms: list[str]) -> list[dict[str, Any]]:
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    text = component.content[:100_000]
    if component.title and component.component_kind.startswith("markdown"):
        norm = normalize_ws(component.title).lower()
        found[("heading", norm, "heading")] = {
            "concept_kind": "heading", "value": component.title, "normalized_value": norm,
            "go_term": "TERM", "source": "markdown_heading", "confidence_bps": 50,
        }
    if component.symbol:
        norm = component.symbol.lower()
        go = "ALGORITHM" if component.language in {"python", "rust", "javascript", "typescript", "sql", "shell", "powershell"} else "TOOL"
        found[("symbol", norm, "code_symbol")] = {
            "concept_kind": "symbol", "value": component.symbol, "normalized_value": norm,
            "go_term": go, "source": "code_symbol", "confidence_bps": 69,
        }
    for kind, go, rx, conf in CONCEPT_PATTERNS:
        for m in rx.finditer(text):
            val = normalize_ws(" -> ".join(g for g in m.groups() if g) if kind == "relationship" else m.group(1))[:240]
            norm = val.lower()
            if val:
                found[(kind, norm, "regex")] = {
                    "concept_kind": kind, "value": val, "normalized_value": norm,
                    "go_term": go, "source": f"regex:{kind}", "confidence_bps": conf,
                }
    # Preserve strong GO route concepts even without explicit marker.
    for go in go_terms[:3]:
        if go in {"TOOL", "ALGORITHM", "ACTION", "PATTERN", "HYPOTHESIS", "CLAIM", "EVIDENCE", "RELATIONSHIP"}:
            val = component.title or component.symbol or text[:160]
            norm = normalize_ws(val).lower()[:240]
            found[("go_route", norm + go.lower(), "go_route")] = {
                "concept_kind": "go_route", "value": val[:240], "normalized_value": f"{go.lower()}:{norm}",
                "go_term": go, "source": "ckdog1_go_route", "confidence_bps": 10,
            }
    return sorted(found.values(), key=lambda c: (c["concept_kind"], c["normalized_value"]))


def iter_files(roots: list[Path], include_hidden: bool, no_default_excludes: bool) -> Iterable[tuple[Path, Path]]:
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file():
            yield root.parent, root
            continue
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            if not no_default_excludes:
                dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                filenames = [f for f in filenames if not f.startswith(".")]
            for name in filenames:
                p = dp / name
                if not no_default_excludes:
                    rp = rel(p)
                    if any(rp.startswith(prefix) for prefix in DEFAULT_EXCLUDE_REL_PREFIXES):
                        continue
                yield root, p


def quick_mime(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0]
    if mime:
        return mime, "mimetypes"
    return detect_mime(path)[0], "file"


def process_file(root: Path, path: Path, max_file_mb: int, max_text_mb: int, deep_media: bool) -> FileResult:
    try:
        st = path.stat()
        if not path.is_file():
            return FileResult(False, str(path), str(root), "", skipped_reason="not_file")
        digest = sha256_file(path)
        file_uuid = ""  # DB assigns ordered UUIDv7 on first sight; sha256 is the dedupe key.
        suffix = path.suffix.lower()
        mime, mime_source = quick_mime(path)
        file_kind = "code" if suffix in CODE_SUFFIXES else classify_kind(path, mime)
        chrono = chrono_file_date(path)
        base_detail = {
            "mime_source": mime_source,
            "chrono_original_date": chrono["iso"],
            "chrono_date_source": chrono["source"],
            "chrono_confidence_bps": chrono["confidence_bps"],
            "chrono_raw": chrono.get("raw", ""),
            "chrono_candidate_count": chrono.get("candidate_count", 0),
        }
        cas_uri, locked_path = store_locked_cas(path, digest, mime, "korpus_krampii")
        components: list[dict[str, Any]] = []
        file_sig: list[int] = []
        max_bytes = max_text_mb * 1024 * 1024
        is_textual = file_kind in {"text", "code", "document"} or suffix in TEXT_SUFFIXES or suffix in CODE_SUFFIXES
        if is_textual and st.st_size > max_file_mb * 1024 * 1024:
            return FileResult(True, str(path), str(root), rel_to_root(root, path), digest, file_uuid, st.st_size, mime, file_kind, suffix, cas_uri, rel(locked_path), utc_from_ts(st.st_mtime), [], [], detail={**base_detail, "parser_status": "deferred", "deferred_reason": "text_larger_than_fast_path_limit"})
        if (not is_textual) and (not deep_media):
            return FileResult(True, str(path), str(root), rel_to_root(root, path), digest, file_uuid, st.st_size, mime, file_kind, suffix, cas_uri, rel(locked_path), utc_from_ts(st.st_mtime), [], [], detail={**base_detail, "parser_status": "deferred", "deferred_reason": "unsupported_mime_store_now_parse_later"})
        if is_textual and st.st_size <= max_file_mb * 1024 * 1024:
            text = read_text(path, max_bytes)
            file_sig = minhash_for_text(text, k=64)
            drafts = generic_text_components(text, suffix, ext_language(suffix))
            route_cache: dict[str, list[str]] = {}
            for idx, comp in enumerate(drafts):
                content_sha = hashlib.sha256(comp.content.encode("utf-8", errors="ignore")).hexdigest()
                if content_sha not in route_cache:
                    route_cache[content_sha] = [r["term"] for r in route_go25(comp.content[:4000], top_n=5)]
                go_terms = route_cache[content_sha]
                ents = extract_entities(comp.content, {"FileName": path.name, "Title": comp.title or path.stem})
                concepts = concepts_from_component(comp, go_terms)
                components.append({
                    "component_uuid": "",
                    "component_index": idx,
                    "component_kind": comp.component_kind,
                    "language": comp.language,
                    "title": comp.title,
                    "symbol": comp.symbol,
                    "start_line": comp.start_line,
                    "end_line": comp.end_line,
                    "sha256": content_sha,
                    "token_count": token_count(comp.content),
                    "content": comp.content,
                    "go_terms": go_terms,
                    "minhash_signature": minhash_for_text(comp.content, k=64),
                    "entropy": entropy_for_text(comp.content),
                    "entities": ents,
                    "concepts": concepts,
                })
        elif deep_media:
            # Deep media is deliberately queued to artifact-ingest instead of faking recognition in mass fast path.
            components.append({
                "component_uuid": "",
                "component_index": 0,
                "component_kind": "media_placeholder",
                "language": "",
                "title": path.name,
                "symbol": "",
                "start_line": 0,
                "end_line": 0,
                "sha256": digest,
                "token_count": 0,
                "content": "",
                "go_terms": ["EVIDENCE"],
                "minhash_signature": [],
                "entropy": 0.0,
                "entities": [],
                "concepts": [{"concept_kind":"deep_media_queued","value":path.name,"normalized_value":path.name.lower(),"go_term":"EVIDENCE","source":"korpus_fast_path","confidence_bps":10}],
            })
        return FileResult(True, str(path), str(root), rel_to_root(root, path), digest, file_uuid, st.st_size, mime, file_kind, suffix, cas_uri, rel(locked_path), utc_from_ts(st.st_mtime), components, file_sig, detail=base_detail)
    except Exception as exc:
        return FileResult(False, str(path), str(root), rel_to_root(root, path), error=str(exc))


def rel_to_root(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def ensure_storage(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(VAULT_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(GO_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(INVESTIGATION_SCHEMA.read_text(encoding="utf-8"))
        cur.execute(KORPUS_SCHEMA.read_text(encoding="utf-8"))
        if DERIVED_QUEUE_SCHEMA.exists():
            cur.execute(DERIVED_QUEUE_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def emit_event(workflow_id: str, run_id: str, phase: str, status: str, detail: dict[str, Any]) -> str:
    with psycopg.connect(STATE_DSN) as conn:
        ensure_state(conn)
        conn.execute(WORKFLOW_SCHEMA.read_text(encoding="utf-8"))
        row = conn.execute(
            """
            INSERT INTO lucidota_control.workflow_event(workflow_id, run_id, phase, status, source, detail)
            VALUES (%s,%s,%s,%s,'korpus_krampii',%s::jsonb)
            RETURNING event_id::text
            """,
            (workflow_id, run_id, phase, status, jdump(detail)),
        ).fetchone()
        conn.commit()
        return str(row[0])


def artifact_tag_for_kind(file_kind: str) -> str:
    if file_kind in {"image", "document", "video", "audio", "text", "binary"}:
        return f"artifact:{file_kind}"
    if file_kind == "code":
        return "artifact:text"
    return "artifact:binary"


def component_tag_for_kind(kind: str) -> str:
    direct = {
        "markdown_section": "component:markdown_section",
        "markdown_document": "component:markdown_section",
        "markdown_code_block": "component:markdown_code_block",
        "python_function": "component:python_function",
        "python_class": "component:python_class",
        "python_method": "component:python_method",
        "code_symbol": "component:code_symbol",
        "code_chunk": "component:code_chunk",
        "code_prelude": "component:code_chunk",
        "text_chunk": "component:text_chunk",
        "python_docstring": "component:text_chunk",
        "python_function_docstring": "component:text_chunk",
        "python_class_docstring": "component:text_chunk",
        "python_method_docstring": "component:text_chunk",
    }
    return direct.get(kind, "component:file")


def insert_file_tag(conn: psycopg.Connection, file_uuid: str, tag_key: str, value: str, source: str, confidence: int = 50) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_korpus.file_tag(file_uuid, tag_key, value, confidence_bps, source, detail)
        VALUES (%s::uuid,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT(file_uuid, tag_key, value, source) DO UPDATE SET confidence_bps=GREATEST(lucidota_korpus.file_tag.confidence_bps, EXCLUDED.confidence_bps), detail=lucidota_korpus.file_tag.detail || EXCLUDED.detail
        """,
        (file_uuid, tag_key, value[:240], confidence, source, jdump({})),
    )


def insert_component_tag(conn: psycopg.Connection, component_uuid: str, tag_key: str, value: str, source: str, confidence: int = 50) -> None:
    conn.execute(
        """
        INSERT INTO lucidota_korpus.component_tag(component_uuid, tag_key, value, confidence_bps, source, detail)
        VALUES (%s::uuid,%s,%s,%s,%s,%s::jsonb)
        ON CONFLICT(component_uuid, tag_key, value, source) DO UPDATE SET confidence_bps=GREATEST(lucidota_korpus.component_tag.confidence_bps, EXCLUDED.confidence_bps), detail=lucidota_korpus.component_tag.detail || EXCLUDED.detail
        """,
        (component_uuid, tag_key, value[:240], confidence, source, jdump({})),
    )


def links_from_component(comp: dict[str, Any]) -> list[dict[str, Any]]:
    text = comp.get("content") or ""
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in re.finditer(r"\[([^\]]{0,240})\]\(([^)\s]{1,1000})\)", text):
        target = m.group(2).strip()
        kind = "url" if target.lower().startswith(("http://", "https://")) else "markdown"
        key = (kind, target, m.group(1))
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": kind, "raw_target": target, "anchor_text": m.group(1), "source": "markdown_link"})
    for m in re.finditer(r"\[\[([^\]|#]{1,500})(?:#[^\]|]+)?(?:\|([^\]]{1,240}))?\]\]", text):
        target = m.group(1).strip()
        anchor = (m.group(2) or target).strip()
        key = ("wikilink", target, anchor)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "wikilink", "raw_target": target, "anchor_text": anchor, "source": "wikilink"})
    for m in re.finditer(r"\bhttps?://[^\s<>'\")]+", text, re.I):
        target = m.group(0).rstrip(".,;")
        key = ("url", target, target)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "url", "raw_target": target, "anchor_text": target[:240], "source": "bare_url"})
    if comp.get("language") == "python":
        for m in re.finditer(r"^\s*(?:from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import|import\s+([A-Za-z_][A-Za-z0-9_.]*))", text, re.M):
            target = next((g for g in m.groups() if g), "")
            if target:
                key = ("import", target, target)
                if key not in seen:
                    seen.add(key)
                    links.append({"link_kind": "import", "raw_target": target, "anchor_text": target, "source": "python_import"})
    return links


def normalized_link_target(source_path: str, raw_target: str, link_kind: str) -> tuple[str, str, str | None]:
    target = raw_target.strip()
    if link_kind == "url" or target.lower().startswith(("http://", "https://")):
        return target, "external", None
    if link_kind == "import":
        return target, "unresolved", None
    cleaned = target.split("#", 1)[0]
    if not cleaned:
        return target, "unresolved", None
    p = Path(cleaned)
    if not p.suffix and link_kind == "wikilink":
        p = p.with_suffix(".md")
    resolved = (Path(source_path).parent / p).resolve()
    return str(resolved), "unresolved", str(resolved)


def insert_file_link(conn: psycopg.Connection, file_uuid: str, component_uuid: str, source_path: str, comp_graph_uuid: str | None, link: dict[str, Any]) -> None:
    normalized, status, resolved_path = normalized_link_target(source_path, link["raw_target"], link["link_kind"])
    target_file_uuid = None
    target_graph_uuid = None
    if resolved_path:
        row = conn.execute(
            """
            SELECT fo.file_uuid::text, fo.graph_item_uuid::text
            FROM lucidota_korpus.file_occurrence occ
            JOIN lucidota_korpus.file_object fo ON fo.file_uuid=occ.file_uuid
            WHERE occ.absolute_path=%s
            ORDER BY occ.created_at DESC
            LIMIT 1
            """,
            (resolved_path,),
        ).fetchone()
        if row:
            target_file_uuid, target_graph_uuid = str(row[0]), str(row[1]) if row[1] else None
            status = "resolved"
    edge_uuid = None
    if target_graph_uuid and comp_graph_uuid:
        edge_uuid = insert_graph_edge(conn, comp_graph_uuid, target_graph_uuid, "COMPONENT_LINKS_TO_FILE", {"raw_target": link["raw_target"], "link_kind": link["link_kind"]}, evidence_uuid=comp_graph_uuid)
    conn.execute(
        """
        INSERT INTO lucidota_korpus.file_link(source_file_uuid, source_component_uuid, raw_target, normalized_target, target_file_uuid, link_kind, anchor_text, status, graph_edge_uuid, detail)
        VALUES (%s::uuid,%s::uuid,%s,%s,%s::uuid,%s,%s,%s,%s::uuid,%s::jsonb)
        ON CONFLICT(source_component_uuid, raw_target, link_kind, anchor_text) DO UPDATE SET
          normalized_target=EXCLUDED.normalized_target, target_file_uuid=EXCLUDED.target_file_uuid,
          status=EXCLUDED.status, graph_edge_uuid=COALESCE(lucidota_korpus.file_link.graph_edge_uuid, EXCLUDED.graph_edge_uuid), detail=EXCLUDED.detail
        """,
        (file_uuid, component_uuid, link["raw_target"], normalized, target_file_uuid, link["link_kind"], link.get("anchor_text", "")[:240], status, edge_uuid, jdump({"source": link.get("source", "")})),
    )


def insert_vibe_tags(conn: psycopg.Connection, component_uuid: str, comp: dict[str, Any]) -> None:
    for key, score, source in vibe_tags_for(comp):
        conn.execute(
            """
            INSERT INTO lucidota_korpus.component_vibe_tag(component_uuid, vibe_tag_key, score, source, detail)
            VALUES (%s::uuid,%s,%s,%s,%s::jsonb)
            ON CONFLICT(component_uuid, vibe_tag_key, source) DO UPDATE SET score=GREATEST(lucidota_korpus.component_vibe_tag.score, EXCLUDED.score), detail=lucidota_korpus.component_vibe_tag.detail || EXCLUDED.detail
            """,
            (component_uuid, key, score, source, jdump({})),
        )


def parse_iso_utc(raw: str) -> dt.datetime | None:
    if not raw:
        return None
    try:
        val = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if val.tzinfo is None:
            val = val.replace(tzinfo=dt.timezone.utc)
        return val.astimezone(dt.timezone.utc)
    except ValueError:
        return parse_loose_datetime(str(raw))


def insert_vibe_telemetry(
    conn: psycopg.Connection,
    batch_uuid: str,
    file_uuid: str,
    component_uuid: str,
    result: FileResult,
    river_decision: dict[str, Any],
) -> None:
    detail = river_decision.get("detail") or {}
    features = river_decision.get("features") or {}
    chrono_raw = (result.detail or {}).get("chrono_original_date") or result.mtime
    original_date = parse_iso_utc(str(chrono_raw))
    assigned = detail.get("dbstream_cluster")
    assigned_cluster: int | None
    try:
        assigned_cluster = int(assigned) if assigned is not None else None
    except (TypeError, ValueError):
        assigned_cluster = None
    conn.execute(
        """
        INSERT INTO lucidota_korpus.vibe_telemetry(
          batch_uuid, file_uuid, component_uuid, model_key, original_file_date,
          original_file_date_source, original_file_date_confidence_bps,
          assigned_cluster, assigned_cluster_label, decision_route, score, anomaly_score,
          vibe_spike, features, stream_vector, detail
        ) VALUES (%s::uuid,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
        ON CONFLICT(component_uuid, model_key) DO UPDATE SET
          batch_uuid=EXCLUDED.batch_uuid,
          file_uuid=EXCLUDED.file_uuid,
          original_file_date=EXCLUDED.original_file_date,
          original_file_date_source=EXCLUDED.original_file_date_source,
          original_file_date_confidence_bps=EXCLUDED.original_file_date_confidence_bps,
          assigned_cluster=EXCLUDED.assigned_cluster,
          assigned_cluster_label=EXCLUDED.assigned_cluster_label,
          decision_route=EXCLUDED.decision_route,
          score=EXCLUDED.score,
          anomaly_score=EXCLUDED.anomaly_score,
          vibe_spike=EXCLUDED.vibe_spike,
          features=EXCLUDED.features,
          stream_vector=EXCLUDED.stream_vector,
          detail=EXCLUDED.detail,
          ingested_at=now()
        """,
        (
            batch_uuid,
            file_uuid,
            component_uuid,
            RIVER_MODEL_KEY,
            original_date,
            str((result.detail or {}).get("chrono_date_source") or "os_mtime"),
            int((result.detail or {}).get("chrono_confidence_bps") or 0),
            assigned_cluster,
            str(detail.get("operator_cluster_hint") or ""),
            str(river_decision.get("decision_route") or ""),
            int(river_decision.get("score") or 0),
            detail.get("anomaly_score"),
            bool(river_decision.get("vibe_spike")),
            jdump(features),
            jdump(detail.get("dbstream_features") or {}),
            jdump(detail),
        ),
    )


def insert_near_duplicates(conn: psycopg.Connection, component_uuid: str, comp: dict[str, Any]) -> int:
    near_dup_mode = os.environ.get(NEAR_DUP_MODE_ENV, "full").strip().lower()
    river_mode = os.environ.get(RIVER_BULK_LIGHT_ENV, "full").strip().lower()
    if near_dup_mode in {"off", "skip", "disabled", "bulk-light"} or river_mode in {"light", "bulk-light", "off", "disabled"}:
        return 0
    sig = comp.get("minhash_signature") or []
    if not sig:
        return 0
    rows = conn.execute(
        """
        SELECT component_uuid::text, minhash_signature
        FROM lucidota_korpus.component
        WHERE component_uuid <> %s::uuid
          AND minhash_signature <> '[]'::jsonb
          AND component_kind=%s
        ORDER BY created_at DESC
        LIMIT 80
        """,
        (component_uuid, comp.get("component_kind", "")),
    ).fetchall()
    inserted = 0
    for other_uuid, other_sig in rows:
        try:
            sim = minhash_similarity([int(x) for x in sig], [int(x) for x in other_sig])
        except (TypeError, ValueError):
            continue
        if sim < 0.86:
            continue
        left, right = sorted([component_uuid, str(other_uuid)])
        conn.execute(
            """
            INSERT INTO lucidota_korpus.near_duplicate(left_component_uuid, right_component_uuid, similarity, algorithm, detail)
            VALUES (%s::uuid,%s::uuid,%s,'minhash64',%s::jsonb)
            ON CONFLICT(left_component_uuid, right_component_uuid, algorithm) DO UPDATE SET similarity=GREATEST(lucidota_korpus.near_duplicate.similarity, EXCLUDED.similarity), detail=lucidota_korpus.near_duplicate.detail || EXCLUDED.detail
            """,
            (left, right, float(sim), jdump({"threshold": 0.86})),
        )
        inserted += 1
    return inserted


def upsert_file_result(conn: psycopg.Connection, batch_uuid: str, result: FileResult, case: dict[str, Any] | None, graph_components: bool, graph_concepts: bool) -> dict[str, int]:
    counts = {"files": 0, "new_files": 0, "duplicates": 0, "components": 0, "entities": 0, "concepts": 0, "errors": 0, "skipped": 0}
    if not result.ok:
        counts["errors"] += 1 if result.error else 0
        counts["skipped"] += 1 if result.skipped_reason else 0
        return counts
    graph_mode = os.environ.get(GRAPH_MODE_ENV, "storage-only").strip().lower()
    graph_enabled = graph_mode not in {"off", "skip", "disabled", "bulk-light", "storage-only"}
    graph_components = graph_components and graph_enabled
    graph_concepts = graph_concepts and graph_enabled
    pre = conn.execute("SELECT file_uuid::text, graph_item_uuid::text FROM lucidota_korpus.file_object WHERE sha256=%s", (result.sha256,)).fetchone()
    is_duplicate = pre is not None
    graph_uuid = pre[1] if pre and pre[1] else None
    if graph_enabled and not graph_uuid:
        graph_uuid = insert_graph_item(conn, "EVIDENCE", f"Korpus file {Path(result.path).name} [{result.sha256[:12]}]", f"korpus://sha256/{result.sha256}", {"kind":"korpus_file","sha256":result.sha256,"path":result.path,"file_kind":result.file_kind,"mime":result.mime,"evidence_note":"File hashed, UUIDed, locked in CAS, and indexed by KORPUS KRAMPII."}, layer="digital_world", role="korpus_file")
    row = conn.execute(
        """
        INSERT INTO lucidota_korpus.file_object(
          sha256, size_bytes, mime, file_kind, status, deferred_reason, suffix, cas_uri, locked_relative_path,
          first_seen_path, minhash_signature, graph_item_uuid, detail
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::uuid,%s::jsonb)
        ON CONFLICT(sha256) DO UPDATE SET
          last_seen_at=now(), seen_count=lucidota_korpus.file_object.seen_count+1,
          mime=EXCLUDED.mime, file_kind=EXCLUDED.file_kind, suffix=EXCLUDED.suffix,
          status=CASE WHEN lucidota_korpus.file_object.status='indexed' THEN 'indexed' ELSE EXCLUDED.status END,
          deferred_reason=CASE WHEN EXCLUDED.deferred_reason <> '' THEN EXCLUDED.deferred_reason ELSE lucidota_korpus.file_object.deferred_reason END,
          cas_uri=EXCLUDED.cas_uri, locked_relative_path=EXCLUDED.locked_relative_path,
          minhash_signature=CASE WHEN EXCLUDED.minhash_signature <> '[]'::jsonb THEN EXCLUDED.minhash_signature ELSE lucidota_korpus.file_object.minhash_signature END,
          graph_item_uuid=COALESCE(lucidota_korpus.file_object.graph_item_uuid, EXCLUDED.graph_item_uuid),
          detail=lucidota_korpus.file_object.detail || EXCLUDED.detail,
          updated_at=now()
        RETURNING file_uuid::text
        """,
        (result.sha256, result.size_bytes, result.mime, result.file_kind, (result.detail or {}).get("parser_status", "indexed"), (result.detail or {}).get("deferred_reason", ""), result.suffix, result.cas_uri, result.locked_relative_path, result.path, jdump(result.file_minhash or []), graph_uuid, jdump(result.detail or {})),
    ).fetchone()
    file_uuid = str(row[0])
    result.file_uuid = file_uuid
    insert_file_tag(conn, file_uuid, artifact_tag_for_kind(result.file_kind), result.file_kind, "korpus:file_kind", 50)
    insert_file_tag(conn, file_uuid, "case:associated" if case else "case:unassigned", case["case_key"] if case else "", "korpus:case_state", 50)
    if result.file_kind == "code":
        insert_file_tag(conn, file_uuid, "processing:algorithm", result.suffix or result.file_kind, "korpus:code_file", 50)
    conn.execute(
        """
        INSERT INTO lucidota_vault.cas_manifest(sha256, cas_uri, relative_path, size_bytes, mime, source, last_seen_at)
        VALUES (%s,%s,%s,%s,%s,'korpus_krampii',now())
        ON CONFLICT(sha256) DO UPDATE SET relative_path=EXCLUDED.relative_path, size_bytes=EXCLUDED.size_bytes, mime=EXCLUDED.mime, source=EXCLUDED.source, last_seen_at=now()
        """,
        (result.sha256, result.cas_uri, result.locked_relative_path, result.size_bytes, result.mime),
    )
    conn.execute(
        """
        INSERT INTO lucidota_korpus.file_occurrence(batch_uuid, file_uuid, absolute_path, relative_path, root_path, mtime, status, detail)
        VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s::timestamptz,%s,%s::jsonb)
        ON CONFLICT(batch_uuid, absolute_path) DO UPDATE SET status=EXCLUDED.status, detail=EXCLUDED.detail
        """,
        (batch_uuid, file_uuid, result.path, result.relative_path, result.root, result.mtime, "duplicate" if is_duplicate else ("deferred" if (result.detail or {}).get("parser_status") == "deferred" else "indexed"), jdump({"sha256": result.sha256, "parser_status": (result.detail or {}).get("parser_status", "indexed")})),
    )
    counts["files"] += 1
    counts["duplicates" if is_duplicate else "new_files"] += 1
    if case and graph_enabled and graph_uuid:
        insert_graph_edge(conn, case["graph_item_uuid"], graph_uuid, "CASE_HAS_KORPUS_FILE", {"case_key": case["case_key"], "sha256": result.sha256}, evidence_uuid=graph_uuid)
    comp_count = ent_count = concept_count = 0
    for comp in result.components or []:
        row = conn.execute(
            """
            INSERT INTO lucidota_korpus.component(
              file_uuid, component_index, component_kind, language, title, symbol,
              start_line, end_line, sha256, token_count, content, go_terms, minhash_signature, entropy, graph_item_uuid, detail
            ) VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::uuid,%s::jsonb)
            ON CONFLICT(file_uuid, component_index) DO UPDATE SET
              component_kind=EXCLUDED.component_kind, language=EXCLUDED.language, title=EXCLUDED.title,
              symbol=EXCLUDED.symbol, start_line=EXCLUDED.start_line, end_line=EXCLUDED.end_line,
              sha256=EXCLUDED.sha256, token_count=EXCLUDED.token_count, content=EXCLUDED.content,
              go_terms=EXCLUDED.go_terms, minhash_signature=EXCLUDED.minhash_signature, entropy=EXCLUDED.entropy,
              graph_item_uuid=COALESCE(lucidota_korpus.component.graph_item_uuid, EXCLUDED.graph_item_uuid), detail=EXCLUDED.detail
            RETURNING component_uuid::text, graph_item_uuid::text
            """,
            (file_uuid, comp["component_index"], comp["component_kind"], comp["language"], comp["title"], comp["symbol"], comp["start_line"], comp["end_line"], comp["sha256"], comp["token_count"], comp["content"], jdump(comp.get("go_terms", [])), jdump(comp.get("minhash_signature", [])), comp.get("entropy", 0.0), None, jdump({"source_path": result.path})),
        ).fetchone()
        component_uuid = str(row[0])
        comp["component_uuid"] = component_uuid
        comp_graph_uuid = str(row[1]) if row and row[1] else None
        if graph_components and not comp_graph_uuid:
            comp_graph_uuid = insert_graph_item(
                conn,
                (comp.get("go_terms") or ["COMMENT"])[0],
                f"{Path(result.path).name}:{comp['component_kind']}:{comp.get('title') or comp.get('symbol') or comp['component_index']}",
                f"korpus-component://{component_uuid}",
                {"kind":"korpus_component","component_uuid":component_uuid,"file_sha256":result.sha256,"component_sha256":comp["sha256"],"go_terms":comp.get("go_terms",[]),"evidence_note":"Component extracted deterministically from a KORPUS file."},
                layer="map",
                role="korpus_component",
            )
            conn.execute("UPDATE lucidota_korpus.component SET graph_item_uuid=%s::uuid WHERE component_uuid=%s::uuid", (comp_graph_uuid, component_uuid))
            insert_graph_edge(conn, graph_uuid, comp_graph_uuid, "FILE_HAS_COMPONENT", {"component_kind": comp["component_kind"], "index": comp["component_index"]}, evidence_uuid=graph_uuid)
        comp_count += 1
        insert_component_tag(conn, component_uuid, component_tag_for_kind(comp["component_kind"]), comp.get("title") or comp.get("symbol") or comp["component_kind"], "korpus:component_kind", 50)
        if comp["component_kind"].startswith("python_"):
            insert_component_tag(conn, component_uuid, "processing:ast", comp.get("symbol") or comp["component_kind"], "python:ast", 69)
        if comp["component_kind"] in {"python_function", "python_method", "code_symbol", "markdown_code_block"}:
            insert_component_tag(conn, component_uuid, "processing:algorithm", comp.get("symbol") or comp.get("language") or comp["component_kind"], "korpus:algorithm", 69)
        for ent in comp.get("entities", []):
            ent_graph = None
            if graph_concepts and should_promote_entity_to_graph(ent):
                term = entity_graph_term(ent)
                ent_graph = insert_graph_item(conn, term, f"{ent['entity_kind']}: {ent['value']}", entity_graph_location(ent), entity_graph_payload(ent, result.sha256), layer="map", role="korpus_entity")
                insert_graph_edge(conn, comp_graph_uuid or graph_uuid, ent_graph, "COMPONENT_MENTIONS_ENTITY", {"entity_kind": ent["entity_kind"], "source": ent["source"]}, evidence_uuid=graph_uuid)
            conn.execute(
                """
                INSERT INTO lucidota_korpus.entity(component_uuid, file_uuid, entity_kind, value, normalized_value, confidence_bps, source, context, graph_item_uuid, detail)
                VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s::uuid,%s::jsonb)
                ON CONFLICT(component_uuid, entity_kind, normalized_value, source) DO UPDATE SET value=EXCLUDED.value, confidence_bps=EXCLUDED.confidence_bps, context=EXCLUDED.context, graph_item_uuid=COALESCE(lucidota_korpus.entity.graph_item_uuid, EXCLUDED.graph_item_uuid), detail=EXCLUDED.detail
                """,
                (component_uuid, file_uuid, ent["entity_kind"], ent["value"], ent["normalized_value"], ent["confidence_bps"], ent["source"], ent["context"], ent_graph, jdump(ent.get("detail", {}))),
            )
            tag_key = f"entity:{ent['entity_kind']}"
            if tag_key in {"entity:name", "entity:alias", "entity:phone", "entity:ip", "entity:email", "entity:url", "entity:domain", "entity:address", "entity:date", "entity:identifier"}:
                insert_component_tag(conn, component_uuid, tag_key, ent["normalized_value"], "korpus:entity", ent.get("confidence_bps", 50))
            ent_count += 1
        for link in links_from_component(comp):
            insert_file_link(conn, file_uuid, component_uuid, result.path, comp_graph_uuid, link)
        insert_vibe_tags(conn, component_uuid, comp)
        insert_near_duplicates(conn, component_uuid, comp)
        river_decision = river_decide_component(conn, batch_uuid, file_uuid, component_uuid, comp)
        comp["river_decision"] = river_decision
        insert_vibe_telemetry(conn, batch_uuid, file_uuid, component_uuid, result, river_decision)
        if comp.get("content"):
            conn.execute(
                """
                INSERT INTO lucidota_korpus.embedding_queue(component_uuid, status, embedding_model)
                VALUES (%s::uuid,'queued',%s)
                ON CONFLICT(component_uuid, embedding_model) DO NOTHING
                """,
                (component_uuid, EMBED_MODEL),
            )
            conn.execute("UPDATE lucidota_korpus.component SET embedding_status='queued', embedding_model=%s WHERE component_uuid=%s::uuid AND embedding_status IN ('pending','failed')", (EMBED_MODEL, component_uuid))
        for concept in comp.get("concepts", []):
            concept_graph = None
            if graph_concepts:
                concept_graph = insert_graph_item(conn, concept.get("go_term") or "COMMENT", f"{concept['concept_kind']}: {concept['value']}", f"korpus-concept://{concept['concept_kind']}/{hashlib.sha256(concept['normalized_value'].encode()).hexdigest()[:16]}", {"kind":"korpus_concept","concept":concept,"file_sha256":result.sha256,"evidence_note":"Concept extracted deterministically from KORPUS component."}, layer="map", role="korpus_concept")
                insert_graph_edge(conn, comp_graph_uuid or graph_uuid, concept_graph, "COMPONENT_HAS_CONCEPT", {"concept_kind": concept["concept_kind"], "source": concept["source"]}, evidence_uuid=graph_uuid)
            conn.execute(
                """
                INSERT INTO lucidota_korpus.concept(component_uuid, file_uuid, concept_kind, value, normalized_value, go_term, source, confidence_bps, graph_item_uuid, detail)
                VALUES (%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s::uuid,%s::jsonb)
                ON CONFLICT(component_uuid, concept_kind, normalized_value, source) DO UPDATE SET value=EXCLUDED.value, go_term=EXCLUDED.go_term, confidence_bps=EXCLUDED.confidence_bps, graph_item_uuid=COALESCE(lucidota_korpus.concept.graph_item_uuid, EXCLUDED.graph_item_uuid), detail=EXCLUDED.detail
                """,
                (component_uuid, file_uuid, concept["concept_kind"], concept["value"], concept["normalized_value"], concept.get("go_term", ""), concept["source"], concept["confidence_bps"], concept_graph, jdump(concept.get("detail", {}))),
            )
            concept_count += 1
    conn.execute("UPDATE lucidota_korpus.file_object SET component_count=%s, entity_count=%s, concept_count=%s, updated_at=now() WHERE file_uuid=%s::uuid", (comp_count, ent_count, concept_count, file_uuid))
    counts["components"] += comp_count
    counts["entities"] += ent_count
    counts["concepts"] += concept_count
    return counts


def algo_role(path: Path, summary: str) -> list[str]:
    text = f"{path.name} {summary}".lower()
    roles = []
    for key, role in [
        ("minhash", "dedupe/near-duplicate detection"), ("dedupe", "dedupe"), ("sketch", "bulk indexing/cardinality"),
        ("entropy", "priority/noise scoring"), ("pheromone", "scheduler prioritization"), ("bandit", "adaptive routing"),
        ("physarum", "network flow"), ("temporal", "timeline motif detection"), ("label", "weak supervision tagging"),
        ("privacy", "redaction/privacy"), ("gini", "inequality/gap analysis"), ("cockpit", "KPI/honesty metrics"),
        ("hoeffding", "stream split/gate"), ("serpentina", "recovery/self-righting"), ("honeybee", "fleet store/rate control"),
    ]:
        if key in text:
            roles.append(role)
    return roles or ["general deterministic algorithm"]


def cmd_algos_audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        for path in sorted((ROOT / "ALGOS").glob("*.py")):
            if path.name == "__init__.py":
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'"""(.*?)"""', text, re.S)
            summary = normalize_ws(m.group(1)) if m else ""
            funcs = re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.M)
            key = path.stem.replace("_", "-")
            digest = sha256_file(path)
            detail = {"functions": funcs, "roles": algo_role(path, summary)}
            conn.execute(
                """
                INSERT INTO lucidota_korpus.algo_inventory(algo_key, relative_path, module_name, title, summary, sha256, status, detail)
                VALUES (%s,%s,%s,%s,%s,%s,'available',%s::jsonb)
                ON CONFLICT(algo_key) DO UPDATE SET relative_path=EXCLUDED.relative_path, module_name=EXCLUDED.module_name, title=EXCLUDED.title, summary=EXCLUDED.summary, sha256=EXCLUDED.sha256, status='available', detail=EXCLUDED.detail, updated_at=now()
                """,
                (key, rel(path), f"ALGOS.{path.stem}", path.stem, summary, digest, jdump(detail)),
            )
            rows.append({"algo_key": key, "path": rel(path), "summary": summary, **detail})
        conn.commit()
    event_id = emit_event("korpus-krampii-algos-audit", "algos", "audit", "succeeded", {"algorithm_count": len(rows), "roles": sorted({r for row in rows for r in row["roles"]})})
    return {"ok": True, "algorithm_count": len(rows), "algorithms": rows, "workflow_event": event_id}


def cmd_ingest(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "river_mode", ""):
        os.environ[RIVER_BULK_LIGHT_ENV] = args.river_mode
    if getattr(args, "near_dup_mode", ""):
        os.environ[NEAR_DUP_MODE_ENV] = args.near_dup_mode
    if getattr(args, "graph_mode", ""):
        os.environ[GRAPH_MODE_ENV] = args.graph_mode
    if os.environ.get(GRAPH_MODE_ENV, "storage-only").strip().lower() == "full" and os.environ.get(LEGACY_DIRECT_GRAPH_ENV) != "1":
        return {
            "ok": False,
            "error": "legacy_direct_graph_ingest_blocked",
            "graph_mode_requested": "full",
            "required_override": LEGACY_DIRECT_GRAPH_ENV + "=1",
            "safe_default": "storage-only",
            "reason": "KORPUS ingestion must preserve custody/components and route graph candidates through graph promotion, not legacy direct graph writes.",
        }
    roots = [Path(p).expanduser() for p in (args.roots or [str(DROP_ROOT)])]
    if not args.roots:
        DROP_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    label = args.label or "korpus-krampii-bulk"
    options = {
        "workers": args.workers,
        "max_file_mb": args.max_file_mb,
        "max_text_mb": args.max_text_mb,
        "deep_media": args.deep_media,
        "graph_components": not args.no_graph_components,
        "graph_concepts": not args.no_graph_concepts,
        "chronological": bool(args.chronological),
        "chronological_date_source": "st_birthtime_else_st_mtime",
        "river_mode": os.environ.get(RIVER_BULK_LIGHT_ENV, "full"),
        "near_dup_mode": os.environ.get(NEAR_DUP_MODE_ENV, "full"),
        "graph_mode": os.environ.get(GRAPH_MODE_ENV, "storage-only"),
        "skip_existing_paths": bool(args.skip_existing_paths),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_ROOT / f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.korpus.results.jsonl"
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        case = ensure_case(conn, args.case, args.case_title or args.case, "KORPUS KRAMPII mass-ingestion case.") if args.case else None
        batch_uuid = str(conn.execute(
            """
            INSERT INTO lucidota_korpus.ingest_batch(batch_label, case_uuid, status, root_paths, options)
            VALUES (%s,%s::uuid,'running',%s::jsonb,%s::jsonb)
            RETURNING batch_uuid::text
            """,
            (label, case["case_uuid"] if case else None, jdump([str(r) for r in roots]), jdump(options)),
        ).fetchone()[0])
        conn.commit()
    files = list(iter_files(roots, args.include_hidden, args.no_default_excludes))
    if args.chronological:
        files.sort(key=chronological_key)
    if args.limit:
        files = files[: args.limit]
    observed_paths = len(files)
    skipped_existing_paths = 0
    if args.skip_existing_paths and files:
        paths = [str(path) for _root, path in files]
        existing: set[str] = set()
        with psycopg.connect(STORAGE_DSN, connect_timeout=3) as conn:
            ensure_storage(conn)
            for start in range(0, len(paths), 1000):
                chunk = paths[start : start + 1000]
                for (absolute_path,) in conn.execute(
                    "SELECT DISTINCT absolute_path FROM lucidota_korpus.file_occurrence WHERE absolute_path = ANY(%s)",
                    (chunk,),
                ):
                    existing.add(str(absolute_path))
        if existing:
            files = [(root, path) for root, path in files if str(path) not in existing]
            skipped_existing_paths = len(existing)
    total_counts = {"files": 0, "new_files": 0, "duplicates": 0, "components": 0, "entities": 0, "concepts": 0, "errors": 0, "skipped": 0}
    with psycopg.connect(STORAGE_DSN) as conn, results_path.open("w", encoding="utf-8") as out:
        ensure_storage(conn)
        case = ensure_case(conn, args.case, args.case_title or args.case, "KORPUS KRAMPII mass-ingestion case.") if args.case else None
        def persist_result(result: FileResult) -> None:
            transient = (psycopg.errors.DeadlockDetected, psycopg.errors.SerializationFailure, psycopg.errors.LockNotAvailable)
            last_error: Exception | None = None
            for attempt in range(1, 4):
                conn.execute("SAVEPOINT korpus_file")
                try:
                    counts = upsert_file_result(conn, batch_uuid, result, case, not args.no_graph_components, not args.no_graph_concepts)
                    conn.execute("RELEASE SAVEPOINT korpus_file")
                    out.write(jdump(asdict(result)) + "\n")
                    for k, v in counts.items():
                        total_counts[k] += v
                    return
                except transient as exc:
                    last_error = exc
                    conn.execute("ROLLBACK TO SAVEPOINT korpus_file")
                    conn.execute("RELEASE SAVEPOINT korpus_file")
                    time.sleep(0.25 * attempt)
                except Exception as exc:
                    last_error = exc
                    conn.execute("ROLLBACK TO SAVEPOINT korpus_file")
                    conn.execute("RELEASE SAVEPOINT korpus_file")
                    break
            total_counts["errors"] += 1
            failed = asdict(result)
            failed.update({"ok": False, "error": f"db_persist_failed: {last_error}"})
            out.write(jdump(failed) + "\n")
        if args.chronological:
            # River/DBSTREAM learns order-sensitive concept drift; commit each file in lived order.
            for idx, (root, path) in enumerate(files, 1):
                result = process_file(root, path, args.max_file_mb, args.max_text_mb, args.deep_media)
                persist_result(result)
                if idx % args.commit_every == 0:
                    conn.commit()
        else:
            with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futs = [pool.submit(process_file, root, path, args.max_file_mb, args.max_text_mb, args.deep_media) for root, path in files]
                for idx, fut in enumerate(cf.as_completed(futs), 1):
                    try:
                        result = fut.result()
                    except Exception as exc:
                        total_counts["errors"] += 1
                        out.write(jdump({"ok": False, "error": f"worker_failed: {exc}"}) + "\n")
                        continue
                    persist_result(result)
                    if idx % args.commit_every == 0:
                        conn.commit()
        conn.execute(
            """
            UPDATE lucidota_korpus.ingest_batch SET
              status='succeeded', file_count=%s, new_file_count=%s, duplicate_file_count=%s,
              component_count=%s, concept_count=%s, entity_count=%s, skipped_count=%s, error_count=%s,
              detail=detail || %s::jsonb, finished_at=now()
            WHERE batch_uuid=%s::uuid
            """,
            (total_counts["files"], total_counts["new_files"], total_counts["duplicates"], total_counts["components"], total_counts["concepts"], total_counts["entities"], total_counts["skipped"], total_counts["errors"], jdump({"results_path": rel(results_path), "elapsed_seconds": round(time.time() - started, 3), "observed_paths": observed_paths, "paths_after_resume_skip": len(files), "skipped_existing_paths": skipped_existing_paths, "river_mode": os.environ.get(RIVER_BULK_LIGHT_ENV, "full"), "near_dup_mode": os.environ.get(NEAR_DUP_MODE_ENV, "full"), "graph_mode": os.environ.get(GRAPH_MODE_ENV, "storage-only")}), batch_uuid),
        )
        conn.commit()
    detail = {"batch_uuid": batch_uuid, "label": label, "roots": [str(r) for r in roots], "observed_paths": observed_paths, "paths_after_resume_skip": len(files), "skipped_existing_paths": skipped_existing_paths, "river_mode": os.environ.get(RIVER_BULK_LIGHT_ENV, "full"), "near_dup_mode": os.environ.get(NEAR_DUP_MODE_ENV, "full"), "graph_mode": os.environ.get(GRAPH_MODE_ENV, "storage-only"), "counts": total_counts, "results_path": rel(results_path), "elapsed_seconds": round(time.time() - started, 3), "software": "KORPUS KRAMPII", "llm_calls": 0}
    event_id = emit_event("korpus-krampii-mass-ingest", batch_uuid, "mass_ingest", "succeeded", detail)
    return {"ok": True, **detail, "workflow_event": event_id}


def cmd_semantic_diff(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        source_component = ""
        query_text = args.text or ""
        if args.component_uuid:
            row = conn.execute("SELECT component_uuid::text, content, title FROM lucidota_korpus.component WHERE component_uuid=%s::uuid", (args.component_uuid,)).fetchone()
            if not row:
                raise SystemExit(f"component not found: {args.component_uuid}")
            source_component = str(row[0])
            query_text = query_text or str(row[1] or row[2] or "")
        if not query_text.strip():
            raise SystemExit("semantic-diff requires --text or --component-uuid")
        params: list[Any] = [vector_literal(query_text), vector_literal(query_text)]
        filters = ["c.embedding IS NOT NULL"]
        if source_component:
            filters.append("c.component_uuid <> %s::uuid")
            params.append(source_component)
        if args.from_date:
            filters.append("fo.first_seen_at >= %s::timestamptz")
            params.append(args.from_date)
        if args.to_date:
            filters.append("fo.first_seen_at <= %s::timestamptz")
            params.append(args.to_date)
        params.append(args.limit)
        sql = f"""
            SELECT c.component_uuid::text, fo.file_uuid::text, fo.first_seen_path, fo.first_seen_at::text,
                   c.component_kind, c.title, c.symbol, c.start_line, left(c.content, 500) AS excerpt,
                   (c.embedding <=> %s::vector) AS distance
            FROM lucidota_korpus.component c
            JOIN lucidota_korpus.file_object fo ON fo.file_uuid=c.file_uuid
            WHERE {' AND '.join(filters)}
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        """
        rows = conn.execute(sql, params).fetchall()
    keys = ["component_uuid","file_uuid","path","first_seen_at","component_kind","title","symbol","start_line","excerpt","distance"]
    return {"ok": True, "source_component_uuid": source_component, "query_preview": query_text[:240], "matches": [dict(zip(keys, r)) for r in rows]}


def cmd_embed_pending(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    embedded = failed = 0
    rows_out: list[dict[str, Any]] = []
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        rows = conn.execute(
            """
            SELECT c.component_uuid::text, left(c.content, %s) AS content, q.embedding_job_uuid::text
            FROM lucidota_korpus.component c
            LEFT JOIN lucidota_korpus.embedding_queue q ON q.component_uuid=c.component_uuid AND q.embedding_model=%s
            WHERE (q.status IN ('queued','failed') OR (q.embedding_job_uuid IS NULL AND c.embedding_status IN ('pending','queued','failed')))
              AND c.content <> ''
            ORDER BY coalesce(q.created_at, c.created_at)
            LIMIT %s
            """,
            (args.max_chars, EMBED_MODEL, args.limit),
        ).fetchall()
        for component_uuid, content, job_uuid in rows:
            try:
                if job_uuid is None:
                    job_uuid = str(conn.execute(
                        """
                        INSERT INTO lucidota_korpus.embedding_queue(component_uuid, status, embedding_model)
                        VALUES (%s::uuid,'queued',%s)
                        ON CONFLICT(component_uuid, embedding_model) DO UPDATE SET status='queued'
                        RETURNING embedding_job_uuid::text
                        """,
                        (component_uuid, EMBED_MODEL),
                    ).fetchone()[0])
                conn.execute("UPDATE lucidota_korpus.embedding_queue SET status='running', attempts=attempts+1, last_error='' WHERE embedding_job_uuid=%s::uuid", (job_uuid,))
                conn.execute(
                    """
                    UPDATE lucidota_korpus.component
                    SET embedding=%s::vector, embedding_status='embedded', embedding_model=%s, embedded_at=now()
                    WHERE component_uuid=%s::uuid
                    """,
                    (vector_literal(content or ""), EMBED_MODEL, component_uuid),
                )
                conn.execute("UPDATE lucidota_korpus.embedding_queue SET status='succeeded', last_error='' WHERE embedding_job_uuid=%s::uuid", (job_uuid,))
                embedded += 1
                rows_out.append({"component_uuid": component_uuid, "status": "embedded"})
            except Exception as exc:
                failed += 1
                conn.execute("UPDATE lucidota_korpus.embedding_queue SET status='failed', last_error=%s WHERE embedding_job_uuid=%s::uuid", (str(exc)[:500], job_uuid))
                conn.execute("UPDATE lucidota_korpus.component SET embedding_status='failed' WHERE component_uuid=%s::uuid", (component_uuid,))
                rows_out.append({"component_uuid": component_uuid, "status": "failed", "error": str(exc)[:200]})
        conn.commit()
    detail = {"embedded": embedded, "failed": failed, "model": EMBED_MODEL, "elapsed_seconds": round(time.time() - started, 3), "rows": rows_out[:20]}
    event_id = emit_event("korpus-krampii-embed-pending", "embed-pending", "embed", "succeeded" if failed == 0 else "failed", detail)
    return {"ok": failed == 0, **detail, "workflow_event": event_id}


def cmd_deferred(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        rows = conn.execute(
            """
            SELECT file_uuid::text, sha256, mime, file_kind, deferred_reason, first_seen_path, seen_count, last_seen_at::text
            FROM lucidota_korpus.file_object
            WHERE status='deferred'
            ORDER BY last_seen_at DESC
            LIMIT %s
            """,
            (args.limit,),
        ).fetchall()
    return {"ok": True, "deferred_count_returned": len(rows), "deferred": [dict(zip(["file_uuid","sha256","mime","file_kind","deferred_reason","first_seen_path","seen_count","last_seen_at"], r)) for r in rows]}


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    with psycopg.connect(STORAGE_DSN) as conn:
        ensure_storage(conn)
        batch = conn.execute("SELECT batch_uuid::text, batch_label, status, file_count, component_count, entity_count, concept_count, started_at::text, finished_at::text, detail FROM lucidota_korpus.ingest_batch ORDER BY started_at DESC LIMIT 1").fetchone()
        counts = conn.execute("SELECT (SELECT count(*) FROM lucidota_korpus.file_object), (SELECT count(*) FROM lucidota_korpus.component), (SELECT count(*) FROM lucidota_korpus.entity), (SELECT count(*) FROM lucidota_korpus.concept), (SELECT count(*) FROM lucidota_korpus.algo_inventory), (SELECT count(*) FROM lucidota_korpus.file_object WHERE status='deferred'), (SELECT count(*) FROM lucidota_korpus.component WHERE embedding_status='queued'), (SELECT count(*) FROM lucidota_korpus.component WHERE embedding_status='embedded'), (SELECT count(*) FROM lucidota_korpus.river_decision WHERE vibe_spike), (SELECT count(*) FROM lucidota_korpus.vibe_telemetry)").fetchone()
    return {"ok": True, "latest_batch": dict(zip(["batch_uuid","label","status","files","components","entities","concepts","started_at","finished_at","detail"], batch)) if batch else None, "totals": dict(zip(["files","components","entities","concepts","algorithms","deferred_files","embedding_queued","embedding_embedded","river_vibe_spikes","vibe_telemetry_points"], counts))}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="korpus-krampii")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("algos-audit")
    p.set_defaults(func=cmd_algos_audit)
    p = sub.add_parser("ingest")
    p.add_argument("roots", nargs="*")
    p.add_argument("--case", default="")
    p.add_argument("--case-title", default="")
    p.add_argument("--label", default="")
    p.add_argument("--workers", type=int, default=max(2, min(8, (os.cpu_count() or 4))))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-file-mb", type=int, default=64)
    p.add_argument("--max-text-mb", type=int, default=8)
    p.add_argument("--commit-every", type=int, default=25)
    p.add_argument("--chronological", action="store_true", help="Sort by extracted frontmatter/path/content date first, OS time only as fallback, and upsert sequentially so River/DBSTREAM learns lived order.")
    p.add_argument("--include-hidden", action="store_true")
    p.add_argument("--no-default-excludes", action="store_true")
    p.add_argument("--deep-media", action="store_true", help="Queue media stubs in mass path; use artifact-ingest for full OCR/EXIF/media deep run.")
    p.add_argument("--no-graph-components", action="store_true", help="Skip GO graph items for components; file/entity/concept tables still persist.")
    p.add_argument("--no-graph-concepts", action="store_true")
    p.add_argument("--river-mode", choices=["full", "light", "bulk-light", "off", "disabled"], default=os.environ.get(RIVER_BULK_LIGHT_ENV, "full"), help="Use light/off during giant bulk ingest to bypass heavyweight pickled River state.")
    p.add_argument("--near-dup-mode", choices=["full", "off", "skip", "disabled", "bulk-light"], default=os.environ.get(NEAR_DUP_MODE_ENV, "full"), help="Use off/skip during giant bulk ingest; near-duplicate compute can be replayed later.")
    p.add_argument("--graph-mode", choices=["full", "off", "skip", "disabled", "bulk-light", "storage-only"], default=os.environ.get(GRAPH_MODE_ENV, "storage-only"), help="Default is storage-only. Legacy full direct graph writes require LUCIDOTA_KORPUS_ALLOW_LEGACY_DIRECT_GRAPH=1; graph promotion replay is the canonical path.")
    p.add_argument("--skip-existing-paths", action="store_true", help="Resume mode: skip paths already present in lucidota_korpus.file_occurrence.")
    p.set_defaults(func=cmd_ingest)
    p = sub.add_parser("semantic-diff")
    p.add_argument("--component-uuid", default="")
    p.add_argument("--text", default="")
    p.add_argument("--from-date", default="")
    p.add_argument("--to-date", default="")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_semantic_diff)
    p = sub.add_parser("embed-pending")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-chars", type=int, default=200000)
    p.set_defaults(func=cmd_embed_pending)
    p = sub.add_parser("deferred")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_deferred)
    p = sub.add_parser("status")
    p.set_defaults(func=cmd_status)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    result = args.func(args)
    print(jdump(result) if args.json else json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
