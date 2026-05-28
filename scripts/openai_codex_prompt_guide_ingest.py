#!/usr/bin/env python3
"""Ingest OpenAI's Codex prompting guide as an INDY_READs prompt-policy source."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.indy_book_learning_pipeline import run_pipeline  # noqa: E402

OFFICIAL_URL = "https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide"
RAW_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/gpt-5/codex_prompting_guide.ipynb"
SOURCE_DIR = ROOT / "04_RUNTIME" / "indy_reads_sources"
POLICY_JSON = ROOT / "00_PROJECT_BRAIN" / "CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json"
POLICY_MD = ROOT / "00_PROJECT_BRAIN" / "CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.md"
OUT = ROOT / "05_OUTPUTS" / "indy_reads"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def extract_markdown(raw: str) -> str:
    try:
        data = json.loads(raw)
    except Exception:
        return raw
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list):
        return raw
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source") or []
        parts.append("".join(str(x) for x in src) if isinstance(src, list) else str(src))
    return "\n\n".join(parts).strip() + "\n"


def prompt_laws(markdown: str) -> list[dict[str, Any]]:
    low = markdown.lower()
    candidates = [
        ("prefer_fast_search", "Prefer rg/rg --files for search before slower shell scans.", ("prefer using `rg`", "prefer rg", "rg --files")),
        ("prefer_dedicated_tools", "Use dedicated tools before raw shell when the tool exists.", ("prefer to use the tool", "dedicated tool")),
        ("parallelize_independent_reads", "Batch and parallelize independent reads/searches/tool calls.", ("parallel", "batch everything")),
        ("persist_to_verified_work", "Gather context, implement, test, and refine end-to-end when feasible.", ("persist until", "fully handled end-to-end")),
        ("suppress_upfront_preamble", "Do not force upfront plans, preambles, or status chatter that can stop rollout.", ("upfront plan", "preambles")),
        ("working_code_not_plan", "Default to working code over plan-only output.", ("deliver working code", "not just a plan")),
        ("root_cause_not_symptom", "Cover the root cause/core ask, not only a narrow symptom.", ("root cause", "not just a symptom")),
        ("preserve_dirty_worktree", "Never revert unrelated dirty worktree changes.", ("dirty git worktree", "never revert")),
        ("use_apply_patch_for_edits", "Use apply_patch for focused source edits when practical.", ("apply_patch", "patch")),
        ("plan_hygiene", "Use update_plan for multi-step work and keep it current.", ("update_plan", "plan tool")),
    ]
    out = []
    for ident, rule, needles in candidates:
        if any(n in low for n in needles):
            out.append({"id": ident, "rule": rule, "source": "openai_codex_prompting_guide"})
    return out


def build_policy(markdown: str, *, source_url: str = OFFICIAL_URL) -> dict[str, Any]:
    laws = prompt_laws(markdown)
    return {
        "schema": "lucidota.prompting.codex_guide_policy.v1",
        "generated_at": now(),
        "source_url": source_url,
        "raw_source_url": RAW_URL,
        "source_sha256": sha_text(markdown),
        "policy_status": "active_staged",
        "scope": ["project2501_admin_prompt", "goal_agent_packets", "local_model_prompts", "groq_cohere_prompts", "indy_reads"],
        "laws": laws,
        "router_features": {
            "prefer_codex_for": ["large_codebase_edits", "apply_patch_heavy_rollouts", "long_horizon_refactors"],
            "prefer_deterministic_for": ["routing", "ledger_checks", "schema_validation", "receipt_admission"],
            "prefer_other_models_for": ["independent_audit", "comparison", "cheap_synthesis"],
        },
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fetch_raw(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"user-agent": "lucidota-indy-reads/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def run(args: argparse.Namespace) -> dict[str, Any]:
    raw = Path(args.source_file).read_text(encoding="utf-8") if args.source_file else fetch_raw(args.raw_url, args.timeout_sec)
    markdown = extract_markdown(raw)
    source_path = SOURCE_DIR / "openai_codex_prompting_guide.md"
    write_text(source_path, markdown)
    policy = build_policy(markdown, source_url=args.official_url)
    write_json(POLICY_JSON, policy)
    write_text(POLICY_MD, "# Codex Prompting Guide - LUCIDOTA Policy\n\n" + "\n".join(f"- {x['id']}: {x['rule']}" for x in policy["laws"]) + "\n")
    child = run_pipeline(Namespace(source_file=str(source_path), source_url=None, annas_url=None, annas_query=None, title="OpenAI Codex Prompting Guide", author="OpenAI", max_tokens=500, overlap_tokens=25, timeout_sec=args.timeout_sec, json=True))
    receipt = {
        "schema": "lucidota.indy_reads.openai_codex_prompt_guide_ingest.v1",
        "generated_at": now(),
        "status": "PASS",
        "official_url": args.official_url,
        "raw_url": args.raw_url,
        "source_markdown": rel(source_path),
        "policy_json": rel(POLICY_JSON),
        "policy_md": rel(POLICY_MD),
        "policy_law_count": len(policy["laws"]),
        "indy_receipt": child.get("receipt_path"),
        "chunks_written": child.get("chunks_written"),
        "lora_manifest": child.get("lora_manifest"),
        "lora_training_status": child.get("lora_training_status"),
        "max_observed_chunk_tokens": child.get("max_observed_chunk_tokens"),
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
    }
    out = OUT / f"openai_codex_prompt_guide_ingest_{stamp()}.json"
    receipt["receipt_path"] = rel(out)
    write_json(out, receipt)
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-file")
    ap.add_argument("--raw-url", default=RAW_URL)
    ap.add_argument("--official-url", default=OFFICIAL_URL)
    ap.add_argument("--timeout-sec", type=float, default=60.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt = run(args)
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    print("REPORT_PATH=" + receipt["receipt_path"])
    print("OPENAI_CODEX_PROMPT_GUIDE_INGEST=" + receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
