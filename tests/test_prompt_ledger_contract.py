from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIVE_BASE_URL = "http://127.0.0.1:3000"

EXPLICIT_PROMPT_SOURCES = [
    ROOT / "GOALS" / "CURRENT_HANDOFF.md",
    ROOT / "GOALS" / "GOAL_LOG.md",
    ROOT / "GOALS" / "GOAL_PROMPTS.md",
    ROOT / "GOALS" / "GOAL_HANDOFF_PROMPT.md",
    ROOT / "GOALS" / "OPERATION_ROOT_ROTOR_SENDABLE_PROMPT.md",
]
PROMPT_GLOBS = [
    ROOT / "04_RUNTIME" / "luci_delegate",
    ROOT / "04_RUNTIME" / "goals",
]


@dataclass(frozen=True)
class PromptLedgerRecord:
    prompt_id: str
    received_at: str
    source: str
    source_model: str
    receiving_model: str
    target_model: str
    raw_prompt_text: str
    normalized_prompt_text: str
    prompt_hash: str
    session_id: str | None
    parent_prompt_id: str | None
    derived_prompt_ids: tuple[str, ...]
    linked_work_order_uuid: tuple[str, ...]
    linked_receipt_uuid: tuple[str, ...]
    linked_goal_id: str | None
    ontology_tags: tuple[str, ...]
    subsystem_tags: tuple[str, ...]
    status: str
    notes: str
    blockers: str
    explicit_unlinked_reason: str | None
    idempotency_key: str


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_prompt_text(text: str) -> str:
    # Keep semantics stable but strip only incidental trailing whitespace.
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def stable_prompt_id(idempotency_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"lucidota.prompt-ledger::{idempotency_key}"))


def prompt_file_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def build_prompt_record(
    *,
    path: Path,
    source: str,
    source_model: str = "",
    receiving_model: str = "",
    target_model: str = "",
    session_id: str | None = None,
    parent_prompt_id: str | None = None,
    linked_work_order_uuid: tuple[str, ...] = (),
    linked_receipt_uuid: tuple[str, ...] = (),
    linked_goal_id: str | None = None,
    ontology_tags: tuple[str, ...] = (),
    subsystem_tags: tuple[str, ...] = (),
    status: str = "filed",
    notes: str = "",
    blockers: str = "",
    explicit_unlinked_reason: str | None = None,
) -> PromptLedgerRecord:
    raw_prompt_text = path.read_text(encoding="utf-8", errors="replace")
    normalized_prompt_text = normalize_prompt_text(raw_prompt_text)
    prompt_hash = sha256_text(normalized_prompt_text)
    idempotency_key = sha256_text(
        json.dumps(
            {
                "source": source,
                "source_model": source_model,
                "receiving_model": receiving_model,
                "target_model": target_model,
                "prompt_hash": prompt_hash,
                "session_id": session_id,
                "parent_prompt_id": parent_prompt_id,
                "linked_work_order_uuid": list(linked_work_order_uuid),
                "linked_receipt_uuid": list(linked_receipt_uuid),
                "linked_goal_id": linked_goal_id,
                "status": status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    prompt_id = stable_prompt_id(idempotency_key)
    return PromptLedgerRecord(
        prompt_id=prompt_id,
        received_at=prompt_file_mtime(path),
        source=source,
        source_model=source_model,
        receiving_model=receiving_model,
        target_model=target_model,
        raw_prompt_text=raw_prompt_text,
        normalized_prompt_text=normalized_prompt_text,
        prompt_hash=prompt_hash,
        session_id=session_id,
        parent_prompt_id=parent_prompt_id,
        derived_prompt_ids=(),
        linked_work_order_uuid=linked_work_order_uuid,
        linked_receipt_uuid=linked_receipt_uuid,
        linked_goal_id=linked_goal_id,
        ontology_tags=ontology_tags,
        subsystem_tags=subsystem_tags,
        status=status,
        notes=notes,
        blockers=blockers,
        explicit_unlinked_reason=explicit_unlinked_reason,
        idempotency_key=idempotency_key,
    )


def discover_prompt_sources() -> list[Path]:
    sources: list[Path] = []
    for path in EXPLICIT_PROMPT_SOURCES:
        if path.exists():
            sources.append(path)
    for directory in PROMPT_GLOBS:
        if directory.exists():
            for path in sorted(directory.glob("*.prompt")):
                sources.append(path)
    # Keep source discovery deterministic and deduplicated.
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in sources:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def prompt_requires_linkage_or_unlinked(record: PromptLedgerRecord) -> bool:
    return bool(record.linked_work_order_uuid) or bool(record.explicit_unlinked_reason)


def live_openapi_paths() -> set[str]:
    with urllib.request.urlopen(LIVE_BASE_URL + "/", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return set(payload.get("paths", {}).keys())


def live_manual_current_text() -> str:
    with urllib.request.urlopen(LIVE_BASE_URL + "/manual_current?limit=1", timeout=5) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return json.dumps(payload[0], sort_keys=True, ensure_ascii=False)


def test_prompt_hash_and_idempotency_are_stable() -> None:
    sample = ROOT / "GOALS" / "CURRENT_HANDOFF.md"
    first = build_prompt_record(path=sample, source="operator", source_model="shell", receiving_model="codex", target_model="local_model")
    second = build_prompt_record(path=sample, source="operator", source_model="shell", receiving_model="codex", target_model="local_model")

    assert first.raw_prompt_text == second.raw_prompt_text
    assert first.normalized_prompt_text == second.normalized_prompt_text
    assert first.prompt_hash == second.prompt_hash
    assert first.idempotency_key == second.idempotency_key
    assert first.prompt_id == second.prompt_id


def test_prompt_source_catalog_discovers_steering_prompts_and_preserves_raw_text() -> None:
    sources = discover_prompt_sources()
    assert sources, "expected at least one steering prompt source in GOALS or 04_RUNTIME"

    sample = sources[0]
    record = build_prompt_record(
        path=sample,
        source="assistant",
        source_model="codex",
        receiving_model="postgrest",
        target_model="db",
        subsystem_tags=("prompt-ledger",),
        ontology_tags=("GO", "CO", "IO"),
    )

    assert record.raw_prompt_text == sample.read_text(encoding="utf-8", errors="replace")
    assert record.received_at.endswith(("+00:00", "Z")) or "+" in record.received_at
    assert record.prompt_hash == sha256_text(record.normalized_prompt_text)
    assert record.source in {"operator", "assistant", "codex", "vibe", "groq", "local_model", "system"}
    assert record.ontology_tags == ("GO", "CO", "IO")
    assert record.subsystem_tags == ("prompt-ledger",)


def test_prompt_completion_requires_work_order_or_explicit_unlinked_status() -> None:
    sample = ROOT / "GOALS" / "CURRENT_HANDOFF.md"
    linked = build_prompt_record(
        path=sample,
        source="operator",
        linked_work_order_uuid=("00000000-0000-0000-0000-000000000001",),
    )
    unlinked = build_prompt_record(
        path=sample,
        source="operator",
        explicit_unlinked_reason="no work-order yet; queued for later decomposition",
    )
    incomplete = build_prompt_record(path=sample, source="operator")

    assert prompt_requires_linkage_or_unlinked(linked) is True
    assert prompt_requires_linkage_or_unlinked(unlinked) is True
    assert prompt_requires_linkage_or_unlinked(incomplete) is False


def test_prompt_ledger_routes_and_manual_surface_are_blocked_until_live_routes_exist() -> None:
    paths = live_openapi_paths()
    required_routes = {
        "/prompts_filed",
        "/prompt_work_order_links",
        "/prompt_recent",
        "/prompt_unlinked",
        "/prompt_catalog_status",
    }
    missing = sorted(route for route in required_routes if route not in paths)
    if missing:
        pytest.skip(f"prompt ledger routes not live yet: {missing}")

    manual_text = live_manual_current_text().lower()
    assert "prompt ledger" in manual_text
    assert "/prompts_filed" in manual_text
    assert "/prompt_unlinked" in manual_text
    assert "/prompt_catalog_status" in manual_text

