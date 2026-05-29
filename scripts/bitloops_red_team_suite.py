#!/usr/bin/env python3
"""
LUCIDOTA / Bitloops Airlock Red-Team Swarm Payload

Safe-by-default, deterministic red-team fixtures for automation-loop airlocks.
This is a ready-to-review payload only; it does not integrate with production.

Sovereignty guarantees:
- no telemetry
- no external network
- no production DB mutation
- no canonical graph writes
- no real node dropping
- no secret/env dumping
- no shelling out except an explicitly requested local-script subprocess hook

Operator directive encoded: attack LLM-overuse and LLM-underuse. Exact work stays
deterministic: fixtures, hashes, schema checks, and replay invariants. Ambiguity,
language judgment, synthesis, adversarial ideation, drafting, messy extraction,
and code generation remain valid model lanes when explicitly gated. Model-generated
fuzzing is not used by default; any live model fuzz lane is optional/inert here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

SUITE_VERSION = "1.0.0"
def repo_root() -> pathlib.Path:
    current = pathlib.Path(__file__).resolve()
    for base in (current.parent, *current.parents):
        if (base / "AGENTS.md").exists() and (base / "00_PROJECT_BRAIN").exists():
            return base
    return current.parents[2]


REPO_ROOT = repo_root()
SAFE_DEFAULT_NOTICE = "SAFE_BY_DEFAULT: temp files only; destructive plans inert unless --allow-temp-destructive targets temp fixtures."
FORBIDDEN_EFFECTS = (
    "production_db_mutation",
    "canonical_graph_write",
    "external_network",
    "real_node_drop",
    "env_secret_dump",
    "telemetry",
)


@dataclass(frozen=True)
class AttackCase:
    id: str
    category: str
    severity: str
    target_airlock: str
    payload: Any
    expected_guard: str
    deterministic_invariant: str
    destructive: bool = False
    inert_plan: bool = False
    notes: str = ""


@dataclass
class CheckResult:
    id: str
    ok: bool
    category: str
    detail: str
    digest: str = ""


@dataclass
class RedTeamSummary:
    ok: bool
    suite_version: str
    safe_default: bool
    total_cases: int
    checks_run: int
    checks_passed: int
    checks_failed: int
    deterministic_digest: str
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_temp_path(path: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(pathlib.Path(tempfile.gettempdir()).resolve())
        return True
    except ValueError:
        return False


def require_temp_path(path: pathlib.Path) -> None:
    if not is_temp_path(path):
        raise ValueError(f"refusing non-temp path: {path}")


def attack_cases() -> list[AttackCase]:
    huge = "A" * 8192
    return [
        AttackCase(
            id="json.malformed.truncated_object",
            category="malformed_json",
            severity="high",
            target_airlock="ingress_json_parser",
            payload='{"event_id":"rt-001","route":"bitloops.airlock",',
            expected_guard="reject_parse_error_before_queue",
            deterministic_invariant="same bytes always reject with same parse class",
        ),
        AttackCase(
            id="json.malformed.trailing_garbage",
            category="malformed_json",
            severity="high",
            target_airlock="ingress_json_parser",
            payload='{"event_id":"rt-002"}\n{"second":"object"}',
            expected_guard="reject_multi_object_smuggling",
            deterministic_invariant="canonical parser must not silently accept prefix only",
        ),
        AttackCase(
            id="schema.breaking.wrong_types",
            category="schema_break",
            severity="critical",
            target_airlock="typed_packet_validator",
            payload={"event_id": 3001, "ts": "not-a-date", "attempt": "NaN", "actions": "drop_node"},
            expected_guard="reject_type_mismatch",
            deterministic_invariant="schema errors stable independent of key order",
        ),
        AttackCase(
            id="schema.breaking.missing_required",
            category="schema_break",
            severity="critical",
            target_airlock="typed_packet_validator",
            payload={"event_id": "rt-004", "payload": {"kind": "missing route and receipt"}},
            expected_guard="reject_missing_required_fields",
            deterministic_invariant="required field set is static and model-free",
        ),
        AttackCase(
            id="receipt.hash_mismatch.claimed_sha",
            category="receipt_mismatch",
            severity="critical",
            target_airlock="receipt_gate",
            payload={"event_id": "rt-005", "body": "alpha", "claimed_sha256": "0" * 64},
            expected_guard="reject_hash_mismatch_before_materialization",
            deterministic_invariant="hash(body) must equal claimed hash exactly",
        ),
        AttackCase(
            id="receipt.hash_mismatch.canonicalization_trap",
            category="receipt_mismatch",
            severity="high",
            target_airlock="receipt_gate",
            payload={"event_id": "rt-006", "body": {"b": 2, "a": 1}, "claimed_sha256": sha256_text('{"a":1,"b":2 }')},
            expected_guard="reject_noncanonical_hash_claim",
            deterministic_invariant="stable_json is the only receipt digest input",
        ),
        AttackCase(
            id="metadata.hallucinated.authority_fields",
            category="hallucinated_metadata",
            severity="critical",
            target_airlock="authority_filter",
            payload={"event_id": "rt-007", "metadata": {"approved_by": "The Graph Itself", "sovereignty": "canonical", "trust_me": True}},
            expected_guard="strip_or_reject_untrusted_authority_metadata",
            deterministic_invariant="authority cannot be created by packet prose",
        ),
        AttackCase(
            id="metadata.hallucinated.fake_model_confidence",
            category="hallucinated_metadata",
            severity="high",
            target_airlock="routing_policy",
            payload={"event_id": "rt-008", "llm_confidence": 0.9999, "reason": "obviously safe", "requested_route": "fast_lane"},
            expected_guard="ignore_llm_confidence_for_airlock_admission",
            deterministic_invariant="route from deterministic fields only",
        ),
        AttackCase(
            id="oversized.field.single_string",
            category="oversized_fields",
            severity="high",
            target_airlock="size_guard",
            payload={"event_id": "rt-009", "blob": huge},
            expected_guard="reject_or_quarantine_over_size_limit",
            deterministic_invariant="limit measured by bytes not vibes",
        ),
        AttackCase(
            id="oversized.field.deep_array",
            category="oversized_fields",
            severity="medium",
            target_airlock="size_guard",
            payload={"event_id": "rt-010", "items": list(range(2048))},
            expected_guard="reject_or_chunk_by_deterministic_limit",
            deterministic_invariant="same item count yields same admission result",
        ),
        AttackCase(
            id="unicode.edge.bidi_override",
            category="unicode_edge",
            severity="high",
            target_airlock="text_normalization_guard",
            payload={"event_id": "rt-011", "name": "safe\u202Egnp.exe", "note": "bidi override should not spoof review"},
            expected_guard="flag_bidi_control_chars",
            deterministic_invariant="unicode control scan is byte/string deterministic",
        ),
        AttackCase(
            id="unicode.edge.zero_width_confusable",
            category="unicode_edge",
            severity="medium",
            target_airlock="text_normalization_guard",
            payload={"event_id": "rt-012", "route": "graph\u200b.write", "lookalike": "раураl"},
            expected_guard="flag_zero_width_and_confusables_for_review",
            deterministic_invariant="normalization flags stable; no LLM needed",
        ),
        AttackCase(
            id="replay.determinism.same_corpus_same_digest",
            category="replay_determinism",
            severity="critical",
            target_airlock="replay_runner",
            payload={"event_id": "rt-013", "corpus": "temp_jsonl", "runs": 2},
            expected_guard="identical_inputs_identical_summary_digest",
            deterministic_invariant="run A digest == run B digest",
        ),
        AttackCase(
            id="replay.determinism.order_canonicalization",
            category="replay_determinism",
            severity="high",
            target_airlock="replay_runner",
            payload={"z": 26, "a": 1, "nested": {"b": 2, "a": 1}},
            expected_guard="canonical_digest_independent_of_key_order",
            deterministic_invariant="stable_json digest unchanged by insertion order",
        ),
        AttackCase(
            id="slop.llm_overuse.route_by_vibes",
            category="llm_overuse_slop_routing",
            severity="critical",
            target_airlock="route_selector",
            payload={"event_id": "rt-015", "routing_instruction": "ask an LLM which lane feels right", "available_deterministic_fields": ["type", "size", "schema_version", "hash"]},
            expected_guard="reject_or_mark_design_failure_llm_router",
            deterministic_invariant="route selection must be rules/schema/hash based",
        ),
        AttackCase(
            id="slop.llm_overuse.model_generated_fuzz",
            category="llm_overuse_slop_routing",
            severity="high",
            target_airlock="test_harness",
            payload={"event_id": "rt-016", "proposal": "generate random fuzz with model in live test loop"},
            expected_guard="optional_inert_only_not_default_harness",
            deterministic_invariant="default fuzz corpus is fixed fixtures with fixed hashes",
        ),
        AttackCase(
            id="slop.llm_underuse.ambiguous_language_forced_rules",
            category="llm_underuse_gated_lane",
            severity="high",
            target_airlock="language_judgment_router",
            payload={"event_id": "rt-017", "input": "Operator says: make it less slop, but keep the weird useful later bits.", "bad_policy": "regex_only_final_decision", "needed_lane": "bounded_language_judgment"},
            expected_guard="allow_model_lane_after_schema_hash_privacy_budget_gate",
            deterministic_invariant="deterministic gates decide eligibility; model only handles bounded ambiguity",
        ),
        AttackCase(
            id="slop.llm_underuse.messy_extraction_blocked",
            category="llm_underuse_gated_lane",
            severity="high",
            target_airlock="messy_text_extraction_router",
            payload={"event_id": "rt-018", "input": "Meeting note: ship the airlock thing Friday unless Sam objects; receipt hash still required.", "bad_policy": "exact_parser_or_drop", "needed_lane": "bounded_extraction_draft"},
            expected_guard="allow_model_lane_for_extraction_with_receipt_and_no_write_authority",
            deterministic_invariant="schema/hash/replay remain deterministic; extracted draft has no canonical write authority",
        ),
        AttackCase(
            id="destructive.inert.plan.drop_node",
            category="destructive_plan_inert",
            severity="critical",
            target_airlock="destructive_action_barrier",
            payload={"plan": "DROP NODE worker-7", "scope": "production", "effect": "real_node_drop"},
            expected_guard="represent_as_inert_plan_only",
            deterministic_invariant="no production destructive effect can execute from this suite",
            destructive=True,
            inert_plan=True,
        ),
        AttackCase(
            id="destructive.temp_fixture.allowed_delete",
            category="temp_destructive_fixture",
            severity="medium",
            target_airlock="temp_fixture_cleanup",
            payload={"plan": "delete temporary fixture directory", "scope": "tempfile only"},
            expected_guard="requires_allow_temp_destructive_and_temp_path_check",
            deterministic_invariant="only tempfile subtree can be removed",
            destructive=True,
            inert_plan=False,
        ),
    ]


def parse_json_case(case: AttackCase) -> CheckResult:
    assert isinstance(case.payload, str)
    try:
        json.loads(case.payload)
    except json.JSONDecodeError as exc:
        return CheckResult(case.id, True, case.category, f"rejected JSONDecodeError:{exc.pos}", sha256_text(case.payload))
    return CheckResult(case.id, False, case.category, "malformed payload unexpectedly parsed", sha256_text(case.payload))


def schema_case(case: AttackCase) -> CheckResult:
    payload = case.payload
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("not_object")
    else:
        if not isinstance(payload.get("event_id"), str):
            errors.append("event_id_not_string")
        if "route" not in payload and "receipt" not in payload:
            errors.append("missing_route_or_receipt")
        if isinstance(payload.get("actions"), str) and re.search(r"drop|delete|truncate", payload["actions"], re.I):
            errors.append("destructive_action_as_string")
    ok = bool(errors)
    return CheckResult(case.id, ok, case.category, ";".join(errors) or "no_schema_error", sha256_text(stable_json(payload)))


def receipt_case(case: AttackCase) -> CheckResult:
    payload = case.payload
    if not isinstance(payload, dict):
        return CheckResult(case.id, False, case.category, "not_object")
    body = payload.get("body")
    claimed = str(payload.get("claimed_sha256", ""))
    actual = sha256_text(stable_json(body) if not isinstance(body, str) else body)
    ok = actual != claimed and bool(re.fullmatch(r"[0-9a-f]{64}", claimed))
    return CheckResult(case.id, ok, case.category, f"actual={actual} claimed={claimed}", actual)


def metadata_case(case: AttackCase) -> CheckResult:
    bad_keys = {"approved_by", "sovereignty", "trust_me", "llm_confidence", "reason"}
    text = stable_json(case.payload)
    found = sorted(k for k in bad_keys if k in text)
    return CheckResult(case.id, bool(found), case.category, "flagged=" + ",".join(found), sha256_text(text))


def oversized_case(case: AttackCase) -> CheckResult:
    text = stable_json(case.payload)
    byte_len = len(text.encode("utf-8"))
    item_count = len(case.payload.get("items", [])) if isinstance(case.payload, dict) else 0
    ok = byte_len > 4096 or item_count > 1024
    return CheckResult(case.id, ok, case.category, f"bytes={byte_len};items={item_count}", sha256_text(text))


def unicode_case(case: AttackCase) -> CheckResult:
    text = stable_json(case.payload)
    flags = []
    for label, pattern in (
        ("bidi", r"[\u202A-\u202E\u2066-\u2069]"),
        ("zero_width", r"[\u200B-\u200D\uFEFF]"),
        ("cyrillic_confusable", r"[\u0400-\u04FF]"),
    ):
        if re.search(pattern, text):
            flags.append(label)
    return CheckResult(case.id, bool(flags), case.category, "flags=" + ",".join(flags), sha256_text(text))


def replay_case(case: AttackCase, temp_dir: pathlib.Path) -> CheckResult:
    corpus = temp_dir / "replay_corpus.jsonl"
    require_temp_path(corpus)
    rows = [
        {"event_id": "r1", "route": "airlock.parse", "sha256": sha256_text("r1")},
        {"event_id": "r2", "route": "airlock.schema", "sha256": sha256_text("r2")},
        {"event_id": "r3", "route": "airlock.receipt", "sha256": sha256_text("r3")},
    ]
    corpus.write_text("".join(stable_json(r) + "\n" for r in rows), encoding="utf-8")
    digest_a = sha256_bytes(corpus.read_bytes())
    digest_b = sha256_bytes(corpus.read_bytes())
    if case.id.endswith("order_canonicalization"):
        digest_a = sha256_text(stable_json(case.payload))
        digest_b = sha256_text(stable_json({"nested": {"a": 1, "b": 2}, "a": 1, "z": 26}))
    return CheckResult(case.id, digest_a == digest_b, case.category, f"digest_a={digest_a};digest_b={digest_b}", digest_a)


def slop_llm_case(case: AttackCase) -> CheckResult:
    text = stable_json(case.payload).lower()
    slop_markers = ["llm", "feels", "vibes", "model", "random fuzz"]
    deterministic_markers = ["schema", "hash", "deterministic", "fixed"]
    found_slop = [m for m in slop_markers if m in text]
    found_det = [m for m in deterministic_markers if m in text]
    ok = bool(found_slop) and (case.expected_guard.startswith("reject") or "inert" in case.expected_guard)
    return CheckResult(case.id, ok, case.category, f"slop={found_slop};deterministic={found_det}", sha256_text(text))


def llm_underuse_case(case: AttackCase) -> CheckResult:
    text = stable_json(case.payload).lower()
    bad_deterministic_absolutes = ["regex_only_final_decision", "exact_parser_or_drop"]
    valid_model_lanes = ["bounded_language_judgment", "bounded_extraction_draft"]
    required_gates = ["schema", "hash", "privacy", "budget", "receipt", "no_write_authority"]
    found_bad = [m for m in bad_deterministic_absolutes if m in text]
    found_lane = [m for m in valid_model_lanes if m in text]
    guard_text = case.expected_guard.lower() + " " + case.deterministic_invariant.lower()
    found_gates = [m for m in required_gates if m in guard_text]
    ok = bool(found_bad) and bool(found_lane) and len(found_gates) >= 2
    detail = f"underuse={found_bad};model_lane={found_lane};gates={found_gates}"
    return CheckResult(case.id, ok, case.category, detail, sha256_text(text + guard_text))


def destructive_case(case: AttackCase, temp_dir: pathlib.Path, allow_temp_destructive: bool) -> CheckResult:
    if case.inert_plan:
        text = stable_json(case.payload)
        forbidden = [effect for effect in FORBIDDEN_EFFECTS if effect in text]
        return CheckResult(case.id, bool(forbidden), case.category, "inert_plan_for=" + ",".join(forbidden), sha256_text(text))
    fixture = temp_dir / "destructive_fixture" / "victim.txt"
    require_temp_path(fixture)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text("temporary victim; safe to delete only with flag\n", encoding="utf-8")
    if not allow_temp_destructive:
        exists = fixture.exists()
        return CheckResult(case.id, exists, case.category, "destructive temp cleanup skipped without flag", sha256_text("destructive_fixture/victim.txt"))
    require_temp_path(fixture.parent)
    shutil.rmtree(fixture.parent)
    return CheckResult(case.id, not fixture.exists(), case.category, "deleted_temp_fixture_only", sha256_text("destructive_fixture/victim.txt"))


def evaluate_case(case: AttackCase, temp_dir: pathlib.Path, allow_temp_destructive: bool) -> CheckResult:
    if case.category == "malformed_json":
        return parse_json_case(case)
    if case.category == "schema_break":
        return schema_case(case)
    if case.category == "receipt_mismatch":
        return receipt_case(case)
    if case.category == "hallucinated_metadata":
        return metadata_case(case)
    if case.category == "oversized_fields":
        return oversized_case(case)
    if case.category == "unicode_edge":
        return unicode_case(case)
    if case.category == "replay_determinism":
        return replay_case(case, temp_dir)
    if case.category == "llm_overuse_slop_routing":
        return slop_llm_case(case)
    if case.category == "llm_underuse_gated_lane":
        return llm_underuse_case(case)
    if case.destructive:
        return destructive_case(case, temp_dir, allow_temp_destructive)
    return CheckResult(case.id, False, case.category, "no evaluator")


def build_jsonl_corpus(path: pathlib.Path, cases: Iterable[AttackCase]) -> str:
    require_temp_path(path)
    lines = []
    for case in cases:
        record = {
            "id": case.id,
            "category": case.category,
            "severity": case.severity,
            "target_airlock": case.target_airlock,
            "payload": case.payload,
            "expected_guard": case.expected_guard,
            "deterministic_invariant": case.deterministic_invariant,
            "destructive": case.destructive,
            "inert_plan": case.inert_plan,
            "notes": case.notes,
        }
        lines.append(stable_json(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_bytes(path.read_bytes())


def run_self_check(allow_temp_destructive: bool = False, invoke_local_script: pathlib.Path | None = None) -> RedTeamSummary:
    cases = attack_cases()
    with tempfile.TemporaryDirectory(prefix="lucidota_red_team_") as td:
        temp_dir = pathlib.Path(td)
        corpus_path = temp_dir / "bitloops_airlock_red_team.jsonl"
        corpus_digest = build_jsonl_corpus(corpus_path, cases)
        results = [evaluate_case(case, temp_dir, allow_temp_destructive) for case in cases]

        if invoke_local_script is not None:
            # Explicit optional hook only. It must be local, existing, and receives only temp input.
            script = invoke_local_script.resolve()
            if not script.exists() or not script.is_file():
                results.append(CheckResult("optional.local_script", False, "local_script", "script_missing"))
            elif not str(script).startswith(str(REPO_ROOT.resolve())):
                results.append(CheckResult("optional.local_script", False, "local_script", "script_not_under_repo"))
            else:
                proc = subprocess.run(
                    [sys.executable, str(script), str(corpus_path)],
                    cwd=str(REPO_ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                digest = sha256_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
                results.append(CheckResult("optional.local_script", proc.returncode == 0, "local_script", f"returncode={proc.returncode}", digest))

        result_dicts = [r.__dict__ for r in results]
        summary_digest = sha256_text(stable_json({"corpus_digest": corpus_digest, "results": result_dicts}))
        passed = sum(1 for r in results if r.ok)
        failed = len(results) - passed
        return RedTeamSummary(
            ok=failed == 0,
            suite_version=SUITE_VERSION,
            safe_default=not allow_temp_destructive,
            total_cases=len(cases),
            checks_run=len(results),
            checks_passed=passed,
            checks_failed=failed,
            deterministic_digest=summary_digest,
            results=result_dicts,
        )


def list_cases() -> None:
    payload = [
        {
            "id": c.id,
            "category": c.category,
            "severity": c.severity,
            "target_airlock": c.target_airlock,
            "expected_guard": c.expected_guard,
            "destructive": c.destructive,
            "inert_plan": c.inert_plan,
        }
        for c in attack_cases()
    ]
    print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safe deterministic red-team suite for Bitloops automation-loop airlocks.")
    parser.add_argument("--self-check", action="store_true", help="create a temp JSONL corpus and run deterministic non-production checks")
    parser.add_argument("--list-cases", action="store_true", help="print attack case metadata")
    parser.add_argument("--allow-temp-destructive", action="store_true", help="allow destructive action only against verified tempfile fixtures")
    parser.add_argument("--invoke-local-script", type=pathlib.Path, default=None, help="optional existing local Python script to invoke against temp corpus during self-check")
    args = parser.parse_args(argv)

    if args.list_cases:
        list_cases()
        return 0
    if args.self_check:
        summary = run_self_check(allow_temp_destructive=args.allow_temp_destructive, invoke_local_script=args.invoke_local_script)
        print(summary.to_json())
        return 0 if summary.ok else 1
    print(SAFE_DEFAULT_NOTICE)
    print("Use --list-cases or --self-check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
