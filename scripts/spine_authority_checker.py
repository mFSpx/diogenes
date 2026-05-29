#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "00_PROJECT_BRAIN" / "spine_authority_registry.json"
OUT = ROOT / "05_OUTPUTS" / "spine"
ALLOWED_EFFECTS = {"queue_absurd_work_order", "stage_graph_packet", "materialize_canonical_graph", "no_canonical_mutation", "external_command", "operator_override"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_registry_hash(data: dict[str, Any]) -> str:
    stripped = copy.deepcopy(data)
    stripped.pop("registry_hash", None)
    return hashlib.sha256(canonical_json(stripped).encode("utf-8")).hexdigest()


def load_registry(path: Path | str = REGISTRY) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def authorities_as_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("authorities") or []
    if isinstance(raw, dict):
        out = []
        for key, value in raw.items():
            if isinstance(value, dict):
                v = dict(value)
                v.setdefault("authority_key", key)
                v.setdefault("authority_class", key)
                out.append(v)
        return out
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def normalize_effect(raw: str | None) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if text in ALLOWED_EFFECTS:
        return text
    low = text.lower()
    if "stage_conversation_command" in low or "stage conversation command" in low:
        return "queue_absurd_work_order" if "queue" in low else "no_canonical_mutation"
    if "queue_dbos" in low or ("queue" in low and "dbos" in low and "work order" in low):
        return "queue_absurd_work_order"
    if "graph_promotion_materialize" in low:
        return "stage_graph_packet"
    if "materialize" in low and "no graph" not in low and "no canonical" not in low:
        return "materialize_canonical_graph"
    if "queue" in low and ("absurd" in low or "work order" in low):
        return "queue_absurd_work_order"
    if "stage" in low and "graph" in low:
        return "stage_graph_packet"
    if "external" in low or "shell" in low or "command" in low and "worker" in low:
        return "external_command"
    if "no graph" in low or "no canonical" in low or "staging" in low:
        return "no_canonical_mutation"
    return re.sub(r"[^a-z0-9]+", "_", low).strip("_")


def validate_registry(data: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if data.get("schema") != "lucidota.spine_authority_registry.v1":
        blockers.append("schema_not_lucidota.spine_authority_registry.v1")
    entries = data.get("roles") or []
    canonical_acceptors = [e.get("path") for e in entries if e.get("canonical_acceptor")]
    canonical_gates = [e.get("path") for e in entries if e.get("canonical_graph_gate")]
    if len(canonical_acceptors) != 1:
        blockers.append(f"canonical_acceptor_count:{len(canonical_acceptors)}")
    if len(canonical_gates) != 1:
        blockers.append(f"canonical_graph_gate_count:{len(canonical_gates)}")
    for e in entries:
        roles = set(e.get("roles") or [])
        path = e.get("path", "<missing>")
        if "proof_furnace" in roles and e.get("direct_canonical_authority_claim"):
            blockers.append(f"proof_furnace_claims_canonical_authority:{path}")
        if "adapter_worker" in roles and e.get("bypasses_canonical_gate"):
            blockers.append(f"adapter_worker_bypasses_canonical_gate:{path}")
        if "adapter_worker" in roles and not e.get("must_route_through"):
            blockers.append(f"adapter_worker_missing_route_contract:{path}")
        if "legacy_danger_surface" in roles and (not e.get("dangerous") or not e.get("quarantined")):
            blockers.append(f"legacy_danger_surface_not_quarantined:{path}")
        if e.get("materialization_helper") and not e.get("blocked_by_policy"):
            blockers.append(f"materialization_helper_not_policy_blocked:{path}")
        if not (ROOT / path).exists():
            blockers.append(f"registered_path_missing:{path}")

    seen: set[str] = set()
    for rule in authorities_as_list(data):
        key = str(rule.get("authority_key") or rule.get("authority_class") or "")
        if not key:
            blockers.append("authority_key_missing")
            continue
        if key in seen:
            blockers.append(f"duplicate_authority_key:{key}")
        seen.add(key)
        evidence = rule.get("required_evidence") or []
        if not isinstance(evidence, list) or not evidence or not all(str(x).strip() for x in evidence):
            blockers.append(f"empty_evidence_rules:{key}")
        effects = rule.get("allowed_effects") or []
        if not isinstance(effects, list) or not effects:
            blockers.append(f"empty_allowed_effects:{key}")
        for effect in effects:
            if effect not in ALLOWED_EFFECTS:
                blockers.append(f"unknown_effect:{key}:{effect}")
        lanes = rule.get("permitted_lanes") or []
        if not isinstance(lanes, list) or not lanes:
            blockers.append(f"empty_permitted_lanes:{key}")
        if not rule.get("owning_subsystem"):
            blockers.append(f"missing_owning_subsystem:{key}")
    return blockers


def find_rule(data: dict[str, Any], authority_class: str) -> dict[str, Any] | None:
    for rule in authorities_as_list(data):
        if authority_class in {rule.get("authority_key"), rule.get("authority_class")}:
            return rule
    return None


def decide_authority(
    *,
    authority_class: str | None,
    effect: str | None,
    lane: str | None,
    evidence_refs: list[str] | None = None,
    operator_override: bool = False,
    registry_data: dict[str, Any] | None = None,
    expected_registry_hash: str | None = None,
) -> dict[str, Any]:
    data = registry_data or load_registry()
    blockers = validate_registry(data)
    rhash = compute_registry_hash(data)
    if expected_registry_hash and expected_registry_hash != rhash:
        blockers.append("stale_registry_hash")
    if not authority_class:
        blockers.append("missing_authority")
        rule = None
    else:
        rule = find_rule(data, authority_class)
        if rule is None:
            blockers.append("unknown_authority")
    normalized_effect = normalize_effect(effect)
    if normalized_effect and normalized_effect not in ALLOWED_EFFECTS:
        blockers.append(f"unknown_requested_effect:{normalized_effect}")
    if rule is not None:
        if normalized_effect and normalized_effect not in set(rule.get("allowed_effects") or []):
            blockers.append("effect_broader_than_registry_permits")
        if lane and lane not in set(rule.get("permitted_lanes") or []):
            blockers.append("lane_not_permitted_for_authority")
        required = list(rule.get("required_evidence") or [])
        if required and not evidence_refs:
            blockers.append("required_evidence_refs_missing")
        if operator_override:
            if authority_class != "operator_authored_assertion":
                blockers.append("operator_override_requires_operator_authored_assertion")
            if not evidence_refs:
                blockers.append("operator_override_requires_evidence")
            if normalized_effect not in {"operator_override", "queue_absurd_work_order", "stage_graph_packet", "no_canonical_mutation"}:
                blockers.append("operator_override_cannot_expand_privilege")
    return {
        "allowed": not blockers,
        "blockers": blockers,
        "authority_class": authority_class,
        "requested_effect": effect,
        "normalized_effect": normalized_effect,
        "lane": lane,
        "operator_override": operator_override,
        "registry_hash": rhash,
        "registry_version": data.get("registry_version"),
        "rule": rule,
    }


def check(data: dict[str, Any]) -> dict[str, Any]:
    blockers = validate_registry(data)
    entries = data.get("roles") or []
    canonical_acceptors = [e["path"] for e in entries if e.get("canonical_acceptor")]
    canonical_gates = [e["path"] for e in entries if e.get("canonical_graph_gate")]
    return {
        "schema": "lucidota.spine_authority_check.v1",
        "generated_at": now(),
        "registry_version": data.get("registry_version"),
        "registry_hash": compute_registry_hash(data),
        "canonical_acceptors": canonical_acceptors,
        "canonical_graph_gates": canonical_gates,
        "authority_rules_checked": len(authorities_as_list(data)),
        "roles_checked": entries,
        "blockers": blockers,
        "canonical_graph_materialization": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS" if not blockers else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate and apply the spine authority registry")
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--output")
    ap.add_argument("--check-decision", action="store_true")
    ap.add_argument("--authority-class")
    ap.add_argument("--effect")
    ap.add_argument("--lane")
    ap.add_argument("--evidence-ref", action="append", default=[])
    ap.add_argument("--expected-registry-hash")
    ap.add_argument("--operator-override", action="store_true")
    args = ap.parse_args()
    data = load_registry(args.registry)
    if args.check_decision:
        decision = decide_authority(
            authority_class=args.authority_class,
            effect=args.effect,
            lane=args.lane,
            evidence_refs=args.evidence_ref,
            operator_override=args.operator_override,
            registry_data=data,
            expected_registry_hash=args.expected_registry_hash,
        )
        payload = {
            "schema": "lucidota.spine_authority_decision.v1",
            "generated_at": now(),
            "authority_decision": decision,
            "blockers": decision["blockers"],
            "canonical_graph_materialization": False,
            "canonical_graph_writes_performed": False,
            "verdict": "PASS" if decision["allowed"] else "FAIL",
        }
    else:
        payload = check(data)
    OUT.mkdir(parents=True, exist_ok=True)
    out = Path(args.output) if args.output else OUT / f"spine_authority_check_{stamp()}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"] = rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH=" + rel(out))
    print("SPINE_AUTHORITY_CHECK=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
