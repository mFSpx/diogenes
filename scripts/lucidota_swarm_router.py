#!/usr/bin/env python3
"""Deterministic sovereign swarm routing payload for LUCIDOTA.

This module is intentionally inert: it makes no provider calls, reads no secrets,
opens no network sockets, shells out nowhere, and emits no telemetry.  It is a
ready-to-review routing blueprint plus adapter stubs for Central Commander to
airlock before any provider is enabled.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

SCHEMA = "lucidota.swarm_router.final.v1"

SECURITY_RANK = {
    "public": 0,
    "low": 1,
    "internal": 2,
    "confidential": 3,
    "secret": 4,
    "sovereign": 5,
}
PRIVACY_RANK = {
    "none": 0,
    "low": 1,
    "personal": 2,
    "sensitive": 3,
    "regulated": 4,
    "sovereign": 5,
}
LATENCY_RANK = {"batch": 0, "normal": 1, "fast": 2, "urgent": 3}
COMPUTE_RANK = {"tiny": 0, "light": 1, "medium": 2, "heavy": 3, "extreme": 4}

CAP_TEXT = "text"
CAP_CODE = "code"
CAP_STRUCTURED = "structured_output"
CAP_LONG_CONTEXT = "long_context"
CAP_TOOL_USE = "tool_use"
CAP_VISION = "vision"
CAP_REASONING = "reasoning"
CAP_LOCAL_ONLY = "local_only"
CAP_EXACT_CHECK = "exact_check"
CAP_POLICY = "policy"
CAP_RECEIPT = "receipt"
CAP_LANGUAGE_JUDGMENT = "language_judgment"
CAP_MODEL_JUDGMENT = "model_judgment"

DETERMINISTIC_CAPABILITIES = {CAP_TEXT, CAP_CODE, CAP_STRUCTURED, CAP_TOOL_USE, CAP_LOCAL_ONLY, CAP_EXACT_CHECK, CAP_POLICY, CAP_RECEIPT}
MODEL_JUDGMENT_CAPABILITIES = {CAP_REASONING, CAP_VISION, CAP_LONG_CONTEXT, CAP_LANGUAGE_JUDGMENT, CAP_MODEL_JUDGMENT}
SOVEREIGN_CAPABILITIES = {CAP_TEXT, CAP_CODE, CAP_STRUCTURED, CAP_TOOL_USE, CAP_REASONING, CAP_LOCAL_ONLY, CAP_LANGUAGE_JUDGMENT, CAP_MODEL_JUDGMENT}
GROQ_CAPABILITIES = {CAP_TEXT, CAP_CODE, CAP_STRUCTURED, CAP_TOOL_USE, CAP_REASONING}
COHERE_CAPABILITIES = {CAP_TEXT, CAP_STRUCTURED, CAP_LONG_CONTEXT, CAP_REASONING}
DEFAULT_CAPABILITIES = {CAP_TEXT, CAP_CODE, CAP_STRUCTURED, CAP_TOOL_USE, CAP_VISION, CAP_REASONING, CAP_LONG_CONTEXT}
DEFAULT_LANE_COST_UNITS_PER_1K_TOKENS = {
    "deterministic_workflow": 0.0,
    "local_sovereign": 80.0,
    "groq_like_fast": 20.0,
    "cohere_like_context": 60.0,
    "default_airlock": 250.0,
}
DEFAULT_LANE_COST_SOURCE = "default_relative_cost_units"


@dataclass(frozen=True)
class TaskSpec:
    """Provider-neutral task description used by the deterministic router."""

    task_id: str = "task"
    security_level: str = "internal"
    privacy_sensitivity: str = "low"
    latency_preference: str = "normal"
    compute_weight: str = "medium"
    required_capabilities: tuple[str, ...] = (CAP_TEXT,)
    allow_external: bool = False
    prefer_local: bool = True
    requires_model_judgment: bool = False
    deterministic_workflow_available: bool = True
    expected_tokens: int = 2048
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TaskSpec":
        caps = data.get("required_capabilities", (CAP_TEXT,))
        if isinstance(caps, str):
            caps = [caps]
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {"raw_metadata": metadata}
        cap_tuple = tuple(sorted({str(c).strip().lower() for c in caps if str(c).strip()}))
        explicit_model = data.get("requires_model_judgment", None)
        if explicit_model is None:
            explicit_model = bool(set(cap_tuple) & MODEL_JUDGMENT_CAPABILITIES)
        return cls(
            task_id=str(data.get("task_id", "task")),
            security_level=normalize_choice(data.get("security_level"), SECURITY_RANK, "internal"),
            privacy_sensitivity=normalize_choice(data.get("privacy_sensitivity"), PRIVACY_RANK, "low"),
            latency_preference=normalize_choice(data.get("latency_preference"), LATENCY_RANK, "normal"),
            compute_weight=normalize_choice(data.get("compute_weight"), COMPUTE_RANK, "medium"),
            required_capabilities=cap_tuple,
            allow_external=bool(data.get("allow_external", False)),
            prefer_local=bool(data.get("prefer_local", True)),
            requires_model_judgment=bool(explicit_model),
            deterministic_workflow_available=bool(data.get("deterministic_workflow_available", True)),
            expected_tokens=max(1, int(data.get("expected_tokens", 2048))),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "security_level": self.security_level,
            "privacy_sensitivity": self.privacy_sensitivity,
            "latency_preference": self.latency_preference,
            "compute_weight": self.compute_weight,
            "required_capabilities": list(self.required_capabilities),
            "allow_external": self.allow_external,
            "prefer_local": self.prefer_local,
            "requires_model_judgment": self.requires_model_judgment,
            "deterministic_workflow_available": self.deterministic_workflow_available,
            "expected_tokens": self.expected_tokens,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LaneProfile:
    name: str
    provider_family: str
    capabilities: frozenset[str]
    max_security_rank: int
    max_privacy_rank: int
    latency_score: int
    max_compute_rank: int
    external: bool
    model_lane: bool
    base_timeout_seconds: int
    notes: str

    def supports(self, required: set[str]) -> bool:
        return required.issubset(self.capabilities)


class ModelAdapter(Protocol):
    """Airlock interface only. Implementations must be injected explicitly."""

    lane_name: str

    def complete(self, task: TaskSpec, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return provider output. Stub contract; this module never calls it."""


class DisabledAdapter:
    """Non-network adapter stub for review, tests, and safe default wiring."""

    def __init__(self, lane_name: str) -> None:
        self.lane_name = lane_name

    def complete(self, task: TaskSpec, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError(
            f"Adapter for lane {self.lane_name!r} is disabled. "
            "Central Commander must explicitly airlock and inject a real provider."
        )


LANES: tuple[LaneProfile, ...] = (
    LaneProfile(
        name="deterministic_workflow",
        provider_family="deterministic",
        capabilities=frozenset(DETERMINISTIC_CAPABILITIES),
        max_security_rank=SECURITY_RANK["sovereign"],
        max_privacy_rank=PRIVACY_RANK["sovereign"],
        latency_score=LATENCY_RANK["urgent"],
        max_compute_rank=COMPUTE_RANK["heavy"],
        external=False,
        model_lane=False,
        base_timeout_seconds=10,
        notes="Preferred non-LLM lane for exact checks, rules, policies, receipts, schemas, and fixed workflows.",
    ),
    LaneProfile(
        name="local_sovereign",
        provider_family="local",
        capabilities=frozenset(SOVEREIGN_CAPABILITIES),
        max_security_rank=SECURITY_RANK["sovereign"],
        max_privacy_rank=PRIVACY_RANK["sovereign"],
        latency_score=LATENCY_RANK["normal"],
        max_compute_rank=COMPUTE_RANK["heavy"],
        external=False,
        model_lane=True,
        base_timeout_seconds=120,
        notes="Default for confidential, sensitive, local-only, and sovereignty-bound tasks.",
    ),
    LaneProfile(
        name="groq_like_fast",
        provider_family="groq_like",
        capabilities=frozenset(GROQ_CAPABILITIES),
        max_security_rank=SECURITY_RANK["low"],
        max_privacy_rank=PRIVACY_RANK["low"],
        latency_score=LATENCY_RANK["urgent"],
        max_compute_rank=COMPUTE_RANK["heavy"],
        external=True,
        model_lane=True,
        base_timeout_seconds=30,
        notes="Fast external lane for low-risk text/code/structured tasks only.",
    ),
    LaneProfile(
        name="cohere_like_context",
        provider_family="cohere_like",
        capabilities=frozenset(COHERE_CAPABILITIES),
        max_security_rank=SECURITY_RANK["low"],
        max_privacy_rank=PRIVACY_RANK["personal"],
        latency_score=LATENCY_RANK["normal"],
        max_compute_rank=COMPUTE_RANK["medium"],
        external=True,
        model_lane=True,
        base_timeout_seconds=75,
        notes="External long-context/text lane for explicitly external-safe work.",
    ),
    LaneProfile(
        name="default_airlock",
        provider_family="default",
        capabilities=frozenset(DEFAULT_CAPABILITIES),
        max_security_rank=SECURITY_RANK["public"],
        max_privacy_rank=PRIVACY_RANK["none"],
        latency_score=LATENCY_RANK["normal"],
        max_compute_rank=COMPUTE_RANK["extreme"],
        external=True,
        model_lane=True,
        base_timeout_seconds=90,
        notes="Last-resort external placeholder; must remain disabled until reviewed.",
    ),
)


def normalize_choice(value: Any, allowed: Mapping[str, int], default: str) -> str:
    key = str(value if value is not None else default).strip().lower().replace("-", "_")
    return key if key in allowed else default


def rank_task(task: TaskSpec) -> dict[str, int]:
    return {
        "security": SECURITY_RANK[task.security_level],
        "privacy": PRIVACY_RANK[task.privacy_sensitivity],
        "latency": LATENCY_RANK[task.latency_preference],
        "compute": COMPUTE_RANK[task.compute_weight],
    }


def lane_blockers(task: TaskSpec, lane: LaneProfile) -> list[str]:
    ranks = rank_task(task)
    required = set(task.required_capabilities)
    blockers: list[str] = []
    if lane.external and not task.allow_external:
        blockers.append("external_not_allowed")
    if lane.model_lane and not task.requires_model_judgment:
        blockers.append("model_judgment_not_required")
    if lane.name == "deterministic_workflow" and not task.deterministic_workflow_available:
        blockers.append("deterministic_workflow_unavailable")
    if CAP_LOCAL_ONLY in required and lane.external:
        blockers.append("local_only_required")
    if ranks["security"] > lane.max_security_rank:
        blockers.append(f"security>{lane.name}_max")
    if ranks["privacy"] > lane.max_privacy_rank:
        blockers.append(f"privacy>{lane.name}_max")
    if ranks["compute"] > lane.max_compute_rank:
        blockers.append(f"compute>{lane.name}_max")
    missing = sorted(required - lane.capabilities)
    if missing:
        blockers.append("missing_capabilities:" + ",".join(missing))
    return blockers


def timeout_budget(task: TaskSpec, lane: LaneProfile) -> dict[str, int]:
    ranks = rank_task(task)
    token_bonus = min(120, max(0, (task.expected_tokens - 2048) // 512) * 5)
    compute_bonus = ranks["compute"] * 20
    privacy_review_bonus = 30 if ranks["privacy"] >= PRIVACY_RANK["sensitive"] else 0
    urgent_discount = 10 if task.latency_preference == "urgent" and lane.latency_score >= LATENCY_RANK["fast"] else 0
    primary = max(10, lane.base_timeout_seconds + compute_bonus + token_bonus + privacy_review_bonus - urgent_discount)
    return {
        "connect_seconds": 0 if not lane.external else 5,
        "first_token_seconds": max(5, min(primary, primary // 3)),
        "total_seconds": primary,
        "fallback_total_seconds": max(20, primary + 45),
    }


def cost_priority(task: TaskSpec) -> str:
    raw = str(task.metadata.get("cost_priority", task.metadata.get("budget_priority", "normal"))).strip().lower()
    return raw if raw in {"low", "normal", "high"} else "normal"


def lane_cost_unit(task: TaskSpec, lane: LaneProfile) -> tuple[float, str]:
    overrides = task.metadata.get("lane_cost_units_per_1k_tokens", {})
    if isinstance(overrides, Mapping) and lane.name in overrides:
        try:
            return max(0.0, float(overrides[lane.name])), "task_metadata_override"
        except (TypeError, ValueError):
            return DEFAULT_LANE_COST_UNITS_PER_1K_TOKENS[lane.name], "invalid_task_metadata_override_defaulted"
    return DEFAULT_LANE_COST_UNITS_PER_1K_TOKENS[lane.name], DEFAULT_LANE_COST_SOURCE


def lane_cost_evaluation(task: TaskSpec, lane: LaneProfile) -> dict[str, Any]:
    per_1k, source = lane_cost_unit(task, lane)
    estimated_units = round(per_1k * (task.expected_tokens / 1000.0), 6)
    return {
        "unit": "relative_cost_units_per_1k_tokens",
        "cost_units_per_1k_tokens": per_1k,
        "estimated_cost_units": estimated_units,
        "source": source,
        "network_calls": False,
        "provider_calls": False,
    }


def cost_penalty(task: TaskSpec, lane: LaneProfile) -> int:
    if not lane.model_lane:
        return 0
    estimated = float(lane_cost_evaluation(task, lane)["estimated_cost_units"])
    priority = cost_priority(task)
    if priority == "high":
        return int(min(240, round(estimated * 0.5)))
    if priority == "low":
        return int(min(20, round(estimated * 0.03)))
    return int(min(80, round(estimated * 0.12)))


def score_lane(task: TaskSpec, lane: LaneProfile) -> int:
    ranks = rank_task(task)
    score = 100
    if lane.external:
        score -= 45
    else:
        score += 30
    if task.prefer_local and not lane.external:
        score += 20
    if task.allow_external and lane.external:
        score += 35 if not task.prefer_local else 8
    if task.latency_preference in {"fast", "urgent"}:
        score += lane.latency_score * 14
        if task.allow_external and not task.prefer_local and lane.name == "groq_like_fast":
            score += 45
    else:
        score += max(0, 3 - lane.latency_score) * 3
    score -= abs(lane.max_compute_rank - ranks["compute"]) * 2
    if CAP_LONG_CONTEXT in task.required_capabilities and lane.name == "cohere_like_context":
        score += 22
    if CAP_VISION in task.required_capabilities and lane.name == "default_airlock":
        score += 12
    if lane.name == "deterministic_workflow":
        if task.requires_model_judgment:
            score -= 120
        else:
            score += 200
    if lane.name == "default_airlock":
        score -= 25
    score -= cost_penalty(task, lane)
    return score


def route_task(task: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(task, TaskSpec):
        task = TaskSpec.from_mapping(task)

    evaluations = []
    eligible: list[tuple[int, LaneProfile]] = []
    for lane in LANES:
        blockers = lane_blockers(task, lane)
        score = score_lane(task, lane) if not blockers else -1
        evaluations.append({
            "lane": lane.name,
            "provider_family": lane.provider_family,
            "eligible": not blockers,
            "score": score,
            "cost_penalty": cost_penalty(task, lane) if not blockers else 0,
            "blockers": blockers,
            "notes": lane.notes,
        })
        if not blockers:
            eligible.append((score, lane))

    if eligible:
        primary = sorted(eligible, key=lambda item: (-item[0], item[1].name))[0][1]
        status = "routed"
        reason = "highest_policy_score_among_eligible_lanes"
    else:
        primary = LANES[0]
        status = "manual_review_required"
        reason = "no_lane_satisfied_all_constraints; retained in local sovereign airlock"

    fallback_lanes = [lane for _, lane in sorted(eligible, key=lambda item: (-item[0], item[1].name)) if lane.name != primary.name]
    if primary.name != "local_sovereign":
        fallback_lanes.append(LANES[0])
    deduped_fallbacks: list[LaneProfile] = []
    seen = {primary.name}
    for lane in fallback_lanes:
        if lane.name not in seen:
            deduped_fallbacks.append(lane)
            seen.add(lane.name)

    required = set(task.required_capabilities)
    return {
        "schema": SCHEMA,
        "deterministic": True,
        "status": status,
        "reason": reason,
        "task": task.to_dict(),
        "routing_policy": {
            "deterministic_first": True,
            "deterministic_preflight_before_model": True,
            "llm_only_when_model_judgment_required": True,
            "model_lanes_preserved_for_judgment": True,
            "cost_aware": True,
            "cost_policy_source": "task_metadata_override_or_default_relative_units",
            "exact_checks_before_model": True,
            "fixed_fallbacks": True,
            "receipt_ready": True,
        },
        "risk_evaluation": {
            "security_rank": SECURITY_RANK[task.security_level],
            "privacy_rank": PRIVACY_RANK[task.privacy_sensitivity],
            "latency_rank": LATENCY_RANK[task.latency_preference],
            "compute_rank": COMPUTE_RANK[task.compute_weight],
            "external_allowed": task.allow_external,
            "local_only": CAP_LOCAL_ONLY in required,
            "requires_model_judgment": task.requires_model_judgment,
            "deterministic_workflow_available": task.deterministic_workflow_available,
        },
        "decision": {
            "primary_lane": primary.name,
            "provider_family": primary.provider_family,
            "model_lane": primary.model_lane,
            "adapter_enabled": False,
            "adapter_contract": "ModelAdapter.complete(task, payload) -> Mapping[str, Any]",
            "timeout_budget": timeout_budget(task, primary),
        },
        "cost_evaluation": {
            "priority": cost_priority(task),
            "expected_tokens": task.expected_tokens,
            "lanes": {
                lane.name: lane_cost_evaluation(task, lane)
                for lane in LANES
            },
        },
        "fallbacks": [
            {
                "lane": lane.name,
                "provider_family": lane.provider_family,
                "model_lane": lane.model_lane,
                "adapter_enabled": False,
                "timeout_budget": timeout_budget(task, lane),
            }
            for lane in deduped_fallbacks
        ],
        "lane_evaluations": evaluations,
        "sovereignty_guards": {
            "telemetry": False,
            "implicit_network": False,
            "shell_out": False,
            "env_secret_dumping": False,
            "provider_calls": False,
            "llm_for_deterministic_work": False,
            "latest_curl_pip_behavior": False,
        },
    }


def adapter_registry() -> dict[str, DisabledAdapter]:
    """Return disabled adapters only; Central Commander must replace explicitly."""
    return {lane.name: DisabledAdapter(lane.name) for lane in LANES}


def self_check_payload() -> list[dict[str, Any]]:
    examples: Sequence[Mapping[str, Any]] = (
        {
            "task_id": "sovereign_code_review",
            "security_level": "sovereign",
            "privacy_sensitivity": "sovereign",
            "latency_preference": "normal",
            "compute_weight": "medium",
            "required_capabilities": [CAP_CODE, CAP_REASONING, CAP_LOCAL_ONLY],
            "allow_external": False,
            "prefer_local": True,
            "expected_tokens": 4096,
            "requires_model_judgment": True,
        },
        {
            "task_id": "deterministic_policy_receipt",
            "security_level": "sovereign",
            "privacy_sensitivity": "sensitive",
            "latency_preference": "urgent",
            "compute_weight": "tiny",
            "required_capabilities": [CAP_POLICY, CAP_EXACT_CHECK, CAP_RECEIPT],
            "allow_external": False,
            "prefer_local": True,
            "requires_model_judgment": False,
            "expected_tokens": 256,
        },
        {
            "task_id": "fast_public_json",
            "security_level": "public",
            "privacy_sensitivity": "none",
            "latency_preference": "urgent",
            "compute_weight": "light",
            "required_capabilities": [CAP_TEXT, CAP_STRUCTURED],
            "allow_external": True,
            "prefer_local": False,
            "expected_tokens": 1024,
            "requires_model_judgment": True,
        },
        {
            "task_id": "long_context_digest",
            "security_level": "low",
            "privacy_sensitivity": "personal",
            "latency_preference": "batch",
            "compute_weight": "medium",
            "required_capabilities": [CAP_TEXT, CAP_LONG_CONTEXT],
            "allow_external": True,
            "prefer_local": False,
            "expected_tokens": 12000,
            "requires_model_judgment": True,
        },
        {
            "task_id": "vision_public_airlock",
            "security_level": "public",
            "privacy_sensitivity": "none",
            "latency_preference": "normal",
            "compute_weight": "heavy",
            "required_capabilities": [CAP_VISION, CAP_TEXT],
            "allow_external": True,
            "prefer_local": False,
            "expected_tokens": 2048,
            "requires_model_judgment": True,
        },
    )
    return [route_task(spec) for spec in examples]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic LUCIDOTA swarm routing payload")
    parser.add_argument("--self-check", action="store_true", help="print deterministic JSON for representative task specs")
    parser.add_argument("--task-json", help="route one task spec JSON object and print deterministic JSON")
    args = parser.parse_args(argv)

    if args.self_check:
        print(json.dumps(self_check_payload(), indent=2, sort_keys=True))
        return 0
    if args.task_json:
        data = json.loads(args.task_json)
        if not isinstance(data, dict):
            raise SystemExit("--task-json must decode to a JSON object")
        print(json.dumps(route_task(data), indent=2, sort_keys=True))
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
