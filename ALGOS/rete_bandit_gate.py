#!/usr/bin/env python3
"""RETE-style deterministic pruning + bandit/regret routing for Bytewax/Absurd.

Pure local execution only. No network calls. No canonical graph writes.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ALGOS.bandit_router import BanditUpdate, select_action, update_policy
from ALGOS.regret_engine import MathAction, MathCounterfactual, compute_regret_weighted_strategy

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "lucidota.rete_bandit_gate.v1"
GO25 = [
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "FRICTION", "LEVERAGE", "VISIBILITY",
    "ACTION", "EVENT", "TIME", "PATTERN", "HYPOTHESIS", "CLAIM", "EVIDENCE",
    "ATOMIC_ID", "SIGNAL", "GLOW", "TERM", "TOOL", "ALGORITHM", "NAUGHTY",
    "NICE", "GROUP", "OPERATOR", "MODE", "COMMENT",
]

ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "gliner_zero_shot": {"path": "ALGOS/gliner_zero_shot_extractor.py", "cost": 0.18, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "minhash": {"path": "ALGOS/minhash.py", "cost": 0.05, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "ternary_router": {"path": "ALGOS/ternary_router.py", "cost": 0.03, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "treelite_date_router": {"path": "03_VAULT/router/treelite_router_v0.tl", "cost": 0.08, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "temporal_motifs": {"path": "ALGOS/temporal_motifs.py", "cost": 0.06, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "decision_hygiene": {"path": "ALGOS/decision_hygiene.py", "cost": 0.04, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "semantic_neighbors": {"path": "ALGOS/semantic_neighbors.py", "cost": 0.07, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
    "needle_classifier": {"path": "scripts/lucidota_needle_worker.py", "cost": 0.12, "vram": 0.05, "engine": "cpu_fairyfuse_ternary"},
    "lora_preemption": {"path": "pypeline/math/model_vram_scheduler.py", "cost": 0.16, "vram": 0.35, "engine": "gpu_q4_deepseek"},
    "possum_filter": {"path": "ALGOS/possum_filter.py", "cost": 0.02, "vram": 0.0, "engine": "cpu_fairyfuse_ternary"},
}

DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b")


@dataclass(frozen=True)
class ReteBanditDecision:
    schema: str
    context_id: str
    algorithm_pool: list[str]
    selected_algorithm: str
    selected_engine: str
    parallel_engine_plan: dict[str, Any]
    bandit_strategy: str
    rule_hits: list[str]
    context_features: dict[str, float]
    execution_status: str
    runtime_ms: float
    facts_yielded: int
    reward: float
    regret_weights: dict[str, float]
    result: dict[str, Any]
    penalty_reason: str | None = None


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_type_from_packet(packet: dict[str, Any]) -> str:
    text = " ".join(str(packet.get(k) or "") for k in ("source_path", "source_ref"))
    payload = packet.get("payload") or {}
    text += " " + str(payload.get("source_path") or payload.get("mime_type") or payload.get("file_type") or "")
    suffix = Path(text.split()[0]).suffix.lower() if text.strip() else ""
    low = text.lower()
    if ".pdf" in low or "application/pdf" in low:
        return "PDF"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff"} or low.startswith("image/"):
        return "IMAGE"
    if suffix in {".json", ".jsonl"} or "json" in low:
        return "JSON"
    if suffix in {".md", ".txt", ".csv", ".log", ".html", ".xml"} or "text/" in low:
        return "TEXT"
    return "BINARY_OR_UNKNOWN"


def friction_from_packet(packet: dict[str, Any]) -> str:
    ca = packet.get("compressed_activity") or {}
    payload = packet.get("payload") or {}
    text = str(packet.get("text_surface") or "").lower()
    explicit = str(payload.get("telemetry_friction") or payload.get("friction") or "").upper()
    if explicit in {"HIGH", "MEDIUM", "LOW"}:
        return explicit
    score = 0.0
    for key in ("friction_score", "error_count", "dead_letter_count", "keystroke_burst", "mouse_delta_sum"):
        try:
            score += float(ca.get(key) or payload.get(key) or 0.0)
        except Exception:
            pass
    if any(w in text for w in ("failed", "dead_letter", "timeout", "lock", "contradiction", "friction")):
        score += 5.0
    if score >= 5.0:
        return "HIGH"
    if score >= 1.0:
        return "MEDIUM"
    return "LOW"


def ontology_set(packet: dict[str, Any]) -> set[str]:
    terms = packet.get("ontology_terms") or []
    out = {str(t) for t in terms if str(t) in GO25}
    text = str(packet.get("text_surface") or "").upper()
    for term in GO25:
        if term in text:
            out.add(term)
    return out or {"SIGNAL", "COMMENT"}


def valid_algorithms(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        meta = ALGORITHM_REGISTRY.get(name)
        if not meta:
            continue
        path = ROOT / str(meta["path"])
        if path.exists() or str(meta["path"]).startswith("03_VAULT/router/treelite"):
            if name not in out:
                out.append(name)
    return out


def engine_for_algorithm(algorithm: str) -> str:
    return str(ALGORITHM_REGISTRY.get(algorithm, {}).get("engine") or "cpu_fairyfuse_ternary")


def parallel_engine_plan(pool: list[str], selected_algorithm: str, packet: dict[str, Any]) -> dict[str, Any]:
    engines = sorted({engine_for_algorithm(a) for a in pool})
    selected_engine = engine_for_algorithm(selected_algorithm)
    text = str(packet.get("text_surface") or "")
    payload = packet.get("payload") or {}
    low = (text + " " + json.dumps(payload, sort_keys=True, default=str)).lower()
    activity = packet.get("compressed_activity") or {}
    mamba_velocity = payload.get("mamba_graph_read_velocity") or payload.get("mamba_read_velocity") or activity.get("mamba_graph_read_velocity") or activity.get("mamba_read_velocity") or packet.get("mamba_graph_read_velocity") or 0
    try:
        mamba_velocity = float(mamba_velocity)
    except Exception:
        mamba_velocity = 0.0
    mamba_maxed = bool(
        payload.get("mamba_graph_read_velocity_maxed")
        or payload.get("mamba_read_velocity_maxed")
        or activity.get("mamba_graph_read_velocity_maxed")
        or activity.get("mamba_read_velocity_maxed")
        or mamba_velocity >= 1.0
    )
    lora_delay_ticks = 2 if mamba_maxed else 0
    return {
        "schema": "lucidota.rete_bandit.dual_engine_plan.v1",
        "selected_engine": selected_engine,
        "available_engines": engines,
        "cpu_fairyfuse_ternary": {
            "always_on": True,
            "role": "continuous semantic stream, fast-negative filters, Treelite hard-date extraction",
            "algorithms": [a for a in pool if engine_for_algorithm(a) == "cpu_fairyfuse_ternary"],
        },
        "gpu_q4_deepseek": {
            "always_on": True,
            "role": "deep synthesis, LoRA hot-swap, abductive validation",
            "algorithms": [a for a in pool if engine_for_algorithm(a) == "gpu_q4_deepseek"],
            "context_reaper_required": selected_engine == "gpu_q4_deepseek",
            "lora_swap_delay_bytewax_ticks": lora_delay_ticks,
            "bus_congestion_rule": "delay_non_critical_lora_swaps_up_to_2_bytewax_ticks_when_mamba_graph_read_velocity_is_maxed",
            "mamba_graph_read_velocity_maxed": mamba_maxed,
            "mamba_graph_read_velocity": mamba_velocity,
        },
        "temporal_fragility_policy": {
            "mtime_snapshot_v1": "256_token_lora_ceiling_and_treelite_date_router",
            "active": "mtime_snapshot_v1" in low,
        },
        "outbound_state": "draft_only",
    }


def rete_prune(packet: dict[str, Any]) -> tuple[list[str], list[str], dict[str, float]]:
    file_type = file_type_from_packet(packet)
    friction = friction_from_packet(packet)
    terms = ontology_set(packet)
    text = str(packet.get("text_surface") or "")
    payload_dump = json.dumps(packet.get("payload") or {}, sort_keys=True, default=str)
    low = (text + " " + payload_dump).lower()
    pool: list[str] = []
    hits: list[str] = []

    def add(rule: str, algos: list[str]) -> None:
        hits.append(rule)
        pool.extend(algos)

    if packet.get("injection_flag") or "prompt-injection" in low or "ignore previous" in low:
        add("prompt_injection_membrane", ["ternary_router", "decision_hygiene", "possum_filter"])
    if file_type == "PDF" and friction == "HIGH":
        add("pdf_high_friction", ["gliner_zero_shot", "minhash", "ternary_router"])
    if "mtime_snapshot_v1" in low:
        add("fragile_mtime_snapshot", ["treelite_date_router", "temporal_motifs", "ternary_router"])
    if {"TIME", "EVENT"} & terms:
        add("time_event_terms", ["treelite_date_router", "temporal_motifs", "minhash"])
    if {"EVIDENCE", "CLAIM", "ENTITY"} & terms:
        add("evidence_claim_entity_terms", ["gliner_zero_shot", "minhash", "semantic_neighbors"])
    if {"FRICTION", "NAUGHTY"} & terms or friction == "HIGH":
        add("friction_or_naughty_terms", ["decision_hygiene", "ternary_router", "possum_filter"])
    if file_type == "IMAGE":
        add("image_artifact", ["semantic_neighbors", "minhash", "ternary_router"])
    if "lora" in low or "adapter" in low or "deepseek" in low:
        add("model_adapter_context", ["lora_preemption", "needle_classifier", "ternary_router"])
    if not pool:
        add("default_signal_comment", ["ternary_router", "minhash", "semantic_neighbors"])

    pool = valid_algorithms(pool)
    if not pool:
        pool = ["ternary_router"]
    features = {
        "is_pdf": 1.0 if file_type == "PDF" else 0.0,
        "is_text": 1.0 if file_type == "TEXT" else 0.0,
        "friction_high": 1.0 if friction == "HIGH" else 0.0,
        "friction_medium": 1.0 if friction == "MEDIUM" else 0.0,
        "term_count": float(len(terms)),
        "text_kchars": min(64.0, len(text) / 1000.0),
        "has_time": 1.0 if "TIME" in terms else 0.0,
        "has_evidence": 1.0 if "EVIDENCE" in terms else 0.0,
        "has_claim": 1.0 if "CLAIM" in terms else 0.0,
    }
    return pool, hits, features


def execute_algorithm(algorithm: str, packet: dict[str, Any]) -> dict[str, Any]:
    text = str(packet.get("text_surface") or "")[:20000]
    payload = packet.get("payload") or {}
    terms = sorted(ontology_set(packet))
    if algorithm == "minhash":
        from ALGOS.minhash import shingles, signature
        sig = signature(shingles(text, width=5), k=64)
        return {"backend": "ALGOS.minhash", "signature_head": sig[:8], "shingle_count": len(shingles(text, width=5)), "fact_markers": int(bool(sig))}
    if algorithm == "gliner_zero_shot":
        from ALGOS.gliner_zero_shot_extractor import literal_fallback
        labels = ["Operator", "Chrono-Ledger", "KRAMPUSCHEWING", "KORPUS", "Command Envelope Protocol", "Evidence", "Claim"]
        spans = literal_fallback(text, labels)
        return {"backend": "ALGOS.gliner_zero_shot_extractor.literal_fallback", "spans": [s.__dict__ for s in spans[:32]], "span_count": len(spans), "fact_markers": len(spans)}
    if algorithm == "treelite_date_router":
        dates = DATE_RE.findall(text)
        artifact = ROOT / "03_VAULT/router/treelite_router_v0.tl"
        return {"backend": "treelite_router_artifact", "artifact": str(artifact.relative_to(ROOT)), "artifact_exists": artifact.exists(), "dates": dates[:32], "date_count": len(dates), "fact_markers": len(dates)}
    if algorithm == "temporal_motifs":
        from ALGOS.temporal_motifs import detect_bursts
        events = [{"type": str(t), "t": i} for i, t in enumerate(terms)]
        bursts = detect_bursts(events)
        return {"backend": "ALGOS.temporal_motifs", "bursts": [b.__dict__ for b in bursts], "fact_markers": len(bursts)}
    if algorithm == "decision_hygiene":
        from ALGOS.decision_hygiene import counts, score_features
        c = counts(text)
        score, label = score_features(c)
        return {"backend": "ALGOS.decision_hygiene", "counts": c, "score": score, "label": label, "fact_markers": c.get("evidence_count", 0) + c.get("outcome_count", 0)}
    if algorithm == "ternary_router":
        from ALGOS.ternary_router import route_packet
        routed = route_packet({**packet, "text_surface": text[:4096], "ontology_terms": terms})
        return {"backend": "ALGOS.ternary_router", "engine_channel": "cpu_fairyfuse_ternary", "route": routed, "fact_markers": 1 if terms else 0}
    if algorithm == "semantic_neighbors":
        toks = {t.lower() for t in re.findall(r"[A-Za-z0-9_]{4,}", text)}
        overlap = sorted(toks & {"evidence", "claim", "timeline", "graph", "event", "fact", "receipt", "source"})
        return {"backend": "ALGOS.semantic_neighbors.lightweight_overlap", "overlap": overlap, "token_count": len(toks), "fact_markers": len(overlap)}
    if algorithm == "needle_classifier":
        model = ROOT / "03_VAULT/models/needle/needle.pkl"
        return {"backend": "needle_presence_probe", "model_path": str(model.relative_to(ROOT)), "model_exists": model.exists(), "fact_markers": 1 if model.exists() else 0}
    if algorithm == "lora_preemption":
        from pypeline.math.model_vram_scheduler import context_reaper_flush, plan_dual_engine_residency, plan_lora_preemption
        plan = plan_lora_preemption(payload, {"source_ref": packet.get("source_ref")}, include_gpu=True)
        dual = plan.get("dual_engine_residency") or plan_dual_engine_residency(payload, {"source_ref": packet.get("source_ref")}, include_gpu=True)
        bus_delay = int((packet.get("parallel_engine_plan") or {}).get("gpu_q4_deepseek", {}).get("lora_swap_delay_bytewax_ticks") or (payload.get("mamba_graph_read_velocity_maxed") and 2) or 0)
        if bus_delay <= 0:
            bus_delay = 2 if bool(payload.get("mamba_graph_read_velocity_maxed") or payload.get("mamba_read_velocity_maxed")) else 0
        reaper = {"status": "not_required_until_after_synthesis_pass"}
        if payload.get("gpu_synthesis_completed"):
            reaper = context_reaper_flush(reason="rete_bandit_gpu_synthesis_completed")
        return {
            "backend": "pypeline.math.model_vram_scheduler",
            "engine_channel": "gpu_q4_deepseek",
            "adapter_requested": bool(payload.get("adapter_id") or payload.get("lora_adapter_id")),
            "lora_swap_delay_bytewax_ticks": bus_delay,
            "bus_congestion_rule": "delay_non_critical_lora_swaps_up_to_2_bytewax_ticks_when_mamba_graph_read_velocity_is_maxed",
            "plan": plan,
            "dual_engine_residency": dual,
            "context_reaper": reaper,
            "fact_markers": 1 if payload.get("adapter_id") or payload.get("lora_adapter_id") or plan.get("decision") == "allow" else 0,
            "vram_penalty": ALGORITHM_REGISTRY[algorithm]["vram"],
        }
    if algorithm == "possum_filter":
        risky = bool(re.search(r"\b(?:ignore previous|sudo|rm -rf|exfiltrate|steal)\b", text, re.I))
        return {"backend": "ALGOS.possum_filter.fast_negative", "risk_detected": risky, "fact_markers": 1 if risky else 0}
    return {"backend": "unsupported", "fact_markers": 0}


def calculate_reward(packet: dict[str, Any], result: dict[str, Any], runtime_ms: float, algorithm: str) -> tuple[int, float, str | None]:
    base_facts = int(result.get("fact_markers") or 0)
    if packet.get("epistemic_flag") == "FACT":
        base_facts += 1
    duration_penalty = runtime_ms / 250.0
    vram_penalty = float(result.get("vram_penalty") or ALGORITHM_REGISTRY.get(algorithm, {}).get("vram") or 0.0)
    engine_penalty = 0.04 if engine_for_algorithm(algorithm) == "gpu_q4_deepseek" else 0.005
    reward = float(base_facts) - duration_penalty - vram_penalty - engine_penalty
    reason = None
    if base_facts <= 0:
        reason = "zero_fact_yield"
    elif duration_penalty > 1.0:
        reason = "cpu_duration_penalty"
    elif vram_penalty:
        reason = "vram_residency_penalty"
    return base_facts, round(reward, 6), reason


def apply_rete_bandit(packet: dict[str, Any], *, strategy: str = "linucb", epsilon: float = 0.08, seed: int | str | None = None) -> dict[str, Any]:
    pool, rule_hits, features = rete_prune(packet)
    context_id = stable_hash({"source": packet.get("source"), "source_ref": packet.get("source_ref"), "features": features, "rules": rule_hits})[:24]
    action = select_action(features, pool, algorithm=strategy, epsilon=epsilon, seed=seed or context_id)
    started = time.perf_counter()
    status = "succeeded"
    result: dict[str, Any]
    try:
        result = execute_algorithm(action.action_id, packet)
    except Exception as exc:
        status = "failed"
        result = {"backend": "exception", "error": f"{type(exc).__name__}: {exc}", "fact_markers": 0}
    runtime_ms = round((time.perf_counter() - started) * 1000.0, 3)
    facts, reward, penalty_reason = calculate_reward(packet, result, runtime_ms, action.action_id)
    selected_engine = engine_for_algorithm(action.action_id)
    engine_plan = parallel_engine_plan(pool, action.action_id, packet)
    regret_actions = [MathAction(id=a, expected_value=1.0 if a == action.action_id else 0.5, cost=float(ALGORITHM_REGISTRY[a]["cost"]), risk=float(ALGORITHM_REGISTRY[a]["vram"]) + (0.04 if engine_for_algorithm(a) == "gpu_q4_deepseek" else 0.005)) for a in pool]
    regret_weights = compute_regret_weighted_strategy(regret_actions, [MathCounterfactual(action.action_id, reward, 1.0)])
    update_policy([BanditUpdate(context_id=context_id, action_id=action.action_id, reward=reward, propensity=action.propensity)])
    decision = ReteBanditDecision(
        schema=SCHEMA_VERSION,
        context_id=context_id,
        algorithm_pool=pool,
        selected_algorithm=action.action_id,
        selected_engine=selected_engine,
        parallel_engine_plan=engine_plan,
        bandit_strategy=strategy,
        rule_hits=rule_hits,
        context_features=features,
        execution_status=status,
        runtime_ms=runtime_ms,
        facts_yielded=facts,
        reward=reward,
        regret_weights={k: round(float(v), 8) for k, v in regret_weights.items()},
        result=result,
        penalty_reason=penalty_reason,
    )
    return asdict(decision)


def bandit_update_from_decision(decision: dict[str, Any]) -> BanditUpdate:
    return BanditUpdate(
        context_id=str(decision["context_id"]),
        action_id=str(decision["selected_algorithm"]),
        reward=float(decision.get("reward") or 0.0),
        propensity=1.0 / max(1, len(decision.get("algorithm_pool") or [])),
    )
