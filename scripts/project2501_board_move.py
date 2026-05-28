#!/usr/bin/env python3
"""Project 2501 board-move pipeline: EventEnvelope -> route -> WorkReceipt -> River row."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.board_effect_doctrine import evaluate_board_effect

OUT = ROOT / "05_OUTPUTS" / "project2501_board_moves"
SCHEMA = ROOT / "06_SCHEMA" / "112_project2501_core_board.sql"

ACTORS = {"operator", "codex", "groq", "cohere", "local_model", "scraper", "auditor", "worker", "system", "external", "unknown"}
LANES = {"fast", "slow", "audit", "external", "dead_letter"}
ACTION_RE = re.compile(r"\b(patch|fix|verify|test|run|ingest|audit|classify|route|write|stage|promote|materialize|scrape|call|read|summarize)\b", re.I)
PATH_RE = re.compile(r"(?:(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+|[A-Za-z0-9_.-]+\.(?:py|sql|md|json|txt|yaml|yml|toml))")
URL_RE = re.compile(r"https?://\S+")
RISK_WORDS = {"delete", "destructive", "secret", "token", "key", "external", "send", "purchase", "legal", "canonical", "materialize"}
MUTATION_WORDS = {"patch", "fix", "write", "edit", "mutate", "delete", "materialize", "promote", "ingest", "stage"}
LOCAL_FILE_WORDS = {"repo", "file", "script", "schema", "test", "pytest", "path", "krampus", "storage", "corpus"}
CLOUD_WORDS = {"groq", "cohere", "openai", "cloud", "external model", "api"}
GRAPH_WORDS = {"graph write", "canonical", "materialize", "graph mutation", "promote graph", "graph-promotion"}
EXTERNAL_WORDS = {"http://", "https://", "scrape", "github", "gitlab", "email", "web", "api"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8", errors="replace"))


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def read_text_arg(value: str) -> str:
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_absolute():
            path = ROOT / path
        return path.read_text(encoding="utf-8", errors="replace")
    return value


def db_url(args: argparse.Namespace | None = None) -> str:
    return (
        (getattr(args, "database_url", None) if args is not None else None)
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def redacted(url: str) -> str:
    if url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in url:
        return "postgresql://<redacted>@" + url.split("@", 1)[1]
    return "set_redacted"


def token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def hits(text: str, words: set[str]) -> list[str]:
    low = text.lower()
    return sorted(w for w in words if w in low)


def extract_entities(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in URL_RE.finditer(text):
        rows.append({"kind": "url", "text": m.group(0), "start": m.start(), "end": m.end()})
    for m in PATH_RE.finditer(text):
        rows.append({"kind": "artifact_path", "text": m.group(0), "start": m.start(), "end": m.end()})
    return rows


def extract_actions(text: str) -> list[str]:
    return sorted({m.group(1).lower() for m in ACTION_RE.finditer(text)})


def extract_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if re.search(r"\b(is|are|was|were|must|should|needs?|has|have)\b", sentence, re.I):
            claims.append({"text": sentence[:500], "authority_class": "operator_authored_assertion"})
    return claims[:20]


def board_features(text: str) -> dict[str, Any]:
    low = text.lower()
    tc = token_count(text)
    mutation = bool(hits(low, MUTATION_WORDS))
    needs_graph = any(w in low for w in GRAPH_WORDS)
    risk = hits(low, RISK_WORDS)
    features = {
        "token_count": tc,
        "urgency": min(1.0, (low.count("now") + low.count("fast") + low.count("urgent") + low.count("fuck")) / 6.0),
        "mutation_requested": mutation,
        "needs_local_files": bool(hits(low, LOCAL_FILE_WORDS)),
        "needs_cloud_model": bool(hits(low, CLOUD_WORDS)),
        "needs_graph_write": needs_graph,
        "needs_external": bool(hits(low, EXTERNAL_WORDS)),
        "can_fast_lane": tc <= 80 and not mutation and not needs_graph,
        "expected_latency_ms": 35 if tc <= 80 else min(5000, 35 + tc * 8),
        "expected_gain": round(min(0.95, 0.15 + len(extract_actions(text)) * 0.08 + (0.22 if mutation else 0.0)), 3),
        "risk_of_slop": round(min(1.0, 0.05 + len(risk) * 0.12 + (0.18 if tc > 500 else 0.0)), 3),
        "risk_terms": risk,
    }
    return features


def build_event_envelope(*, actor: str, source: str, text: str, ts: str | None = None) -> dict[str, Any]:
    actor = actor if actor in ACTORS else "unknown"
    ts = ts or now()
    text_hash = sha256_text(text)
    event_basis = stable_json({"actor": actor, "source": source, "ts": ts, "text_hash": text_hash})
    features = board_features(text)
    entities = extract_entities(text)
    return {
        "schema": "lucidota.project2501.event_envelope.v1",
        "event_id": sha256_text(event_basis),
        "ts": ts,
        "source": source,
        "actor": actor,
        "raw_ref": "inline://" + text_hash[:24],
        "verbatim_hash": text_hash,
        "hash_algo": "sha256",
        "text": text,
        "entities": entities,
        "claims": extract_claims(text),
        "actions_requested": extract_actions(text),
        "artifacts_referenced": [e["text"] for e in entities if e["kind"] == "artifact_path"],
        "risk_flags": features["risk_terms"],
        "route_candidates": ["fast", "slow", "audit", "external", "dead_letter"],
        "board_features": features,
        "embedding_ref": "pending:pgvector_or_local_embedder",
        "detail": {"contract": "00_PROJECT_BRAIN/PROJECT_2501_CORE_CONTRACT.md"},
    }


def treelite_gate(features: dict[str, Any]) -> dict[str, Any]:
    vector = [
        min(float(features.get("token_count") or 0) / 1000.0, 1.0),
        1.0 if features.get("mutation_requested") else 0.0,
        1.0 if features.get("needs_cloud_model") else 0.0,
        1.0 if features.get("needs_graph_write") else 0.0,
        float(features.get("risk_of_slop") or 0.0),
    ]
    try:
        import numpy as np
        from treelite import gtil, model_builder as mb

        meta = mb.Metadata(num_feature=5, task_type="kRegressor", average_tree_output=False, num_target=1, num_class=[1], leaf_vector_shape=(1, 1))
        ann = mb.TreeAnnotation(num_tree=1, target_id=[0], class_id=[0])
        builder = mb.ModelBuilder(
            threshold_type="float32",
            leaf_output_type="float32",
            metadata=meta,
            tree_annotation=ann,
            postprocessor=mb.PostProcessorFunc(name="identity"),
            base_scores=[0.0],
        )
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(3, 0.5, default_left=True, opname="<", left_child_key=1, right_child_key=2)
        builder.end_node()
        builder.start_node(1)
        builder.numerical_test(1, 0.5, default_left=True, opname="<", left_child_key=3, right_child_key=4)
        builder.end_node()
        builder.start_node(2); builder.leaf(0.95); builder.end_node()
        builder.start_node(3); builder.leaf(0.12); builder.end_node()
        builder.start_node(4); builder.leaf(0.66); builder.end_node()
        builder.end_tree()
        model = builder.commit()
        score = float(gtil.predict(model, np.array([vector], dtype=np.float32)).reshape(-1)[0])
        return {
            "available": True,
            "gate": "route_lane",
            "model": "inline_treelite_single_tree:route_lane_v001",
            "score": round(score, 4),
            "features": vector,
            "verdict": "audit_bias" if score >= 0.9 else ("slow_bias" if score >= 0.55 else "fast_bias"),
        }
    except Exception as exc:
        return {"available": False, "gate": "route_lane", "error": f"{type(exc).__name__}:{exc}", "features": vector, "verdict": "rules_only"}


def route_event(envelope: dict[str, Any]) -> dict[str, Any]:
    features = envelope["board_features"]
    treelite = treelite_gate(features)
    reasons: list[str] = []
    if features.get("needs_external"):
        lane = "external"; reasons.append("deterministic:external_signal")
    elif features.get("needs_graph_write"):
        lane = "audit"; reasons.append("deterministic:graph_write_gate_required")
    elif features.get("mutation_requested") or features.get("needs_local_files") or features.get("needs_cloud_model"):
        lane = "slow"; reasons.append("deterministic:durable_work_required")
    elif features.get("can_fast_lane"):
        lane = "fast"; reasons.append("deterministic:fast_lane_safe")
    else:
        lane = "slow"; reasons.append("deterministic:default_slow")
    if treelite.get("available") and treelite.get("verdict") == "audit_bias" and lane == "slow":
        lane = "audit"; reasons.append("treelite:audit_bias")
    route_key = f"{lane}:{envelope['source']}:{envelope['actor']}"
    graph_mode = "staged_only" if features.get("needs_graph_write") else "none"
    return {
        "schema": "lucidota.project2501.route_decision.v1",
        "event_id": envelope["event_id"],
        "lane": lane,
        "route_key": route_key,
        "gate_order": ["deterministic_rules", "treelite_gate", "model_fallback"],
        "deterministic_rule": {"reasons": reasons, "features": features},
        "treelite_gate": treelite,
        "model_fallback": {"used": False, "reason": "deterministic_and_treelite_gates_sufficient"},
        "cost": {
            "tokens": features.get("token_count"),
            "time_ms_estimate": features.get("expected_latency_ms"),
            "cpu": "low",
            "vram": 0,
            "risk": features.get("risk_of_slop"),
            "graph_mutation": bool(features.get("needs_graph_write")),
        },
        "expected_gain": features.get("expected_gain"),
        "confidence": round(0.82 if lane in {"fast", "audit"} else 0.74, 3),
        "graph_write_mode": graph_mode,
        "detail": {"big_board_feature_change_allowed": False, "operator_feature_authority_required": True},
    }


def build_work_receipt(envelope: dict[str, Any], decision: dict[str, Any], *, execute: bool, latency_ms: float, effect_gate: dict[str, Any] | None = None, receipt_path: str = "") -> dict[str, Any]:
    verdict = "win" if execute else "stall"
    gain = {
        "proof": 1 if execute else 0,
        "compression": max(0, len(envelope["text"]) - len(stable_json(envelope["board_features"]))),
        "routing_accuracy": decision["confidence"],
        "artifact": "event_envelope+route_decision+river_training_row",
        "fixed_code": False,
        "reduced_entropy": round(float(decision["expected_gain"]) * (1.0 if execute else 0.4), 3),
    }
    return {
        "schema": "lucidota.project2501.work_receipt.v1",
        "event_id": envelope["event_id"],
        "receipt_path": receipt_path,
        "verdict": verdict,
        "cost": {**decision["cost"], "latency_ms": round(latency_ms, 3)},
        "gain": gain,
        "artifact_refs": [envelope["raw_ref"], "06_SCHEMA/112_project2501_core_board.sql"],
        "canonical_graph_writes_performed": False,
        "graph_write_mode": decision["graph_write_mode"],
        "detail": {"ui_tile_source": "receipt_or_db_only", "big_board_metric_operator_requested": True, "effect_gate": effect_gate or {}},
    }


def build_river_training_row(envelope: dict[str, Any], decision: dict[str, Any], receipt: dict[str, Any], *, latency_ms: float, effect_gate: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema": "lucidota.project2501.river_training_row.v1",
        "event_id": envelope["event_id"],
        "route_chosen": decision["lane"],
        "model_used": "none",
        "estimated_gain": decision["expected_gain"],
        "actual_gain": receipt["gain"]["reduced_entropy"],
        "latency_ms": round(latency_ms, 3),
        "tokens_in": int(envelope["board_features"]["token_count"]),
        "tokens_out": 0,
        "verdict": receipt["verdict"],
        "effect_gate": effect_gate,
        "human_override": False,
        "features": envelope["board_features"],
        "label": {"route_verdict": receipt["verdict"], "lane": decision["lane"]},
    }


def build_board_move(*, actor: str, source: str, text: str, execute: bool, position: str = "current_board") -> dict[str, Any]:
    started = time.perf_counter()
    envelope = build_event_envelope(actor=actor, source=source, text=text)
    decision = route_event(envelope)
    latency_ms = (time.perf_counter() - started) * 1000
    effect_gate = evaluate_board_effect(text=text, persona=None, evidence_refs=envelope["artifacts_referenced"], explicit_effects=None)
    receipt = build_work_receipt(envelope, decision, execute=execute, latency_ms=latency_ms, effect_gate=effect_gate)
    river = build_river_training_row(envelope, decision, receipt, latency_ms=latency_ms, effect_gate=effect_gate)
    board_position = {
        "schema": "lucidota.project2501.board_position.v1",
        "position_key": position,
        "event_id": envelope["event_id"],
        "feature_snapshot": envelope["board_features"],
        "resource_snapshot": {"tokens_in": river["tokens_in"], "vram_mb": 0, "model_calls": 0},
        "graph_authority_snapshot": {"canonical_graph_write_allowed": False, "graph_write_mode": decision["graph_write_mode"]},
        "effect_gate": effect_gate,
    }
    board_move = {
        "schema": "lucidota.project2501.board_move.v1",
        "actor": envelope["actor"],
        "position": position,
        "move": "normalize_route_receipt",
        "lane": decision["lane"],
        "cost": receipt["cost"],
        "gain": receipt["gain"],
        "receipt": receipt["receipt_path"],
        "verdict": receipt["verdict"],
        "effect_gate": effect_gate,
    }
    return {
        "schema": "lucidota.project2501.board_move_bundle.v1",
        "event_envelope": envelope,
        "route_decision": decision,
        "work_receipt": receipt,
        "river_training_row": river,
        "board_position": board_position,
        "board_move": board_move,
        "canonical_graph_writes_performed": False,
    }


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"project2501_board_move_{stamp()}.json"
    payload["report_path"] = rel(path)
    if isinstance(payload.get("work_receipt"), dict):
        payload["work_receipt"]["receipt_path"] = rel(path)
    if isinstance(payload.get("board_move"), dict):
        payload["board_move"]["receipt"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print("REPORT_PATH=" + rel(path))
    return path


def apply_schema(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
        conn.commit()


def persist_bundle(bundle: dict[str, Any], database_url: str) -> dict[str, Any]:
    e = bundle["event_envelope"]; d = bundle["route_decision"]; r = bundle["work_receipt"]; row = bundle["river_training_row"]; bp = bundle["board_position"]; bm = bundle["board_move"]
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
            cur.execute(
                """
                INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, detail)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                ON CONFLICT(raw_ref) DO UPDATE SET detail=EXCLUDED.detail
                RETURNING raw_artifact_uuid::text
                """,
                (e["raw_ref"], e["verbatim_hash"], e["hash_algo"], e["source"], e["actor"], len(e["text"].encode()), len(e["text"]), json.dumps({"source":"project2501_board_move"})),
            )
            raw_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
                VALUES (%s,%s::timestamptz,%s,%s,%s,%s::uuid,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s::jsonb)
                ON CONFLICT(event_id) DO UPDATE SET detail=EXCLUDED.detail
                RETURNING envelope_uuid::text
                """,
                (e["event_id"], e["ts"], e["source"], e["actor"], e["raw_ref"], raw_uuid, e["verbatim_hash"], e["hash_algo"], e["text"], json.dumps(e["entities"]), json.dumps(e["claims"]), json.dumps(e["actions_requested"]), json.dumps(e["artifacts_referenced"]), json.dumps(e["risk_flags"]), json.dumps(e["route_candidates"]), json.dumps(e["board_features"]), e["embedding_ref"], json.dumps(e["detail"])),
            )
            envelope_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.route_decision(event_id, lane, route_key, gate_order, deterministic_rule, treelite_gate, model_fallback, cost, expected_gain, confidence, graph_write_mode, detail)
                VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s::jsonb)
                RETURNING decision_uuid::text
                """,
                (e["event_id"], d["lane"], d["route_key"], json.dumps(d["gate_order"]), json.dumps(d["deterministic_rule"]), json.dumps(d["treelite_gate"]), json.dumps(d["model_fallback"]), json.dumps(d["cost"]), d["expected_gain"], d["confidence"], d["graph_write_mode"], json.dumps(d["detail"])),
            )
            decision_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.board_position(position_key, event_id, feature_snapshot, resource_snapshot, graph_authority_snapshot)
                VALUES (%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                ON CONFLICT(position_key, event_id) DO UPDATE SET feature_snapshot=EXCLUDED.feature_snapshot
                RETURNING position_uuid::text
                """,
                (bp["position_key"], e["event_id"], json.dumps(bp["feature_snapshot"]), json.dumps(bp["resource_snapshot"]), json.dumps(bp["graph_authority_snapshot"])),
            )
            position_uuid = cur.fetchone()[0]
            idem = sha256_text(stable_json({"event_id": e["event_id"], "lane": d["lane"], "move": bm["move"]}))
            cur.execute(
                """
                INSERT INTO lucidota_control.work_order(event_id, decision_uuid, lane, work_kind, status, payload, idempotency_key)
                VALUES (%s,%s::uuid,%s,'normalize_route_receipt','succeeded',%s::jsonb,%s)
                ON CONFLICT(idempotency_key) DO UPDATE SET updated_at=now()
                RETURNING work_order_uuid::text
                """,
                (e["event_id"], decision_uuid, d["lane"], json.dumps({"raw_ref": e["raw_ref"], "route_key": d["route_key"]}), idem),
            )
            work_order_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.work_receipt(event_id, decision_uuid, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
                VALUES (%s,%s::uuid,%s::uuid,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,false,%s,%s::jsonb)
                RETURNING work_receipt_uuid::text
                """,
                (e["event_id"], decision_uuid, work_order_uuid, r["receipt_path"], sha256_text(stable_json(bundle)), r["verdict"], json.dumps(r["cost"]), json.dumps(r["gain"]), json.dumps(r["artifact_refs"]), r["graph_write_mode"], json.dumps(r["detail"])),
            )
            receipt_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.board_move(event_id, position_uuid, actor, position, move, lane, cost, gain, receipt, verdict)
                VALUES (%s,%s::uuid,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s)
                RETURNING move_uuid::text
                """,
                (e["event_id"], position_uuid, bm["actor"], bm["position"], bm["move"], bm["lane"], json.dumps(bm["cost"]), json.dumps(bm["gain"]), bm["receipt"], bm["verdict"]),
            )
            move_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.river_training_row(event_id, decision_uuid, work_receipt_uuid, route_chosen, model_used, estimated_gain, actual_gain, latency_ms, tokens_in, tokens_out, verdict, human_override, features, label)
                VALUES (%s,%s::uuid,%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,false,%s::jsonb,%s::jsonb)
                RETURNING training_uuid::text
                """,
                (e["event_id"], decision_uuid, receipt_uuid, row["route_chosen"], row["model_used"], row["estimated_gain"], row["actual_gain"], row["latency_ms"], row["tokens_in"], row["tokens_out"], row["verdict"], json.dumps(row["features"]), json.dumps(row["label"])),
            )
            training_uuid = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_control.watch_metric(metric_key, metric_value, source_receipt, source_db_ref, operator_requested)
                VALUES ('project2501_board_move_ingest', %s::jsonb, %s, %s, true)
                """,
                (json.dumps({"lane": d["lane"], "event_id": e["event_id"], "verdict": r["verdict"]}), r["receipt_path"], f"lucidota_control.event_envelope:{e['event_id']}"),
            )
        conn.commit()
    return {
        "raw_artifact_uuid": raw_uuid,
        "envelope_uuid": envelope_uuid,
        "decision_uuid": decision_uuid,
        "position_uuid": position_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": receipt_uuid,
        "move_uuid": move_uuid,
        "training_uuid": training_uuid,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Project 2501 board-move core pipeline.")
    ap.add_argument("--database-url")
    sub = ap.add_subparsers(dest="cmd", required=True)
    init = sub.add_parser("init-schema")
    init.add_argument("--execute", action="store_true")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--actor", default="operator")
    ingest.add_argument("--source", default="operator_chat")
    ingest.add_argument("--text", required=True, help="Raw text or @path")
    ingest.add_argument("--position", default="current_board")
    ingest.add_argument("--execute", action="store_true")
    ingest.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "init-schema":
        payload = {"schema": "lucidota.project2501.board_move.init_schema.v1", "generated_at": now(), "execute_performed": bool(args.execute), "database_url": redacted(db_url(args)), "schema_path": rel(SCHEMA), "status": "PASS"}
        if args.execute:
            apply_schema(db_url(args))
        path = write_report({"schema": "lucidota.project2501.board_move.report.v1", "status": "PASS", "execute_performed": bool(args.execute), "init_schema": payload, "event_envelope": {}, "route_decision": {}, "work_receipt": {}, "river_training_row": {}, "board_position": {}, "board_move": {}, "canonical_graph_writes_performed": False})
        print("PROJECT2501_BOARD_SCHEMA=PASS")
        return 0
    text = read_text_arg(args.text)
    bundle = build_board_move(actor=args.actor, source=args.source, text=text, execute=bool(args.execute), position=args.position)
    bundle.update({"generated_at": now(), "execute_performed": bool(args.execute), "database_url": redacted(db_url(args)), "status": "PASS", "db_rows": {}})
    path = write_report(bundle)
    if args.execute:
        try:
            bundle["db_rows"] = persist_bundle(bundle, db_url(args))
            # Rewrite receipt after DB UUIDs exist.
            path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str), encoding="utf-8")
        except Exception as exc:
            bundle["status"] = "FAIL"
            bundle["error"] = f"{type(exc).__name__}:{exc}"
            path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if args.json:
        print(json.dumps(bundle, sort_keys=True, default=str))
    print("PROJECT2501_BOARD_MOVE=" + bundle["status"])
    return 0 if bundle["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
