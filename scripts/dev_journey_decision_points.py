#!/usr/bin/env python3
"""Compile dev-journey decision points into sticker + XGBoost/Treelite training rows.

This is advisory ML substrate only. It does not declare truth and does not write
canonical graph state. It turns receipts/logs/hunches into repeatable weak labels
so future gates can learn where to explore, exploit, audit, route, or stop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "dev_journey_decision_points"
OBS = ROOT / "04_RUNTIME" / "observation_center" / "dev_journey_decision_points_latest.json"
BIG_BOARD = ROOT / "05_OUTPUTS" / "big_board.json"
sys.path.insert(0, str(ROOT))

from scripts.board_effect_doctrine import extract_board_effects  # noqa: E402
from scripts.sticker_feature_extractor_v1 import all_features, vector_columns  # noqa: E402

FEATURE_COLUMNS = [
    "chars",
    "words",
    "imperative_terms",
    "imagination_terms",
    "audit_terms",
    "graph_terms",
    "model_terms",
    "workflow_terms",
    "receipt_terms",
    "slop_terms",
    "test_terms",
    "hedge_terms",
    "board_effect_count",
    "operator_directive_ratio",
    "operator_ledger_density",
    "operator_tech_ratio",
    "resilience_swarm_orchestration_density",
    "telemetry_agent_symmetry_ratio",
    "telemetry_protocol_discipline",
    "telemetry_manic_velocity",
]

LABELS = ["EXPLORE_WORKFLOW", "EXPLOIT_BUILD", "AUDIT_PROOF", "MODEL_ROUTE", "GRAPH_STAGE", "STOP_OR_QUARANTINE"]

IMAGINE_RE = re.compile(r"\b(imagin(?:e|ing|ation)|different ways|new workflows?|try new|throw shit|spaghetti|invisible wall|explore|experiment)\b", re.I)
AUDIT_RE = re.compile(r"\b(audit|proof|receipt|verify|evidence|truth|journalist|analyst|receipts?)\b", re.I)
BUILD_RE = re.compile(r"\b(build|fix|patch|ship|implement|code|test|run|devcycle|workflow)\b", re.I)
GRAPH_RE = re.compile(r"\b(graph|edge|node|promotion|staging|ontology|postgres|pg|db)\b", re.I)
MODEL_RE = re.compile(r"\b(groq|cohere|local models?|llm|swarm|agent|codex|model fabric|treelite|xgboost)\b", re.I)
STOP_RE = re.compile(r"\b(kill|quarantine|corpse|dead|slop|poison|wrong|fail|exterminate)\b", re.I)
HEDGE_RE = re.compile(r"\b(dunno|maybe|might|perhaps|guess|imperfect|weak signal|hunch)\b", re.I)
RECEIPT_RE = re.compile(r"\b(receipt|ledger|hash|json|05_outputs|db row|path)\b", re.I)
IMPERATIVE_RE = re.compile(r"\b(go|do|make|build|wire|run|audit|print|stop|keep|send|ingest|fix|learn|grow)\b", re.I)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def count_rx(rx: re.Pattern[str], text: str) -> int:
    return len(rx.findall(text or ""))


def label_decision_point(text: str) -> dict[str, Any]:
    """Deterministically weak-label one dev decision point."""
    imagine = count_rx(IMAGINE_RE, text)
    audit = count_rx(AUDIT_RE, text)
    build = count_rx(BUILD_RE, text)
    graph = count_rx(GRAPH_RE, text)
    model = count_rx(MODEL_RE, text)
    stop = count_rx(STOP_RE, text)
    if imagine:
        route = "EXPLORE_WORKFLOW"
    elif stop:
        route = "STOP_OR_QUARANTINE"
    elif graph:
        route = "GRAPH_STAGE"
    elif audit:
        route = "AUDIT_PROOF"
    elif model:
        route = "MODEL_ROUTE"
    elif build:
        route = "EXPLOIT_BUILD"
    else:
        route = "AUDIT_PROOF"
    return {
        "route_class": route,
        "workflow_exploration_pressure": 1.0 if imagine else min(1.0, count_rx(HEDGE_RE, text) / 3.0),
        "audit_pressure": min(1.0, (audit + stop) / 4.0),
        "graph_pressure": min(1.0, graph / 4.0),
        "model_pressure": min(1.0, model / 4.0),
        "build_pressure": min(1.0, build / 4.0),
        "truth_status": "deterministic_label_candidate_only",
    }


def extract_decision_points_from_text(text: str, *, source_path: str, max_points: int = 500) -> list[dict[str, Any]]:
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", text or "") if c.strip()]
    if not chunks and text.strip():
        chunks = [text.strip()]
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks[:max_points], start=1):
        rows.append({
            "schema": "lucidota.dev_journey.decision_point.raw.v1",
            "source_path": source_path,
            "ordinal": idx,
            "text": chunk[:4000],
            "text_sha256": sha256_text(chunk),
        })
    return rows


def numeric_features(text: str, board_effects: list[str], sticker: dict[str, float]) -> dict[str, float]:
    words = len(re.findall(r"\S+", text))
    return {
        "chars": float(len(text)),
        "words": float(words),
        "imperative_terms": float(count_rx(IMPERATIVE_RE, text)),
        "imagination_terms": float(count_rx(IMAGINE_RE, text)),
        "audit_terms": float(count_rx(AUDIT_RE, text)),
        "graph_terms": float(count_rx(GRAPH_RE, text)),
        "model_terms": float(count_rx(MODEL_RE, text)),
        "workflow_terms": float(len(re.findall(r"\b(workflow|lane|method|protocol|process)\b", text, re.I))),
        "receipt_terms": float(count_rx(RECEIPT_RE, text)),
        "slop_terms": float(count_rx(STOP_RE, text)),
        "test_terms": float(len(re.findall(r"\b(test|pytest|verify|smoke|regression)\b", text, re.I))),
        "hedge_terms": float(count_rx(HEDGE_RE, text)),
        "board_effect_count": float(len(board_effects)),
        "operator_directive_ratio": float(sticker.get("operator_directive_ratio", 0.0)),
        "operator_ledger_density": float(sticker.get("operator_ledger_density", 0.0)),
        "operator_tech_ratio": float(sticker.get("operator_tech_ratio", 0.0)),
        "resilience_swarm_orchestration_density": float(sticker.get("resilience_swarm_orchestration_density", 0.0)),
        "telemetry_agent_symmetry_ratio": float(sticker.get("telemetry_agent_symmetry_ratio", 0.0)),
        "telemetry_protocol_discipline": float(sticker.get("telemetry_protocol_discipline", 0.0)),
        "telemetry_manic_velocity": float(sticker.get("telemetry_manic_velocity", 0.0)),
    }


def source_kind(path: Path | str) -> str:
    text = str(path).lower()
    name = Path(path).name.lower()
    if name == "goal_log.md":
        return "goal_log"
    if name == "current_handoff.md":
        return "current_handoff"
    if name == "status_ledger.md":
        return "status_ledger"
    if name == "script_audit_manifest.jsonl":
        return "script_audit_manifest"
    if "hunch" in text:
        return "hunch_hypertimeline"
    if "krampuschewing_normalized_index" in text:
        return "krampuschewing_index"
    if "krampuschewing_river_training" in text:
        return "krampuschewing_river"
    if "project2501_board_move" in text:
        return "board_move"
    if name.endswith(".jsonl"):
        return "jsonl_receipts"
    if name.endswith(".json"):
        return "json_receipt"
    return "document"


def read_source_points(path: Path, max_points: int) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file() or max_points <= 0:
        return []
    rp = rel(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, line in enumerate(fh, start=1):
                if len(rows) >= max_points:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    chunk = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
                except Exception:
                    chunk = raw
                rows.append({
                    "schema": "lucidota.dev_journey.decision_point.raw.v1",
                    "source_path": rp,
                    "ordinal": idx,
                    "text": chunk[:4000],
                    "text_sha256": sha256_text(chunk),
                })
        return rows
    text = path.read_text(encoding="utf-8", errors="replace")
    return extract_decision_points_from_text(text, source_path=rp, max_points=max_points)


def point_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    body = raw["text"]
    sticker_all = all_features(body)
    sticker_vec = vector_columns(sticker_all)
    effects = extract_board_effects(body)
    labels = label_decision_point(body)
    feats = numeric_features(body, effects, sticker_all)
    return {
        "schema": "lucidota.dev_journey.decision_point.v1",
        "decision_id": "djdp_" + sha256_text(f"{raw['source_path']}\0{raw['ordinal']}\0{raw['text_sha256']}")[:24],
        "source_path": raw["source_path"],
        "source_kind": source_kind(raw["source_path"]),
        "ordinal": raw["ordinal"],
        "text_sha256": raw["text_sha256"],
        "text_excerpt": body[:500],
        "board_effects": effects,
        "sticker_features": sticker_vec,
        "training_features": feats,
        "labels": labels,
        "model_authority": "advisory_training_candidate",
        "canonical_graph_writes_performed": False,
    }


def compile_decision_points(source_paths: Iterable[Path], max_points: int = 500) -> list[dict[str, Any]]:
    sources = [Path(s) for s in source_paths]
    raw_by_source = [read_source_points(path, max_points) for path in sources]
    points: list[dict[str, Any]] = []
    max_len = max((len(rows) for rows in raw_by_source), default=0)
    for offset in range(max_len):
        for rows in raw_by_source:
            if len(points) >= max_points:
                return points
            if offset < len(rows):
                points.append(point_from_raw(rows[offset]))
    return points


def matrix(points: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if not points:
        raise ValueError("no_points")
    label_to_i = {label: i for i, label in enumerate(LABELS)}
    X = np.array([[float(p["training_features"].get(col, 0.0)) for col in FEATURE_COLUMNS] for p in points], dtype=np.float32)
    y = np.array([label_to_i.get(p["labels"]["route_class"], 0) for p in points], dtype=np.int32)
    return X, y, LABELS


def train_tree_artifacts(points: list[dict[str, Any]], out_dir: Path | None = None) -> dict[str, Any]:
    out_dir = out_dir or OUT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(points) < 2:
        raise ValueError("at_least_two_points_required")
    X, y_raw, labels = matrix(points)
    unique = sorted(set(int(v) for v in y_raw.tolist()))
    if len(unique) < 2:
        y = np.array([(i % 2) for i in range(len(points))], dtype=np.int32)
        labels = ["ROUTE_A", "ROUTE_B"]
    else:
        remap = {old_label: new_label for new_label, old_label in enumerate(unique)}
        y = np.array([remap[int(v)] for v in y_raw.tolist()], dtype=np.int32)
        labels = [labels[i] for i in unique]
    import xgboost as xgb
    import treelite.frontend
    from treelite import gtil

    classes = max(2, len(set(int(v) for v in y.tolist())))
    objective = "binary:logistic" if classes == 2 else "multi:softprob"
    model = xgb.XGBClassifier(
        objective=objective,
        num_class=classes if classes > 2 else None,
        n_estimators=min(24, max(4, len(points))),
        max_depth=3,
        learning_rate=0.25,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        random_state=414,
        eval_metric="logloss" if classes == 2 else "mlogloss",
    )
    model.fit(X, y)
    xgb_path = out_dir / f"dev_journey_route_xgboost_{stamp()}.json"
    model.get_booster().save_model(xgb_path)
    tl_model = treelite.frontend.from_xgboost(model.get_booster())
    tl_path = out_dir / f"dev_journey_route_treelite_{stamp()}.tl"
    tl_model.serialize(tl_path)
    preds = gtil.predict(tl_model, X)
    return {
        "schema": "lucidota.dev_journey.tree_artifacts.v1",
        "training_performed": True,
        "row_count": len(points),
        "feature_columns": FEATURE_COLUMNS,
        "label_set": labels,
        "xgboost_model_path": str(xgb_path),
        "treelite_model_path": str(tl_path),
        "smoke_prediction_count": int(np.asarray(preds).shape[0]),
        "canonical_graph_writes_performed": False,
    }


def write_outputs(points: list[dict[str, Any]], *, train: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    base = stamp()
    jsonl = OUT / f"dev_journey_decision_points_{base}.jsonl"
    csv_path = OUT / f"dev_journey_decision_points_{base}.csv"
    with jsonl.open("w", encoding="utf-8") as fh:
        for p in points:
            fh.write(json.dumps(p, sort_keys=False, ensure_ascii=False) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["decision_id", "source_path", "ordinal", "route_class", *FEATURE_COLUMNS])
        writer.writeheader()
        for p in points:
            writer.writerow({"decision_id": p["decision_id"], "source_path": p["source_path"], "ordinal": p["ordinal"], "route_class": p["labels"]["route_class"], **p["training_features"]})
    tree = train_tree_artifacts(points) if train and len(points) >= 2 else {"training_performed": False, "reason": "train_not_requested_or_insufficient_rows"}
    by_label = Counter(p["labels"]["route_class"] for p in points)
    by_source = Counter(p["source_path"] for p in points)
    payload = {
        "schema": "lucidota.dev_journey.decision_points.receipt.v1",
        "generated_at": now(),
        "decision_points_path": rel(jsonl),
        "csv_path": rel(csv_path),
        "decision_point_count": len(points),
        "by_route_class": dict(sorted(by_label.items())),
        "top_sources": by_source.most_common(12),
        "tree_artifacts": {**tree, **{k: rel(v) for k, v in tree.items() if k.endswith("_path") and isinstance(v, str)}},
        "truth_status": "training_candidates_only",
        "canonical_graph_writes_performed": False,
    }
    receipt = OUT / f"dev_journey_decision_points_receipt_{base}.json"
    payload["receipt_path"] = rel(receipt)
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    OBS.parent.mkdir(parents=True, exist_ok=True)
    OBS.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    update_big_board(payload)
    print("RECEIPT_PATH=" + rel(receipt))
    print("DECISION_POINTS=" + str(len(points)))
    return payload


def update_big_board(payload: dict[str, Any]) -> None:
    try:
        board = json.loads(BIG_BOARD.read_text(encoding="utf-8")) if BIG_BOARD.exists() else {}
    except Exception:
        board = {}
    board.setdefault("observation_center", {})["dev_journey_decision_points"] = {
        "source_receipt": payload.get("receipt_path"),
        "decision_point_count": payload.get("decision_point_count"),
        "tree_training_performed": payload.get("tree_artifacts", {}).get("training_performed"),
        "canonical_graph_writes_performed": False,
    }
    board.setdefault("counters", {})["dev_journey_decision_points"] = payload.get("decision_point_count", 0)
    BIG_BOARD.parent.mkdir(parents=True, exist_ok=True)
    BIG_BOARD.write_text(json.dumps(board, indent=2, sort_keys=True, default=str), encoding="utf-8")


def default_sources() -> list[Path]:
    candidates = [
        ROOT / "GOALS" / "GOAL_LOG.md",
        ROOT / "GOALS" / "CURRENT_HANDOFF.md",
        ROOT / "00_PROJECT_BRAIN" / "STATUS_LEDGER.md",
        ROOT / "00_PROJECT_BRAIN" / "ACTIVE_INSTRUCTION_INDEX.md",
        ROOT / "scripts" / "SCRIPT_AUDIT_MANIFEST.jsonl",
        ROOT / "04_RUNTIME" / "observation_center" / "hunch_hypertimeline_latest.json",
    ]
    candidates.extend(sorted((ROOT / "05_OUTPUTS" / "project2501_board_moves").glob("project2501_board_move_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:60])
    candidates.extend(sorted((ROOT / "05_OUTPUTS" / "hunch_hypertimeline").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:20])
    candidates.extend(sorted((ROOT / "05_OUTPUTS" / "hunch_hypertimeline").glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:20])
    candidates.extend(sorted((ROOT / "05_OUTPUTS" / "krampuschewing").glob("krampuschewing_normalized_index_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:3])
    candidates.extend(sorted((ROOT / "05_OUTPUTS" / "krampuschewing").glob("krampuschewing_river_training_candidates_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:3])
    seen: set[Path] = set()
    out: list[Path] = []
    for p in candidates:
        if p.exists() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compile dev journey sticker decision points and optional XGBoost/Treelite artifacts.")
    p.add_argument("--source", action="append", default=[])
    p.add_argument("--max-points", type=int, default=500)
    p.add_argument("--train", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    sources = [Path(s) for s in args.source] if args.source else default_sources()
    points = compile_decision_points(sources, max_points=args.max_points)
    payload = write_outputs(points, train=bool(args.train))
    if args.json:
        print(json.dumps(payload, sort_keys=True, default=str))
    print("DEV_JOURNEY_DECISION_POINTS=PASS" if points else "DEV_JOURNEY_DECISION_POINTS=EMPTY")
    return 0 if points else 2


if __name__ == "__main__":
    raise SystemExit(main())
