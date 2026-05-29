#!/usr/bin/env python3
"""Deterministic fastlane/slowlane metadata gate with cache-to-slowlane flushes.

PocketFlow lesson: tiny packet -> decision -> post hook.  Syd lesson: extract and
validate facts before any model.  llm-router lesson: route metadata, not vibes.
This script performs no model calls, network calls, or canonical graph writes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from spine_common import now, receipt, rel, sha256_json  # noqa: E402

SCHEMA = "lucidota.fast_slow_lane_gate.v1"
LANE_FAST = "FASTLANE"
LANE_SLOW = "SLOWLANE"
DEFAULT_CACHE_DIR = ROOT / "09_STORAGE" / "lane_cache"
DEFAULT_RECEIPT_ROOT = ROOT / "05_OUTPUTS" / "fast_slow_lane"

FLOW_EDGES = {
    "inbound": ("USER_CLI_INPUT", "PROCESSING_CORE"),
    "core": ("PROCESSING_CORE", "LANE_WORKER"),
    "outbound": ("PROCESSING_CORE", "OUTBOUND"),
    "feedback": ("OUTBOUND", "FAST_SLOW_CACHE"),
}

SLOW_WORDS = {
    "analysis", "analyze", "research", "audit", "refactor", "integrate", "integration",
    "deep", "prove", "verify", "investigate", "summarize", "ontology", "bytewax",
    "mamba", "deepseek", "model", "llm", "benchmark", "compile", "pytest",
    "cargo", "failure", "failed", "blocker", "graph", "workflow", "slowlanes",
}
FAST_WORDS = {
    "status", "help", "probe", "route", "metadata", "cache", "receipt", "cli", "json",
    "ping", "health", "index", "manifest", "fastlane", "check", "list", "next",
}
IMPORTANT_WORDS = {
    "proof", "receipt", "law", "metadata", "gate", "cache", "flush", "pocketflow",
    "syd", "ncnn", "fastlane", "slowlane", "operator", "blocker", "queue",
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def clean_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())[:80].strip("._-")
    return key or "default"


def read_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"metadata JSON invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("metadata JSON must decode to an object")
    return data


def load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        p = Path(args.text_file)
        if not p.is_absolute():
            p = ROOT / p
        return p.read_text(encoding="utf-8", errors="replace")
    return args.text or ""


def token_hits(text: str, words: set[str]) -> list[str]:
    low = text.lower()
    hits = [w for w in sorted(words) if re.search(rf"\b{re.escape(w)}\b", low)]
    return hits


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def route_packet(packet: dict[str, Any], *, slow_char_threshold: int) -> dict[str, Any]:
    text = str(packet.get("text") or "")
    metadata = packet.get("metadata") or {}
    slow_hits = token_hits(text, SLOW_WORDS)
    fast_hits = token_hits(text, FAST_WORDS)
    reasons: list[str] = []
    if len(text) > slow_char_threshold:
        reasons.append(f"text_chars>{slow_char_threshold}")
    if slow_hits:
        reasons.append("slow_keywords:" + ",".join(slow_hits[:8]))
    for key in ("requires_model", "deep_analysis", "external_research", "graph_write", "human_review"):
        if truthy(metadata.get(key)):
            reasons.append(f"metadata:{key}=true")
    hint = str(packet.get("lane_hint") or "auto").strip().upper()
    if hint in {"SLOW", "SLOWLANE"}:
        reasons.append("lane_hint=slowlane")
    lane = LANE_SLOW if reasons else LANE_FAST
    if hint in {"FAST", "FASTLANE"} and lane == LANE_FAST:
        reasons.append("lane_hint=fastlane")
    if lane == LANE_FAST and fast_hits:
        reasons.append("fast_keywords:" + ",".join(fast_hits[:8]))
    if not reasons:
        reasons.append("default_fast_metadata_gate")
    return {
        "lane": lane,
        "route_reason": reasons,
        "slow_keyword_hits": slow_hits,
        "fast_keyword_hits": fast_hits,
        "deterministic": True,
        "model_calls_performed": False,
    }


def importance_score(text: str, metadata: dict[str, Any], explicit: float | None) -> float:
    if explicit is not None:
        return clamp(explicit)
    hits = token_hits(text, IMPORTANT_WORDS)
    score = 0.18 + min(0.45, len(hits) * 0.06)
    for key in ("operator_critical", "proof_required", "queue_candidate", "cache_worthy"):
        if truthy(metadata.get(key)):
            score += 0.12
    return round(clamp(score), 3)


def build_flow(decision: dict[str, Any], stage: str, direction: str) -> list[dict[str, Any]]:
    stages = list(FLOW_EDGES) if stage == "all" else [stage]
    out = []
    for name in stages:
        src, dst = FLOW_EDGES[name]
        edges = []
        if direction in {"forward", "allways"}:
            edges.append({"direction": "forward", "from": src, "to": dst})
        if direction in {"return", "allways"}:
            edges.append({"direction": "return", "from": dst, "to": src})
        out.append({"stage": name, "lane": decision["lane"], "edges": edges, "reason": decision["route_reason"]})
    return out


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def cache_paths(cache_dir: Path, cache_key: str) -> dict[str, Path]:
    root = cache_dir / clean_key(cache_key)
    return {
        "root": root,
        "fast_active": root / "fastlane_active.jsonl",
        "slow_inbox": root / "slowlane_inbox.jsonl",
        "flush_dir": root / "slowlane_flush",
    }


def cache_bit(args: argparse.Namespace, packet: dict[str, Any], decision: dict[str, Any], flow: list[dict[str, Any]], importance: float) -> dict[str, Any]:
    paths = cache_paths(Path(args.cache_dir), args.cache_key)
    text = str(packet.get("text") or "")
    bit = {
        "schema": "lucidota.fast_slow_lane.bit.v1",
        "cached_at": now(),
        "packet_id": packet["packet_id"],
        "lane": decision["lane"],
        "importance": importance,
        "stage_count": len(flow),
        "source": packet["source"],
        "target": packet["target"],
        "text_sha256": packet["text_sha256"],
        "text_chars": len(text),
        "text_preview": text[:240],
        "metadata": packet["metadata"],
        "route_reason": decision["route_reason"],
        "flow": flow,
    }
    if args.store_text:
        bit["text"] = text
    if decision["lane"] == LANE_FAST:
        append_jsonl(paths["fast_active"], bit)
        destination = paths["fast_active"]
    else:
        append_jsonl(paths["slow_inbox"], bit)
        destination = paths["slow_inbox"]
    return {"cached": True, "cache_path": rel(destination), "bit": bit}


def flush_fastlane(cache_dir: Path, cache_key: str, *, force: bool, flush_count: int, flush_importance: float) -> dict[str, Any]:
    paths = cache_paths(cache_dir, cache_key)
    rows = jsonl_rows(paths["fast_active"])
    total = round(sum(float(r.get("importance") or 0.0) for r in rows), 6)
    reasons = []
    if force:
        reasons.append("force")
    if rows and len(rows) >= flush_count:
        reasons.append(f"count>={flush_count}")
    if rows and total >= flush_importance:
        reasons.append(f"importance>={flush_importance}")
    if not rows or not reasons:
        return {
            "flushed": False,
            "reason": "empty" if not rows else "threshold_not_met",
            "fastlane_count": len(rows),
            "importance_total": total,
            "fastlane_active": rel(paths["fast_active"]),
        }
    stamp = now().replace(":", "").replace("-", "").replace(".", "")
    paths["flush_dir"].mkdir(parents=True, exist_ok=True)
    rotated = paths["flush_dir"] / f"fastlane_flushed_{stamp}.jsonl"
    bundle_path = paths["flush_dir"] / f"slowlane_bundle_{stamp}.json"
    rotated.write_text(paths["fast_active"].read_text(encoding="utf-8"), encoding="utf-8")
    paths["fast_active"].write_text("", encoding="utf-8")
    bundle = {
        "schema": "lucidota.fastlane_to_slowlane.bundle.v1",
        "generated_at": now(),
        "cache_key": clean_key(cache_key),
        "flush_reason": reasons,
        "bit_count": len(rows),
        "importance_total": total,
        "target_lane": LANE_SLOW,
        "target_queue": "SLOWLANE_ANALYSIS_QUEUE",
        "source_cache": rel(paths["fast_active"]),
        "rotated_cache": rel(rotated),
        "packet_ids": [r.get("packet_id") for r in rows],
        "bits": rows,
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "flushed": True,
        "reason": reasons,
        "fastlane_count": len(rows),
        "importance_total": total,
        "bundle_path": rel(bundle_path),
        "rotated_cache": rel(rotated),
        "fastlane_active": rel(paths["fast_active"]),
    }


def status_payload(cache_dir: Path, cache_key: str) -> dict[str, Any]:
    paths = cache_paths(cache_dir, cache_key)
    fast = jsonl_rows(paths["fast_active"])
    slow = jsonl_rows(paths["slow_inbox"])
    bundles = sorted(paths["flush_dir"].glob("slowlane_bundle_*.json")) if paths["flush_dir"].exists() else []
    return {
        "schema": "lucidota.fast_slow_lane.status.v1",
        "generated_at": now(),
        "cache_key": clean_key(cache_key),
        "fastlane_active": rel(paths["fast_active"]),
        "fastlane_count": len(fast),
        "fastlane_importance_total": round(sum(float(r.get("importance") or 0.0) for r in fast), 6),
        "slowlane_inbox": rel(paths["slow_inbox"]),
        "slowlane_inbox_count": len(slow),
        "slowlane_bundle_count": len(bundles),
        "latest_slowlane_bundle": rel(bundles[-1]) if bundles else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route CLI/user packets through deterministic fastlane/slowlane metadata gates.")
    sub = parser.add_subparsers(dest="command")
    route = sub.add_parser("route", help="Route and cache one packet.")
    source = route.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    route.add_argument("--metadata-json", type=read_json_object, default={})
    route.add_argument("--source", default="USER_CLI_INPUT")
    route.add_argument("--target", default="PROCESSING_CORE")
    route.add_argument("--stage", choices=["all", *FLOW_EDGES.keys()], default="all")
    route.add_argument("--direction", choices=["forward", "return", "allways"], default="allways")
    route.add_argument("--lane-hint", choices=["auto", "fastlane", "slowlane"], default="auto")
    route.add_argument("--importance", type=float)
    route.add_argument("--slow-char-threshold", type=int, default=1800)
    route.add_argument("--cache-key", default="operator_cli")
    route.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    route.add_argument("--flush-count", type=int, default=5)
    route.add_argument("--flush-importance", type=float, default=1.0)
    route.add_argument("--flush", action="store_true")
    route.add_argument("--store-text", action="store_true")
    route.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    route.add_argument("--no-receipt", action="store_true")
    route.add_argument("--json", action="store_true")

    flush = sub.add_parser("flush", help="Force or threshold-check fastlane cache flush.")
    flush.add_argument("--cache-key", default="operator_cli")
    flush.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    flush.add_argument("--flush-count", type=int, default=5)
    flush.add_argument("--flush-importance", type=float, default=1.0)
    flush.add_argument("--force", action="store_true")
    flush.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    flush.add_argument("--no-receipt", action="store_true")
    flush.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Show fastlane/slowlane cache state.")
    status.add_argument("--cache-key", default="operator_cli")
    status.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    status.add_argument("--json", action="store_true")
    return parser


def route_main(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    text = load_text(args)
    metadata = dict(args.metadata_json or {})
    packet = {
        "source": args.source,
        "target": args.target,
        "text": text,
        "text_sha256": sha256_json({"text": text}),
        "metadata": metadata,
        "lane_hint": args.lane_hint,
    }
    packet["packet_id"] = sha256_json(packet)
    decision = route_packet(packet, slow_char_threshold=args.slow_char_threshold)
    importance = importance_score(text, metadata, args.importance)
    flow = build_flow(decision, args.stage, args.direction)
    cache = cache_bit(args, packet, decision, flow, importance)
    flush = flush_fastlane(Path(args.cache_dir), args.cache_key, force=args.flush, flush_count=args.flush_count, flush_importance=args.flush_importance)
    payload = {
        "schema": SCHEMA,
        "generated_at": now(),
        "command": "route",
        "packet_id": packet["packet_id"],
        "base_lane": decision["lane"],
        "importance": importance,
        "decision": decision,
        "flow": flow,
        "cache": {k: v for k, v in cache.items() if k != "bit"},
        "flush": flush,
        "status": status_payload(Path(args.cache_dir), args.cache_key),
        "model_calls_performed": False,
        "network_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS",
    }
    if not args.no_receipt:
        receipt("fast_slow_lane_gate", payload, root=Path(args.receipt_root))
    return 0, payload


def flush_main(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    flush = flush_fastlane(Path(args.cache_dir), args.cache_key, force=args.force, flush_count=args.flush_count, flush_importance=args.flush_importance)
    payload = {
        "schema": SCHEMA,
        "generated_at": now(),
        "command": "flush",
        "flush": flush,
        "status": status_payload(Path(args.cache_dir), args.cache_key),
        "model_calls_performed": False,
        "network_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "verdict": "PASS",
    }
    if not args.no_receipt:
        receipt("fast_slow_lane_gate", payload, root=Path(args.receipt_root))
    return 0, payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "route":
        code, payload = route_main(args)
    elif args.command == "flush":
        code, payload = flush_main(args)
    else:
        payload = status_payload(Path(args.cache_dir), args.cache_key)
        code = 0
    if getattr(args, "json", False):
        print(json.dumps(payload, sort_keys=True))
    if args.command == "status":
        print(f"FASTLANE_COUNT={payload['fastlane_count']}")
        print(f"SLOWLANE_INBOX_COUNT={payload['slowlane_inbox_count']}")
    else:
        print("FAST_SLOW_LANE_GATE=" + payload["verdict"])
        if payload.get("base_lane"):
            print("BASE_LANE=" + payload["base_lane"])
        if payload.get("flush", {}).get("flushed"):
            print("SLOWLANE_BUNDLE=" + str(payload["flush"].get("bundle_path")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
