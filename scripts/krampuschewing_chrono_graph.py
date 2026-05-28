#!/usr/bin/env python3
"""Build KRAMPUSCHEWING chronology graph packets from an index JSONL."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_CHRONO = ROOT / "05_OUTPUTS" / "krampuschewing" / "chrono"
OUT_GRAPH = ROOT / "05_OUTPUTS" / "krampuschewing" / "graph"
TRUTH = "imported_artifact_graph_not_accepted_truth"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def stable(prefix: str, *parts: Any) -> str:
    basis = "\0".join(str(p) for p in parts)
    return f"{prefix}_" + hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()[:32]


def load_rows(index: Path) -> Iterable[dict[str, Any]]:
    with index.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def load_large_validation(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_archive_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    if path is None:
        candidates = sorted((ROOT / "05_OUTPUTS/krampuschewing/archive_unpack").glob("archive_unpack_manifest_*.jsonl"))
        path = candidates[-1] if candidates else None
    if path is None or not path.exists():
        return maps
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = str(row.get("extracted_path") or "")
            if ep:
                maps[ep] = row
                # also allow lookup by path under unpack root
                marker = "09_STORAGE/krampuschewing_unpacked/"
                if marker in ep:
                    maps[ep.split(marker, 1)[1]] = row
    return maps


def row_path(row: dict[str, Any]) -> str:
    return str(row.get("repo_relative_path") or row.get("relative_path") or row.get("path") or "")


def lane(row: dict[str, Any]) -> str:
    return str(row.get("normalized_lane_guess") or row.get("lane_guess") or "UNKNOWN")


def case_guess(row: dict[str, Any]) -> str | None:
    return row.get("normalized_case_guess") or row.get("case_guess")


def dev_project(row: dict[str, Any]) -> str | None:
    return row.get("normalized_dev_project_guess") or row.get("dev_project_guess")


def node_kind_for(row: dict[str, Any], source_label: str, archive_info: dict[str, Any] | None) -> str:
    if archive_info or source_label == "KRAMPUSCHEWING_UNPACKED":
        return "EXTRACTED_FILE"
    ln = lane(row)
    if row.get("active_runtime_db_risk"):
        return "ACTIVE_RUNTIME_DB_RISK"
    return {
        "CASE_WORK": "EVIDENCE",
        "DEV_WORK": "FILE_ARTIFACT",
        "PROMPT_NOTE": "PROMPT",
        "RECEIPT": "RECEIPT",
        "SOURCE_CODE": "SOURCE_CODE",
        "RUNTIME_LOG": "RUNTIME_LOG",
        "MODEL_ARTIFACT": "MODEL_ARTIFACT",
        "GRAPH_STAGING": "PROOF_CANDIDATE",
        "PROOF_CANDIDATE": "PROOF_CANDIDATE",
        "SAVED_FILE": "FILE_ARTIFACT",
    }.get(ln, "FILE_ARTIFACT")


def confidence(row: dict[str, Any]) -> str:
    if row.get("sha256"):
        return "high"
    if row.get("sha256_status") in {"computed", "skipped_large_file", "skipped_large_member"}:
        return "medium"
    return "low"


def event_time(row: dict[str, Any], archive_info: dict[str, Any] | None = None) -> tuple[str | None, str]:
    t = row.get("modified_time_utc") or (archive_info or {}).get("member_mtime") or (archive_info or {}).get("time_guess")
    if t:
        return str(t), "mtime_or_member_mtime"
    return None, "unknown"


def node(node_id: str, kind: str, label: str, source_path: str | None, source_sha256: str | None, event_time_utc: str | None, time_source: str, lane_value: str, confidence_value: str, provenance: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"schema": "lucidota.krampuschewing.graph_node.v1", "node_id": node_id, "node_kind": kind, "label": label[:300], "source_path": source_path, "source_sha256": source_sha256, "event_time_utc": event_time_utc, "time_source": time_source, "lane": lane_value, "confidence": confidence_value, "provenance": provenance, "krampus_run_id": run_id, "canonical_truth_status": TRUTH}


def edge(edge_id: str, kind: str, source_node_id: str, target_node_id: str, source_path: str | None, source_sha256: str | None, event_time_utc: str | None, time_source: str, lane_value: str, confidence_value: str, provenance: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {"schema": "lucidota.krampuschewing.graph_edge.v1", "edge_id": edge_id, "edge_kind": kind, "source_node_id": source_node_id, "target_node_id": target_node_id, "source_path": source_path, "source_sha256": source_sha256, "event_time_utc": event_time_utc, "time_source": time_source, "lane": lane_value, "confidence": confidence_value, "provenance": provenance, "krampus_run_id": run_id, "canonical_truth_status": TRUTH}


def add_node(nodes: dict[str, dict[str, Any]], n: dict[str, Any]) -> None:
    nodes.setdefault(n["node_id"], n)


def build(index: Path, large_validation: Path | None, source_label: str, run_id: str, archive_manifest: Path | None = None) -> dict[str, Any]:
    OUT_CHRONO.mkdir(parents=True, exist_ok=True)
    OUT_GRAPH.mkdir(parents=True, exist_ok=True)
    manifest_map = load_archive_manifest(archive_manifest)
    large_data = load_large_validation(large_validation)
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    rows_seen = 0
    unknown_time = 0

    for row in load_rows(index):
        rows_seen += 1
        rp = row_path(row)
        archive_info = manifest_map.get(rp) or manifest_map.get(str(row.get("relative_path") or ""))
        t, t_source = event_time(row, archive_info)
        if not t:
            unknown_time += 1
        ln = lane(row)
        conf = confidence(row)
        sha = row.get("sha256") or (archive_info or {}).get("sha256")
        kind = node_kind_for(row, source_label, archive_info)
        file_node_id = stable("node", source_label, "file", rp, sha or row.get("sha256_status"))
        add_node(nodes, node(file_node_id, kind, Path(rp).name or rp, rp, sha, t, t_source, ln, conf, {"source_label": source_label, "index": rel(index), "archive_manifest_row": archive_info}, run_id))
        events.append({"schema": "lucidota.krampuschewing.chrono_event.v1", "event_id": stable("event", source_label, rp, t or "unknown"), "node_id": file_node_id, "source_path": rp, "source_sha256": sha, "event_time_utc": t, "time_source": t_source, "lane": ln, "kind_guess": row.get("kind_guess"), "confidence": conf, "provenance": {"index": rel(index), "archive_manifest_row": archive_info}, "krampus_run_id": run_id, "canonical_truth_status": TRUTH})
        if t:
            date = t[:10]
            hour = t[:13] if len(t) >= 13 else t
            date_id = stable("node", "date", date)
            time_id = stable("node", "time", hour)
            add_node(nodes, node(date_id, "DATE_BUCKET", date, None, None, t, "derived_from_event_time", "TIME", "high", {"source_label": source_label}, run_id))
            add_node(nodes, node(time_id, "TIME_BUCKET", hour, None, None, t, "derived_from_event_time", "TIME", "high", {"source_label": source_label}, run_id))
            edges[stable("edge", file_node_id, "IN_DATE_BUCKET", date_id)] = edge(stable("edge", file_node_id, "IN_DATE_BUCKET", date_id), "IN_DATE_BUCKET", file_node_id, date_id, rp, sha, t, t_source, ln, conf, {"date": date}, run_id)
            edges[stable("edge", file_node_id, "IN_TIME_BUCKET", time_id)] = edge(stable("edge", file_node_id, "IN_TIME_BUCKET", time_id), "IN_TIME_BUCKET", file_node_id, time_id, rp, sha, t, t_source, ln, conf, {"time": hour}, run_id)
        cg = case_guess(row)
        if cg:
            cid = stable("node", "case", cg)
            add_node(nodes, node(cid, "CASE", cg, None, None, t, "derived_from_path_or_index", "CASE_WORK", "medium", {"source_label": source_label}, run_id))
            edges[stable("edge", file_node_id, "BELONGS_TO_CASE", cid)] = edge(stable("edge", file_node_id, "BELONGS_TO_CASE", cid), "BELONGS_TO_CASE", file_node_id, cid, rp, sha, t, t_source, ln, conf, {}, run_id)
        dp = dev_project(row)
        if dp:
            did = stable("node", "dev", dp)
            add_node(nodes, node(did, "DEV_PROJECT", dp, None, None, t, "derived_from_path_or_index", "DEV_WORK", "medium", {"source_label": source_label}, run_id))
            edges[stable("edge", file_node_id, "BELONGS_TO_DEV_PROJECT", did)] = edge(stable("edge", file_node_id, "BELONGS_TO_DEV_PROJECT", did), "BELONGS_TO_DEV_PROJECT", file_node_id, did, rp, sha, t, t_source, ln, conf, {}, run_id)
        if row.get("duplicate_of"):
            dup_id = stable("node", source_label, "file", row.get("duplicate_of"), sha or row.get("sha256_status"))
            edges[stable("edge", file_node_id, "DUPLICATE_OF", dup_id)] = edge(stable("edge", file_node_id, "DUPLICATE_OF", dup_id), "DUPLICATE_OF", file_node_id, dup_id, rp, sha, t, t_source, ln, "medium", {"duplicate_of": row.get("duplicate_of")}, run_id)
        if row.get("contains_graph_terms"):
            graph_id = stable("node", "policy", "graph_mentions")
            add_node(nodes, node(graph_id, "POLICY_EXCLUSION", "Graph term mention / staging only", None, None, t, "derived", "GRAPH_STAGING", "low", {"source_label": source_label}, run_id))
            edges[stable("edge", file_node_id, "MENTIONS_GRAPH", graph_id)] = edge(stable("edge", file_node_id, "MENTIONS_GRAPH", graph_id), "MENTIONS_GRAPH", file_node_id, graph_id, rp, sha, t, t_source, ln, "low", {}, run_id)
        if archive_info:
            archive_path = archive_info.get("archive_path")
            archive_id = stable("node", "archive", archive_path or Path(str(row.get("relative_path") or "")).parts[0])
            add_node(nodes, node(archive_id, "FILE_ARTIFACT", f"Archive: {archive_info.get('archive_path') or Path(rp).parts[0]}", archive_path, None, archive_info.get("archive_mtime"), "archive_mtime", "SAVED_FILE", "medium", {"source_label": source_label, "archive_unpack_manifest": True}, run_id))
            edges[stable("edge", archive_id, "CONTAINS", file_node_id)] = edge(stable("edge", archive_id, "CONTAINS", file_node_id), "CONTAINS", archive_id, file_node_id, rp, sha, t, t_source, ln, conf, {"archive_path": archive_path, "extraction_receipt_ref": "archive_unpack_manifest"}, run_id)
            edges[stable("edge", file_node_id, "DERIVED_FROM", archive_id)] = edge(stable("edge", file_node_id, "DERIVED_FROM", archive_id), "DERIVED_FROM", file_node_id, archive_id, rp, sha, t, t_source, ln, conf, {"archive_path": archive_path}, run_id)
        if ln in {"CASE_WORK", "PROOF_CANDIDATE"} or row.get("contains_claim_evidence_terms"):
            proof_id = stable("node", "proof", "candidate")
            add_node(nodes, node(proof_id, "PROOF_CANDIDATE", "Proof/custody candidate lane", None, None, t, "derived", "PROOF_CANDIDATE", "low", {"source_label": source_label}, run_id))
            edges[stable("edge", file_node_id, "POTENTIAL_EVIDENCE_FOR", proof_id)] = edge(stable("edge", file_node_id, "POTENTIAL_EVIDENCE_FOR", proof_id), "POTENTIAL_EVIDENCE_FOR", file_node_id, proof_id, rp, sha, t, t_source, ln, "low", {}, run_id)
        if ln in {"DEV_WORK", "SOURCE_CODE", "CASE_WORK", "PROMPT_NOTE"}:
            train_id = stable("node", "river", "training_source_candidate")
            add_node(nodes, node(train_id, "POLICY_EXCLUSION", "River training source candidate only", None, None, t, "derived", "MODEL_ARTIFACT", "low", {"river_training_performed": False}, run_id))
            edges[stable("edge", file_node_id, "TRAINING_SOURCE_FOR", train_id)] = edge(stable("edge", file_node_id, "TRAINING_SOURCE_FOR", train_id), "TRAINING_SOURCE_FOR", file_node_id, train_id, rp, sha, t, t_source, ln, "low", {"river_training_performed": False}, run_id)

    # Active DB policy-only handling from validation receipt.
    active_db_risks = 0
    for rec in large_data.get("records", []):
        if rec.get("large_class") == "ACTIVE_RUNTIME_DB_RISK":
            active_db_risks += 1
            rp = rec.get("relative_path")
            t = rec.get("modified_time_utc")
            db_id = stable("node", "active_db_risk", rp)
            policy_id = stable("node", "policy", "do_not_touch_active_runtime_db", rp)
            add_node(nodes, node(db_id, "ACTIVE_RUNTIME_DB_RISK", f"DO NOT TOUCH active runtime DB risk: {rp}", rp, None, t, "validation_receipt_mtime", "MODEL_ARTIFACT", "high", {"large_file_validation": rel(large_validation) if large_validation else None, "recommended_next_action": rec.get("recommended_next_action")}, run_id))
            add_node(nodes, node(policy_id, "POLICY_EXCLUSION", "Policy exclusion: active runtime DB; no content/hash/River", None, None, t, "policy", "MODEL_ARTIFACT", "high", {"large_file_validation": rel(large_validation) if large_validation else None}, run_id))
            for ek in ["EXCLUDED_FROM_NORMAL_PROCESSING", "EXCLUDED_FROM_RIVER", "DO_NOT_TOUCH"]:
                edges[stable("edge", db_id, ek, policy_id)] = edge(stable("edge", db_id, ek, policy_id), ek, db_id, policy_id, rp, None, t, "validation_receipt_mtime", "MODEL_ARTIFACT", "high", {"policy_only_graph_handling": True}, run_id)

    sorted_events = sorted(events, key=lambda e: (e.get("event_time_utc") or "9999", e.get("source_path") or ""))
    # Add chronology adjacency for dated events only; cap is all events but deterministic.
    prev = None
    for ev in sorted_events:
        if not ev.get("event_time_utc"):
            continue
        if prev is not None:
            e_after = edge(stable("edge", prev["node_id"], "BEFORE", ev["node_id"]), "BEFORE", prev["node_id"], ev["node_id"], ev.get("source_path"), ev.get("source_sha256"), ev.get("event_time_utc"), ev.get("time_source"), ev.get("lane"), "low", {"chrono_adjacency": True}, run_id)
            edges[e_after["edge_id"]] = e_after
        prev = ev

    ts = stamp()
    events_path = OUT_CHRONO / f"krampuschewing_chrono_events_{ts}.jsonl"
    events_summary = OUT_CHRONO / f"krampuschewing_chrono_summary_{ts}.json"
    nodes_path = OUT_GRAPH / f"krampuschewing_graph_nodes_{ts}.jsonl"
    nodes_summary = OUT_GRAPH / f"krampuschewing_graph_nodes_summary_{ts}.json"
    edges_path = OUT_GRAPH / f"krampuschewing_graph_edges_{ts}.jsonl"
    edges_summary = OUT_GRAPH / f"krampuschewing_graph_edges_summary_{ts}.json"
    packet_path = OUT_GRAPH / f"krampuschewing_graph_packet_{ts}.json"
    OUT_CHRONO.mkdir(parents=True, exist_ok=True)
    OUT_GRAPH.mkdir(parents=True, exist_ok=True)
    with events_path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, sort_keys=False, ensure_ascii=False) + "\n")
    with nodes_path.open("w", encoding="utf-8") as fh:
        for n in nodes.values():
            fh.write(json.dumps(n, sort_keys=False, ensure_ascii=False) + "\n")
    with edges_path.open("w", encoding="utf-8") as fh:
        for e in edges.values():
            fh.write(json.dumps(e, sort_keys=False, ensure_ascii=False) + "\n")
    event_times = [e["event_time_utc"] for e in events if e.get("event_time_utc")]
    chrono_summary = {"schema": "lucidota.krampuschewing.chrono_summary.v1", "generated_at_utc": now(), "source_index": rel(index), "source_label": source_label, "krampus_run_id": run_id, "events": len(events), "earliest_time": min(event_times) if event_times else None, "latest_time": max(event_times) if event_times else None, "unknown_time_events": unknown_time, "output": rel(events_path), "canonical_truth_status": TRUTH}
    node_counts = Counter(n["node_kind"] for n in nodes.values())
    edge_counts = Counter(e["edge_kind"] for e in edges.values())
    nodes_summary.write_text(json.dumps({"schema": "lucidota.krampuschewing.graph_nodes_summary.v1", "generated_at_utc": now(), "nodes": len(nodes), "by_node_kind": dict(sorted(node_counts.items())), "output": rel(nodes_path), "active_runtime_db_risk_count": active_db_risks, "canonical_truth_status": TRUTH}, indent=2, sort_keys=False), encoding="utf-8")
    edges_summary.write_text(json.dumps({"schema": "lucidota.krampuschewing.graph_edges_summary.v1", "generated_at_utc": now(), "edges": len(edges), "by_edge_kind": dict(sorted(edge_counts.items())), "output": rel(edges_path), "canonical_truth_status": TRUTH}, indent=2, sort_keys=False), encoding="utf-8")
    events_summary.write_text(json.dumps(chrono_summary, indent=2, sort_keys=False), encoding="utf-8")
    packet = {"schema": "lucidota.krampuschewing.graph_packet.v1", "generated_at_utc": now(), "krampus_run_id": run_id, "source_index": rel(index), "source_label": source_label, "large_file_validation": rel(large_validation) if large_validation else None, "archive_manifest": rel(archive_manifest) if archive_manifest else None, "chrono_events": rel(events_path), "chrono_summary": rel(events_summary), "graph_nodes": rel(nodes_path), "graph_nodes_summary": rel(nodes_summary), "graph_edges": rel(edges_path), "graph_edges_summary": rel(edges_summary), "graph_packet": rel(packet_path), "events_count": len(events), "nodes_count": len(nodes), "edges_count": len(edges), "active_runtime_db_risk_count": active_db_risks, "canonical_truth_status": TRUTH, "canonical_graph_materialization": False, "canonical_graph_writes": False}
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=False), encoding="utf-8")
    return packet


def main() -> int:
    ap = argparse.ArgumentParser(description="Build KRAMPUSCHEWING chrono graph packet")
    ap.add_argument("--index", required=True)
    ap.add_argument("--large-file-validation")
    ap.add_argument("--source-label", default="KRAMPUSCHEWING")
    ap.add_argument("--run-id", default="krampuschewing_graph_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    ap.add_argument("--archive-manifest")
    args = ap.parse_args()
    index = Path(args.index)
    if not index.is_absolute():
        index = ROOT / index
    large = Path(args.large_file_validation) if args.large_file_validation else None
    if large is not None and not large.is_absolute():
        large = ROOT / large
    manifest = Path(args.archive_manifest) if args.archive_manifest else None
    if manifest is not None and not manifest.is_absolute():
        manifest = ROOT / manifest
    packet = build(index, large, args.source_label, args.run_id, manifest)
    print("CHRONO_EVENTS=" + packet["chrono_events"])
    print("GRAPH_NODES=" + packet["graph_nodes"])
    print("GRAPH_EDGES=" + packet["graph_edges"])
    print("GRAPH_PACKET=" + packet["graph_packet"])
    print("EVENTS_COUNT=" + str(packet["events_count"]))
    print("NODES_COUNT=" + str(packet["nodes_count"]))
    print("EDGES_COUNT=" + str(packet["edges_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
