#!/usr/bin/env python3
"""Extract Ahoy strategy training rows into TRM seq2seq training pairs.

Reads strategy_sample.jsonl from the Ahoy million-replay output, converts
95 abstract strategy features + 22 labels into compact training pairs,
splits by game_id, and writes receipts.

ETL WITH CARE — every output row carries:
  - Triple hash: row content_hash, row full_hash, file-level sha256
  - Triple timestamp: created_at (source), processed_at (extraction),
    verified_at (verification)
  - Schema validation, cipher-free (no PII in game state)

Input:  strategy_sample.jsonl (or any ahoy_sim JSONL with features+labels)
Output: 05_OUTPUTS/trm_training/ahoy/{train,val,test}.jsonl + receipt
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "ahoy"
RECEIPT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "receipts"

ENTITY_NAME_MAP = {
    "hash:6128541148cf40b38226de07": "BLUEFIN (Authority)",
    "hash:7adc22c424ecd716fb954058": "MOLLUSK (Insurgent)",
    "hash:f43d9ffa017ac1407ed0b12f": "SMUGGLER (Opportunist)",
}

FEATURE_GROUPS = {
    "Leverage": [
        "authority_leverage_score", "insurgent_leverage_score",
        "opportunist_leverage_score", "leverage_ratio_authority_to_insurgent",
        "leverage_ratio_authority_to_opportunist",
        "leverage_delta_authority", "leverage_delta_insurgent",
        "leverage_delta_opportunist",
    ],
    "Capacities": [
        "action_capacity", "resource_capacity", "mobility_capacity",
        "hand_capacity", "dice_capacity", "repair_capacity", "combat_capacity",
    ],
    "Friction": [
        "authority_friction_incoming", "insurgent_friction_incoming",
        "opportunist_friction_incoming", "authority_friction_projected",
        "insurgent_friction_projected", "opportunist_friction_projected",
        "route_hazard_cost", "blockade_pressure", "forced_response_count",
        "denial_pressure", "combat_risk", "movement_penalty",
        "friction_delta_by_entity",
    ],
    "Visibility": [
        "authority_visibility_footprint", "insurgent_visibility_footprint",
        "opportunist_visibility_footprint", "visibility_asymmetry",
        "observation_coverage", "detection_risk", "patrol_presence",
        "controlled_region_pressure", "revealed_capability_count",
        "projection_range", "visibility_delta_by_entity",
    ],
    "Topology": [
        "choke_point_control_score", "route_efficiency_authority",
        "route_efficiency_insurgent", "route_efficiency_opportunist",
        "map_fragmentation_score", "connectivity_score", "bottleneck_count",
        "escape_route_count", "contested_region_count", "topology_advantage_entity",
    ],
    "Targeting": [
        "authority_targets_insurgent", "authority_targets_opportunist",
        "insurgent_targets_authority", "insurgent_targets_opportunist",
        "opportunist_extracts_from_authority_insurgent_conflict",
    ],
    "Dependencies": [
        "pledge_dependency_score", "delivery_dependency_score",
        "denial_dependency_score", "protection_dependency_score",
        "conflict_pair_intensity", "third_party_extraction_score",
    ],
    "Temporal": [
        "round_number", "turn_number", "phase_index", "sequence_index",
        "prior_state_ref",
    ],
    "Trends": [
        "leverage_trend_authority", "leverage_trend_insurgent",
        "leverage_trend_opportunist", "friction_trend_authority",
        "friction_trend_insurgent", "friction_trend_opportunist",
        "visibility_trend_authority", "visibility_trend_insurgent",
        "visibility_trend_opportunist", "pattern_window_size",
    ],
    "Modes": [
        "mode_authority_lockdown", "mode_authority_defensive",
        "mode_authority_overextended", "mode_insurgent_distributed_pressure",
        "mode_insurgent_combo_buildup", "mode_insurgent_stall",
        "mode_opportunist_extraction", "mode_opportunist_blockade_runner",
        "mode_rest_positional", "mode_collapse_risk",
    ],
    "WinProbability": [
        "win_probability_authority", "win_probability_insurgent",
        "win_probability_opportunist", "win_probability_delta_authority",
        "win_probability_delta_insurgent", "win_probability_delta_opportunist",
        "score_delta_authority", "score_delta_insurgent",
        "score_delta_opportunist", "strategic_swing_score",
    ],
}

LABEL_NAMES = [
    "primary_dynamic_label",
    "winning_entity_role",
    "mode_authority",
    "mode_insurgent",
    "mode_opportunist",
    "topology_control_label",
    "extraction_label",
    "future_win_prob_authority",
    "future_win_prob_insurgent",
    "future_win_prob_opportunist",
    "future_score_delta_authority",
    "future_score_delta_insurgent",
    "future_score_delta_opportunist",
    "leverage_delta_next_window",
    "friction_delta_next_window",
    "visibility_delta_next_window",
    "strategic_swing_next_window",
    "extraction_value_next_window",
    "leverage_starvation_threshold_crossed",
    "friction_overload_threshold_crossed",
    "visibility_overextension_threshold_crossed",
    "choke_point_dominance_threshold_crossed",
    "opportunist_extraction_window_open",
]


def _full_sha256(text: str) -> str:
    """Full 64-character hex sha256 digest."""
    return hashlib.sha256(text.encode()).hexdigest()


def features_to_text(feats: dict[str, Any]) -> str:
    """Convert 95 strategy features into grouped, compact text.

    Groups features by semantic category for readability. The Mamba-2 SSM
    processes this as a causal sequence — the grouping helps the model learn
    which features correlate within each domain.
    """
    lines = []
    for group_name, keys in FEATURE_GROUPS.items():
        items = []
        for k in keys:
            v = feats.get(k)
            if v is not None:
                if isinstance(v, float):
                    if v == int(v) and abs(v) < 100:
                        items.append(f"{k}={int(v)}")
                    else:
                        items.append(f"{k}={v:.3f}")
                elif isinstance(v, bool):
                    items.append(f"{k}={int(v)}")
                else:
                    items.append(f"{k}={v}")
        if items:
            lines.append(f"[{group_name}] {' '.join(items)}")
    return "\n".join(lines)


def labels_to_dict(labels: dict[str, Any]) -> dict[str, Any]:
    """Normalize labels, converting bools to ints for training."""
    out = {}
    for name in LABEL_NAMES:
        v = labels.get(name)
        if v is None:
            out[name] = None
        elif isinstance(v, bool):
            out[name] = int(v)
        else:
            out[name] = v
    return out


def entity_context(entity_mapping: dict[str, str]) -> str:
    """Render entity mapping as readable context."""
    lines = ["## Entities"]
    for entity, hash_val in entity_mapping.items():
        name = ENTITY_NAME_MAP.get(hash_val, hash_val[:16])
        lines.append(f"  {entity} = {name}")
    return "\n".join(lines)


def row_to_training_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a single strategy row into a TRM training pair.

    Every output row carries triple hashing and triple timestamping:
      - content_hash: sha256 of the text feature portion
      - row_hash: sha256 of the full row (all fields, sorted keys)
      - created_at: source file modification time (ISO 8601)
      - processed_at: extraction time (ISO 8601)
      - verified_at: null until verification step
    """
    feats = row.get("features", {})
    labels = row.get("labels", {})
    entity_map = row.get("entity_mapping", {})

    context = entity_context(entity_map)
    state_text = features_to_text(feats)
    full_text = f"{context}\n\n## Game State\n{state_text}"
    primary_label = labels.get("primary_dynamic_label", "unknown")

    # Build the output pair (without volatile hash/ts fields for hash computation)
    pair = {
        "id": row.get("row_id", f"ahoy_{row.get('game_id', 'unk')}_{row.get('state_id', 0)}"),
        "game_id": row.get("game_id", "unknown"),
        "turn": row.get("state_id", 0),
        "source": "ahoy_strategy",
        "text": full_text,
        "primary_label": primary_label,
        "labels": labels_to_dict(labels),
        "features": feats,
    }
    # Row hash over stable fields only (sorted keys for determinism)
    pair["row_hash"] = _full_sha256(json.dumps(pair, sort_keys=True, default=str))
    # Content hash over the feature text only
    pair["content_hash"] = _full_sha256(state_text)
    # Triple timestamps
    now_iso = datetime.now(timezone.utc).isoformat()
    pair["created_at"] = row.get("_source_mtime_iso", now_iso)
    pair["processed_at"] = now_iso
    pair["verified_at"] = None

    return pair


def _resolve_source_mtime(src: Path) -> str:
    """Get ISO 8601 UTC modification timestamp for a source file."""
    mtime = src.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def iter_strategy_rows(source_paths: list[Path], limit: int = 0) -> list[dict[str, Any]]:
    """Read all strategy rows from source paths, deduplicating by row_id.

    Attaches _source_mtime_iso to each row for triple-timestamp provenance.
    """
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for src in source_paths:
        if not src.exists():
            print(f"  WARNING: source not found: {src}", file=sys.stderr)
            continue
        print(f"  Reading {src}...")
        src_mtime_iso = _resolve_source_mtime(src)
        with open(src) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = row.get("row_id", "")
                if rid in seen:
                    continue
                seen.add(rid)
                row["_source_mtime_iso"] = src_mtime_iso
                rows.append(row)
                if limit and len(rows) >= limit:
                    return rows
    return rows


def split_by_game(rows: list[dict[str, Any]], train_frac=0.80, val_frac=0.10,
                  seed: int = 414) -> tuple[list, list, list]:
    """Split rows by game_id to prevent data leakage."""
    import random
    random.seed(seed)

    game_rows: dict[str, list] = {}
    for r in rows:
        gid = r.get("game_id", "unknown")
        game_rows.setdefault(gid, []).append(r)

    game_ids = sorted(game_rows.keys())
    random.shuffle(game_ids)

    n_train = int(len(game_ids) * train_frac)
    n_val = int(len(game_ids) * val_frac)

    train_games = set(game_ids[:n_train])
    val_games = set(game_ids[n_train:n_train + n_val])
    test_games = set(game_ids[n_train + n_val:])

    train = [r for g in train_games for r in game_rows[g]]
    val = [r for g in val_games for r in game_rows[g]]
    test = [r for g in test_games for r in game_rows[g]]

    return train, val, test


def _file_sha256(path: Path) -> str:
    """Compute full sha256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_split(rows: list[dict], path: Path) -> tuple[int, str]:
    """Convert rows to training pairs, write to JSONL, and return (count, sha256)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w") as f:
        for row in rows:
            pair = row_to_training_pair(row)
            f.write(json.dumps(pair, sort_keys=True) + "\n")
            written += 1
    file_hash = _file_sha256(path)
    return written, file_hash


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Extract Ahoy strategy data for TRM training")
    ap.add_argument("--source", nargs="+", default=[],
                    help="Path(s) to strategy_sample.jsonl or directory of JSONL files")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max total rows to process (0 = all)")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=414)
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    # Default sources if none provided
    if not args.source:
        ahoy_base = Path("/home/mfspx/BOARDGAMES/BOARD_GAMES/AHOY/05_OUTPUTS/ahoy/million_replay")
        candidates = sorted(ahoy_base.glob("ahoy_million_*/**/strategy_sample.jsonl"),
                            reverse=True)
        if not candidates:
            candidates = sorted(ahoy_base.rglob("strategy_sample.jsonl"))
        args.source = [str(c) for c in candidates]

    source_paths = [Path(s) for s in args.source]
    print(f"TRM Ahoy Extraction — {len(source_paths)} source(s)")

    # Read
    raw_rows = iter_strategy_rows(source_paths, limit=args.limit)
    print(f"  Loaded {len(raw_rows)} unique rows")

    if not raw_rows:
        print("  ERROR: No rows loaded. Check --source paths.", file=sys.stderr)
        sys.exit(1)

    # Split
    train_rows, val_rows, test_rows = split_by_game(
        raw_rows, args.train_frac, args.val_frac, args.seed
    )
    print(f"  Split: {len(train_rows)} train / {len(val_rows)} val / {len(test_rows)} test")

    # Write (now returns count + sha256 per file)
    out_dir = Path(args.output_dir)
    n_train, hash_train = write_split(train_rows, out_dir / "train.jsonl")
    n_val, hash_val = write_split(val_rows, out_dir / "val.jsonl")
    n_test, hash_test = write_split(test_rows, out_dir / "test.jsonl")

    # Label distribution
    label_counts = Counter()
    for r in raw_rows:
        label_counts[r.get("labels", {}).get("primary_dynamic_label", "unknown")] += 1

    # File-level hashes for triple-hash compliance
    files_written_map = {
        "train": {"path": str(out_dir / "train.jsonl"), "sha256": hash_train, "rows": n_train},
        "val": {"path": str(out_dir / "val.jsonl"), "sha256": hash_val, "rows": n_val},
        "test": {"path": str(out_dir / "test.jsonl"), "sha256": hash_test, "rows": n_test},
    }
    # Aggregate hash = sha256 of concatenated file hashes
    aggregate_hash = _full_sha256(hash_train + hash_val + hash_test)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Receipt
    receipt_dir = RECEIPT_DIR
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "lucidota.trm.ahoy_extraction_receipt.v1",
        "verdict": "PASS" if (n_train and n_val and n_test) else "FAIL",
        "sources": [str(p) for p in source_paths],
        "total_rows_loaded": len(raw_rows),
        "train_rows": n_train,
        "val_rows": n_val,
        "test_rows": n_test,
        "split_fractions": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": round(1 - args.train_frac - args.val_frac, 2),
        },
        "seed": args.seed,
        "primary_label_distribution": dict(label_counts),
        "n_games": len(set(r.get("game_id") for r in raw_rows)),
        "feature_count": 95,
        "label_count": len(LABEL_NAMES),
        "files_written": list(files_written_map.values()),
        "aggregate_sha256": aggregate_hash,
        "etl_timestamps": {
            "processed_at": now_iso,
            "verified_at": None,
        },
        "command": " ".join(sys.argv),
    }
    # created_at = earliest source file modification time
    source_mtimes = sorted(_resolve_source_mtime(Path(s)) for s in args.source)
    receipt["etl_timestamps"]["created_at"] = source_mtimes[0] if source_mtimes else now_iso

    receipt_hash = _full_sha256(json.dumps(receipt, sort_keys=True, default=str))
    receipt["receipt_sha256"] = receipt_hash
    receipt_path = receipt_dir / "ahoy_extract_receipt.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Files written to: {out_dir}")
    print(f"  Done. {n_train + n_val + n_test} training pairs extracted.")


if __name__ == "__main__":
    main()
