#!/usr/bin/env python3
"""
MALKOVICH SHAPE WATCHER — 3-Plane Shape Observation Engine

Plane 1 (SIPHON): Reads existing shape vectors from the Malkovich Siphon output.
  Already produced by malkovich_siphon_orchestrator.py and indy_siphon_extractor.py.

Plane 2 (RIVER ML): Watches fidelity drift per source/lane using the OnlineTwin pattern.
  - Tracks running mean/variance of fidelity per (source, lane)
  - Detects drift when batch fidelity deviates > 2 sigma from running mean
  - Emits shape_drift_event rows

Plane 3 (BYTEWAX): Cross-lane shape correlation via stateful window joins.
  - Computes average shape vector per (source, lane)
  - Cross-correlates all lane pairs via cosine similarity + Euclidean distance
  - Tracks correlation shifts over time (lanes converging or diverging?)

Safety:
  - Writes only to lucidota_learning.* tables (shape_observation, shape_drift_event,
    shape_cross_lane_correlation, shape_watcher_run).
  - Does not mutate canonical graph tables.
  - Does not mutate KORPUS custody rows.
  - File-based ingestion (JSONL), Postgres for durable observation state.

Usage:
  python3 scripts/malkovich_shape_watcher.py --oneshot
  python3 scripts/malkovich_shape_watcher.py --daemon --poll-interval 60
  python3 scripts/malkovich_shape_watcher.py --oneshot --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_SQL = ROOT / "06_SCHEMA" / "008_shape_watcher.sql"
LEARNING_SCHEMA = ROOT / "06_SCHEMA" / "004_learning_reflex.sql"
BYTEWAX_SCHEMA = ROOT / "06_SCHEMA" / "007_bytewax_stream.sql"

SIPHON_SHAPE_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "shape_vectors"
INDY_SHAPE_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "indy_reads"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon" / "receipts"

DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL", "postgresql:///lucidota_state")

# Drift detection thresholds
FIDELITY_DRIFT_SIGMA = 2.0       # Sigma threshold for fidelity drift
SHAPE_SHIFT_THRESHOLD = 0.15     # Cosine distance threshold for shape shift
COLLISION_SURGE_FACTOR = 3.0     # Factor above running mean for collision surge
MIN_SAMPLES_FOR_DRIFT = 10       # Minimum samples before drift detection activates


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ═══════════════════════════════════════════════════════════════════════════════
# PLANE 1: Shape Vector Reader
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeVectorReader:
    """Read shape vectors from Malkovich Siphon JSONL output files.

    Groups vectors by (source, lane) for downstream River/Bytewax observation.
    Source is the data origin (AhoyStrategy, IndyReads).
    Lane is the label subdivision (primary_dynamic_label for Ahoy, book_label for Indy).
    """

    def __init__(self, shape_dirs: list[Path] | None = None):
        self.dirs = shape_dirs or [SIPHON_SHAPE_DIR, INDY_SHAPE_DIR]
        self._cursor: dict[str, int] = {}  # file_path -> last line read

    def _load_cursor(self, cursor_path: Path) -> None:
        if cursor_path.exists():
            try:
                self._cursor = json.loads(cursor_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._cursor = {}

    def _save_cursor(self, cursor_path: Path) -> None:
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cursor_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cursor))
        tmp.replace(cursor_path)

    def discover_files(self) -> list[Path]:
        """Find all shape vector JSONL files."""
        files: list[Path] = []
        for d in self.dirs:
            if d.exists():
                for f in sorted(d.glob("*_shape_vectors.jsonl")):
                    if f.is_file():
                        files.append(f)
        return files

    def read_batch(self, files: list[Path], batch_size: int = 500,
                   cursor_path: Path | None = None) -> dict[tuple[str, str], list[dict]]:
        """Read a batch of shape vectors, grouped by (source, lane).

        Returns dict mapping (source, lane) -> list of shape vector dicts.
        """
        self._load_cursor(cursor_path or (RECEIPT_DIR / "shape_watcher_cursor.json"))

        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        total_read = 0

        # Distribute batch across files: each file gets batch_size // len(files)
        # with remainder going to earlier files. This ensures all sources contribute.
        n_files = len(files)
        per_file = max(1, batch_size // n_files) if n_files else batch_size

        for fpath in files:
            key = str(fpath)
            start_line = self._cursor.get(key, 0)
            file_read = 0

            with open(fpath) as f:
                for i, line in enumerate(f):
                    if i < start_line:
                        continue
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    source = self._extract_source(row, fpath)
                    lane = self._extract_lane(row, source)
                    grouped[(source, lane)].append(row)
                    self._cursor[key] = i + 1
                    total_read += 1
                    file_read += 1

                    if file_read >= per_file and total_read >= batch_size:
                        break

        return dict(grouped)

    def _extract_source(self, row: dict, fpath: Path) -> str:
        """Determine the data source from the shape vector row."""
        if "book_label" in row:
            return f"IndyReads:{row.get('book_label', 'unknown')}"
        if "target_head" in row or "primary_label" in row:
            return "AhoyStrategy"
        if "indy" in str(fpath).lower() or "indy" in str(row.get("source", "")).lower():
            return f"IndyReads:{row.get('book_label', 'unknown')}"
        return row.get("source", "AhoyStrategy")

    def _extract_lane(self, row: dict, source: str) -> str:
        """Determine the lane subdivision within a source."""
        if source.startswith("IndyReads"):
            return row.get("book_label", "unknown")
        if "primary_label" in row:
            return row.get("primary_label", "unknown")
        if "target_head" in row:
            return str(row.get("target_head", "unknown"))
        return source


# ═══════════════════════════════════════════════════════════════════════════════
# PLANE 2: River ML Fidelity Drift Watcher
# ═══════════════════════════════════════════════════════════════════════════════

class RiverFidelityTracker:
    """Online running-mean/variance tracker for fidelity per (source, lane).

    Uses Welford's online algorithm for numerically stable mean/variance.
    Detects drift when batch fidelity deviates > FIDELITY_DRIFT_SIGMA sigma
    from the running mean.
    """

    def __init__(self):
        # Per (source, lane): {count, mean, m2, collision_rate_mean, collision_rate_m2}
        self._state: dict[tuple[str, str], dict[str, float]] = {}

    def observe(self, source: str, lane: str, fidelities: list[float],
                collision_count: int) -> dict | None:
        """Ingest a batch of fidelity values. Returns drift_event dict or None."""
        if not fidelities:
            return None

        batch_mean = sum(fidelities) / len(fidelities)
        batch_collision_rate = collision_count / len(fidelities)
        key = (source, lane)

        if key not in self._state:
            # First observation — initialize state
            self._state[key] = {
                "count": len(fidelities),
                "mean": batch_mean,
                "m2": 0.0,
                "collision_rate_mean": batch_collision_rate,
                "collision_rate_m2": 0.0,
            }
            return None

        state = self._state[key]
        prior_mean = state["mean"]
        prior_m2 = state["m2"]
        prior_collision_mean = state["collision_rate_mean"]
        prior_collision_m2 = state["collision_rate_m2"]
        prior_count = int(state["count"])

        # Update fidelity running stats (Welford batch update)
        for v in fidelities:
            prior_count += 1
            delta = v - state["mean"]
            state["mean"] += delta / prior_count
            delta2 = v - state["mean"]
            state["m2"] += delta * delta2

        state["count"] = float(prior_count)

        # Update collision rate stats
        n_cr = state["count"]
        delta_cr = batch_collision_rate - state["collision_rate_mean"]
        # Weighted update: blend batch into running mean
        alpha = len(fidelities) / n_cr
        state["collision_rate_mean"] += alpha * delta_cr

        # Compute current std dev for drift detection
        if prior_count < MIN_SAMPLES_FOR_DRIFT + len(fidelities):
            return None

        variance = state["m2"] / (state["count"] - 1) if state["count"] > 1 else 0.0
        sigma = math.sqrt(variance) if variance > 0 else 1e-6

        # Check for fidelity drift
        deviation = abs(batch_mean - prior_mean)
        drift_event = None

        if sigma > 1e-9 and deviation > FIDELITY_DRIFT_SIGMA * sigma:
            drift_kind = "fidelity_drop" if batch_mean < prior_mean else "fidelity_spike"
            drift_event = {
                "source": source,
                "lane": lane,
                "drift_kind": drift_kind,
                "prior_fidelity": prior_mean,
                "current_fidelity": batch_mean,
                "delta": batch_mean - prior_mean,
                "sigma": sigma,
                "batch_size": len(fidelities),
                "running_count": int(state["count"]),
            }

        # Check for collision surge
        if prior_collision_mean > 0 and batch_collision_rate > COLLISION_SURGE_FACTOR * prior_collision_mean:
            if drift_event:
                drift_event["collision_surge"] = True
                drift_event["prior_collision_rate"] = prior_collision_mean
                drift_event["current_collision_rate"] = batch_collision_rate
            else:
                drift_event = {
                    "source": source,
                    "lane": lane,
                    "drift_kind": "collision_surge",
                    "prior_fidelity": prior_mean,
                    "current_fidelity": batch_mean,
                    "delta": batch_mean - prior_mean,
                    "sigma": sigma,
                    "batch_size": len(fidelities),
                    "running_count": int(state["count"]),
                    "prior_collision_rate": prior_collision_mean,
                    "current_collision_rate": batch_collision_rate,
                }

        return drift_event

    def get_state(self, source: str, lane: str) -> dict | None:
        key = (source, lane)
        if key not in self._state:
            return None
        s = self._state[key]
        variance = s["m2"] / (s["count"] - 1) if s["count"] > 1 else 0.0
        return {
            "source": source,
            "lane": lane,
            "count": int(s["count"]),
            "fidelity_mean": s["mean"],
            "fidelity_sigma": math.sqrt(variance),
            "collision_rate_mean": s["collision_rate_mean"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PLANE 3: Bytewax Cross-Lane Shape Correlator
# ═══════════════════════════════════════════════════════════════════════════════

class CrossLaneCorrelator:
    """Compute cross-lane shape correlations using cosine similarity.

    For each (source, lane), maintains a running average shape vector.
    Cross-correlates all lane pairs to detect:
      - Which lanes have similar geometric signatures
      - Whether lanes are diverging or converging over time
      - Shape shift events (lane's average vector moves significantly)
    """

    def __init__(self):
        # Per (source, lane): {count, shape_sum[]}
        self._lane_shapes: dict[tuple[str, str], dict[str, Any]] = {}
        # Previous correlation values for shift detection
        self._prior_correlations: dict[tuple, float] = {}

    def ingest(self, source: str, lane: str, shape_vectors: list[list[float]]) -> None:
        """Update the running average shape for a lane."""
        if not shape_vectors:
            return

        key = (source, lane)
        dims = len(shape_vectors[0])
        batch_sum = [0.0] * dims
        for sv in shape_vectors:
            for i, v in enumerate(sv):
                batch_sum[i] += v

        if key not in self._lane_shapes:
            self._lane_shapes[key] = {
                "count": len(shape_vectors),
                "shape_sum": batch_sum,
                "dim": dims,
            }
        else:
            state = self._lane_shapes[key]
            for i in range(dims):
                state["shape_sum"][i] += batch_sum[i]
            state["count"] += len(shape_vectors)

    def get_mean_shape(self, source: str, lane: str) -> list[float] | None:
        key = (source, lane)
        if key not in self._lane_shapes:
            return None
        state = self._lane_shapes[key]
        if state["count"] == 0:
            return None
        return [s / state["count"] for s in state["shape_sum"]]

    def correlate_all(self) -> list[dict]:
        """Compute correlations between all lane pairs.

        Returns list of correlation dicts with cosine_similarity and euclidean_distance.
        """
        lanes = list(self._lane_shapes.keys())
        results = []

        for i in range(len(lanes)):
            for j in range(i + 1, len(lanes)):
                src_a, lane_a = lanes[i]
                src_b, lane_b = lanes[j]

                shape_a = self.get_mean_shape(src_a, lane_a)
                shape_b = self.get_mean_shape(src_b, lane_b)

                if shape_a is None or shape_b is None:
                    continue

                cosine_sim = _cosine_similarity(shape_a, shape_b)
                euclidean_dist = _euclidean_distance(shape_a, shape_b)

                pair_key = (src_a, lane_a, src_b, lane_b)
                prior = self._prior_correlations.get(pair_key)
                shift = None
                if prior is not None:
                    delta = abs(cosine_sim - prior)
                    if delta > SHAPE_SHIFT_THRESHOLD:
                        shift = {
                            "prior_cosine": prior,
                            "current_cosine": cosine_sim,
                            "delta": delta,
                            "direction": "converging" if cosine_sim > prior else "diverging",
                        }

                results.append({
                    "source_a": src_a,
                    "lane_a": lane_a,
                    "source_b": src_b,
                    "lane_b": lane_b,
                    "cosine_similarity": cosine_sim,
                    "euclidean_distance": euclidean_dist,
                    "sample_count_a": self._lane_shapes[lanes[i]]["count"],
                    "sample_count_b": self._lane_shapes[lanes[j]]["count"],
                    "shape_shift": shift,
                })

                self._prior_correlations[pair_key] = cosine_sim

        return results


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_schema(conn) -> None:
    """Create shape watcher tables if they don't exist."""
    import psycopg
    with conn.cursor() as cur:
        for schema_file in (LEARNING_SCHEMA, BYTEWAX_SCHEMA, SCHEMA_SQL):
            if schema_file.exists():
                cur.execute(schema_file.read_text(encoding="utf-8"))


def _insert_observation(conn, source: str, lane: str, fidelities: list[float],
                        shape_vectors: list[list[float]], collision_count: int) -> str | None:
    """Insert a shape_observation row. Returns observation_id or None."""
    if not fidelities:
        return None

    import psycopg
    avg_fid = sum(fidelities) / len(fidelities)
    min_fid = min(fidelities)
    max_fid = max(fidelities)
    collision_rate = collision_count / len(fidelities) if fidelities else 0.0

    # Compute shape mean and variance
    dims = len(shape_vectors[0]) if shape_vectors else 64
    shape_mean = [0.0] * dims
    shape_var = [0.0] * dims
    if shape_vectors:
        for sv in shape_vectors:
            for i, v in enumerate(sv):
                shape_mean[i] += v
        for i in range(dims):
            shape_mean[i] /= len(shape_vectors)

        for sv in shape_vectors:
            for i, v in enumerate(sv):
                diff = v - shape_mean[i]
                shape_var[i] += diff * diff
        for i in range(dims):
            shape_var[i] /= max(len(shape_vectors) - 1, 1)

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lucidota_learning.shape_observation
                (source, lane, batch_size, avg_fidelity, min_fidelity, max_fidelity,
                 collision_count, collision_rate, shape_mean, shape_variance,
                 dim_count, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING observation_id::text
        """, (
            source, lane, len(fidelities), avg_fid, min_fid, max_fid,
            collision_count, collision_rate,
            [float(x) for x in shape_mean],
            [float(x) for x in shape_var],
            dims,
            json.dumps({"shape_count": len(shape_vectors)}),
        ))
        row = cur.fetchone()
        return row[0] if row else None


def _insert_drift_event(conn, drift: dict, prior_obs_id: str | None,
                        current_obs_id: str | None) -> str | None:
    """Insert a shape_drift_event row."""
    import psycopg
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lucidota_learning.shape_drift_event
                (source, lane, drift_kind, prior_fidelity, current_fidelity,
                 delta, prior_observation_id, current_observation_id, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s::uuid, %s::uuid, %s::jsonb)
            RETURNING drift_id::text
        """, (
            drift["source"], drift["lane"], drift["drift_kind"],
            drift["prior_fidelity"], drift["current_fidelity"],
            drift["delta"],
            prior_obs_id, current_obs_id,
            json.dumps({k: v for k, v in drift.items()
                       if k not in ("source", "lane", "drift_kind",
                                    "prior_fidelity", "current_fidelity", "delta")}),
        ))
        row = cur.fetchone()
        return row[0] if row else None


def _insert_correlation(conn, corr: dict) -> str | None:
    """Insert a shape_cross_lane_correlation row."""
    import psycopg
    with conn.cursor() as cur:
        detail = {}
        if corr.get("shape_shift"):
            detail["shape_shift"] = corr["shape_shift"]
        cur.execute("""
            INSERT INTO lucidota_learning.shape_cross_lane_correlation
                (source_a, lane_a, source_b, lane_b, cosine_similarity,
                 euclidean_distance, sample_count_a, sample_count_b,
                 window_start, window_end, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now(), now(), %s::jsonb)
            ON CONFLICT (source_a, lane_a, source_b, lane_b, window_start) DO UPDATE
            SET cosine_similarity = EXCLUDED.cosine_similarity,
                euclidean_distance = EXCLUDED.euclidean_distance,
                sample_count_a = EXCLUDED.sample_count_a,
                sample_count_b = EXCLUDED.sample_count_b,
                window_end = now(),
                detail = EXCLUDED.detail
            RETURNING correlation_id::text
        """, (
            corr["source_a"], corr["lane_a"],
            corr["source_b"], corr["lane_b"],
            corr["cosine_similarity"], corr["euclidean_distance"],
            corr["sample_count_a"], corr["sample_count_b"],
            json.dumps(detail),
        ))
        row = cur.fetchone()
        return row[0] if row else None


def _insert_run_receipt(conn, status: str, mode: str, sources: int,
                        batches: int, drifts: int, cross: int,
                        detail: dict) -> str | None:
    """Insert a shape_watcher_run receipt row."""
    import psycopg
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO lucidota_learning.shape_watcher_run
                (status, mode, sources_scanned, batches_ingested,
                 drift_events_emitted, cross_lane_pairs_computed, detail)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING run_id::text
        """, (status, mode, sources, batches, drifts, cross, json.dumps(detail)))
        row = cur.fetchone()
        return row[0] if row else None


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_observation(db_url: str, batch_size: int = 500) -> dict[str, Any]:
    """Run one complete observation cycle across all 3 planes.

    Returns a receipt dict summarizing what was observed.
    """
    import psycopg

    reader = ShapeVectorReader()
    river = RiverFidelityTracker()
    cross_lane = CrossLaneCorrelator()

    files = reader.discover_files()
    if not files:
        return {
            "verdict": "IDLE",
            "sources_found": 0,
            "reason": "no_shape_vector_files_found",
        }

    # Read batches grouped by (source, lane)
    grouped = reader.read_batch(files, batch_size=batch_size)

    drift_events: list[dict] = []
    correlation_results: list[dict] = []
    observation_ids: dict[tuple[str, str], str] = {}
    batches_ingested = 0
    sources_scanned = len(files)

    # Connect to DB
    conn = psycopg.connect(db_url)

    try:
        _ensure_schema(conn)

        for (source, lane), rows in grouped.items():
            fidelities = [r.get("fidelity", 1.0) for r in rows]
            shape_vectors = [r.get("shape_vector", []) for r in rows]
            collision_count = sum(1 for r in rows if r.get("collision"))

            # Plane 1 + 2: Insert observation and check for drift
            obs_id = _insert_observation(
                conn, source, lane, fidelities, shape_vectors, collision_count,
            )
            if obs_id:
                observation_ids[(source, lane)] = obs_id
            batches_ingested += 1

            # Plane 2: River fidelity drift check
            drift = river.observe(source, lane, fidelities, collision_count)
            if drift:
                # Find prior observation for this lane
                prior_key = (source, lane)
                drift["observation_id"] = obs_id
                drift_events.append(drift)

            # Plane 3: Cross-lane correlation
            if shape_vectors:
                cross_lane.ingest(source, lane, shape_vectors)

        # After all lanes ingested, compute cross-correlations
        correlation_results = cross_lane.correlate_all()

        # Insert drift events (now that we have observation IDs)
        drifts_inserted = 0
        for drift in drift_events:
            _insert_drift_event(conn, drift, None, drift.get("observation_id"))
            drifts_inserted += 1

        # Insert correlation results
        cross_inserted = 0
        for corr in correlation_results:
            _insert_correlation(conn, corr)
            cross_inserted += 1

        # Build lane states for receipt
        lane_states = {}
        for (source, lane) in grouped:
            state = river.get_state(source, lane)
            if state:
                lane_states[f"{source}/{lane}"] = state

        # Cross-lane summary
        cross_summary = []
        for corr in correlation_results:
            cross_summary.append({
                "pair": f"{corr['source_a']}/{corr['lane_a']} <-> {corr['source_b']}/{corr['lane_b']}",
                "cosine_similarity": round(corr["cosine_similarity"], 4),
                "euclidean_distance": round(corr["euclidean_distance"], 4),
                "shift": corr.get("shape_shift"),
            })

        status = "succeeded"
        if drift_events:
            status = "partial"  # drift events mean something shifted — worth investigating

        detail = {
            "lane_states": lane_states,
            "cross_lane_correlations": cross_summary,
            "drift_events_count": len(drift_events),
            "drift_events": drift_events if drift_events else None,
        }

        run_id = _insert_run_receipt(
            conn, status, "oneshot", sources_scanned,
            batches_ingested, drifts_inserted, cross_inserted, detail,
        )
        conn.commit()

        receipt = {
            "schema": "lucidota.malkovich.shape_watcher_receipt.v1",
            "verdict": status.upper(),
            "run_id": run_id,
            "sources_scanned": sources_scanned,
            "source_files": [str(f) for f in files],
            "batches_ingested": batches_ingested,
            "drift_events_emitted": drifts_inserted,
            "cross_lane_pairs_computed": cross_inserted,
            "lane_states": lane_states,
            "cross_lane_correlations": cross_summary,
            "drift_events": drift_events if drift_events else [],
            "timestamp": now_z(),
        }

        # Write file receipt
        file_receipt_path = write_file_receipt("shape_watcher", receipt)
        receipt["file_receipt"] = str(file_receipt_path)

        return receipt

    finally:
        conn.close()


def write_file_receipt(name: str, data: dict[str, Any]) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / f"{name}_{now_z().replace(':', '')}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def daemon_loop(db_url: str, poll_interval: int, batch_size: int) -> None:
    """Run observation in a continuous daemon loop."""
    print(f"SHAPE WATCHER DAEMON — polling every {poll_interval}s")
    print(f"  DB: {db_url}")
    print(f"  Batch size: {batch_size}")
    print()

    cycle = 0
    while True:
        cycle += 1
        print(f"[{now_z()}] Cycle {cycle} — scanning for shape vectors...")
        try:
            receipt = run_observation(db_url, batch_size=batch_size)
            verdict = receipt.get("verdict", "UNKNOWN")
            batches = receipt.get("batches_ingested", 0)
            drifts = receipt.get("drift_events_emitted", 0)
            cross = receipt.get("cross_lane_pairs_computed", 0)
            print(f"  {verdict}: {batches} batches, {drifts} drifts, {cross} cross-lane pairs")

            if drifts > 0:
                for de in receipt.get("drift_events", []):
                    print(f"  DRIFT: [{de['drift_kind']}] {de['source']}/{de['lane']} "
                          f"fidelity {de['prior_fidelity']:.4f} → {de['current_fidelity']:.4f} "
                          f"(Δ={de['delta']:+.4f})")

            for corr in receipt.get("cross_lane_correlations", [])[:5]:
                print(f"  CORR: {corr['pair']} cos={corr['cosine_similarity']:.4f}")

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

        time.sleep(poll_interval)


def main():
    ap = argparse.ArgumentParser(
        description="MALKOVICH SHAPE WATCHER — 3-Plane Shape Observation Engine"
    )
    mode_group = ap.add_mutually_exclusive_group()
    mode_group.add_argument("--oneshot", action="store_true", default=True,
                            help="Run one observation cycle (default)")
    mode_group.add_argument("--daemon", action="store_true",
                            help="Run in continuous daemon mode")
    ap.add_argument("--db-url", default=DB_URL,
                    help=f"Postgres connection URL (default: {DB_URL})")
    ap.add_argument("--batch-size", type=int, default=500,
                    help="Max shape vectors per observation batch")
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between daemon cycles (default: 60)")
    ap.add_argument("--json", action="store_true",
                    help="Output receipt as JSON to stdout")
    ap.add_argument("--shape-dir", nargs="+", default=[],
                    help="Additional shape vector directories to scan")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan files but don't write to DB")
    args = ap.parse_args()

    if args.daemon:
        daemon_loop(args.db_url, args.poll_interval, args.batch_size)
        return 0

    # Oneshot mode
    print(f"MALKOVICH SHAPE WATCHER — Oneshot observation")
    print(f"  DB: {args.db_url}")
    print(f"  Batch size: {args.batch_size}")
    print()

    if args.shape_dir:
        reader = ShapeVectorReader([Path(d) for d in args.shape_dir])
        files = reader.discover_files()
        print(f"  Custom dirs: {args.shape_dir}")
    else:
        reader = ShapeVectorReader()
        files = reader.discover_files()

    print(f"  Sources found: {len(files)}")
    for f in files:
        line_count = sum(1 for _ in open(f)) if f.exists() else 0
        print(f"    {f} ({line_count} rows)")

    if args.dry_run:
        print("\n  DRY RUN — no DB writes performed.")
        return 0

    receipt = run_observation(args.db_url, batch_size=args.batch_size)

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    else:
        print(f"\n  Verdict: {receipt.get('verdict')}")
        print(f"  Batches: {receipt.get('batches_ingested')}")
        print(f"  Drift events: {receipt.get('drift_events_emitted')}")
        print(f"  Cross-lane pairs: {receipt.get('cross_lane_pairs_computed')}")
        for de in receipt.get("drift_events", []):
            print(f"  DRIFT: [{de['drift_kind']}] {de['source']}/{de['lane']}")
        for corr in receipt.get("cross_lane_correlations", [])[:8]:
            print(f"  CORR: {corr['pair']} cos={corr['cosine_similarity']:.4f}")

    print(f"\n  Receipt: {receipt.get('file_receipt', 'not written')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
