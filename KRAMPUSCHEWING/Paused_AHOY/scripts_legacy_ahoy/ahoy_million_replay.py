#!/usr/bin/env python3
"""Large Ahoy replay / adapter / model-suite orchestrator with receipts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGETS_DEFAULT = [
    "primary_dynamic_label",
    "winning_entity_role",
    "mode_authority",
    "mode_insurgent",
    "mode_opportunist",
    "topology_control_label",
    "extraction_label",
    "leverage_starvation_threshold_crossed",
    "friction_overload_threshold_crossed",
    "visibility_overextension_threshold_crossed",
    "choke_point_dominance_threshold_crossed",
    "opportunist_extraction_window_open",
]

PHASE_DATASETS_DEFAULT = {
    "first100k": "05_OUTPUTS/ahoy/training/ahoy_training_rows_20260518T090336457315Z.jsonl",
    "next1k": "05_OUTPUTS/ahoy/training/ahoy_training_rows_20260518T193211959637Z.jsonl",
    "next10k": "05_OUTPUTS/ahoy/training/ahoy_training_rows_20260518T195113338585Z.jsonl",
}


def utc_stamp() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def stable_int(obj: Any) -> int:
    data = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return int(hashlib.sha256(data).hexdigest()[:16], 16)


def open_text(path: Path, mode: str = "rt"):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode.replace("t", ""), encoding="utf-8") if "t" in mode else path.open(mode)


class Reservoir:
    def __init__(self, cap: int, seed: int) -> None:
        self.cap = max(0, cap)
        self.random = random.Random(seed)
        self.n = 0
        self.rows: list[Any] = []

    def add(self, row: Any) -> None:
        self.n += 1
        if self.cap <= 0:
            return
        if len(self.rows) < self.cap:
            self.rows.append(row)
            return
        j = self.random.randrange(self.n)
        if j < self.cap:
            self.rows[j] = row


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, bool)) and not isinstance(x, str)


def _numeric_features(feats: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in feats.items():
        if _is_number(v):
            out[k] = float(v)
    return out


class GameComposite:
    def __init__(self, game_id: str, seed: int | None = None) -> None:
        self.game_id = game_id
        self.seed = seed
        self.rows = 0
        self.faction_rows: Counter[str] = Counter()
        self.primary_dynamic: Counter[str] = Counter()
        self.winning_role: Counter[str] = Counter()
        self.topology: Counter[str] = Counter()
        self.extraction: Counter[str] = Counter()
        self.modes: dict[str, Counter[str]] = {
            "authority": Counter(),
            "insurgent": Counter(),
            "opportunist": Counter(),
        }
        self.bool_counts: Counter[str] = Counter()
        self.num_sum: defaultdict[str, float] = defaultdict(float)
        self.num_max: dict[str, float] = {}
        self.num_min: dict[str, float] = {}
        self.num_seen: Counter[str] = Counter()
        self.final_scores: dict[str, Any] = {}
        self.final_winner_class: str | None = None

    def add_strategy_row(self, sr: dict[str, Any]) -> None:
        self.rows += 1
        feats = sr.get("features", {}) if isinstance(sr.get("features"), dict) else {}
        labels = sr.get("labels", {}) if isinstance(sr.get("labels"), dict) else {}
        faction = feats.get("active_entity_role") or feats.get("faction") or "unknown"
        self.faction_rows[str(faction)] += 1
        for field, counter in [
            ("primary_dynamic_label", self.primary_dynamic),
            ("winning_entity_role", self.winning_role),
            ("topology_control_label", self.topology),
            ("extraction_label", self.extraction),
        ]:
            if labels.get(field) is not None:
                counter[str(labels.get(field))] += 1
        if labels.get("mode_authority") is not None:
            self.modes["authority"][str(labels.get("mode_authority"))] += 1
        if labels.get("mode_insurgent") is not None:
            self.modes["insurgent"][str(labels.get("mode_insurgent"))] += 1
        if labels.get("mode_opportunist") is not None:
            self.modes["opportunist"][str(labels.get("mode_opportunist"))] += 1
        for field in TARGETS_DEFAULT:
            val = labels.get(field)
            if isinstance(val, bool) and val:
                self.bool_counts[field] += 1
        # Keep compact means/min/max for the asymmetric role primitive scores.
        for k, v in _numeric_features(feats).items():
            if (
                k.endswith("_score")
                or "asymmetry" in k
                or "pressure" in k
                or "delta" in k
                or k
                in {
                    "round_number",
                    "turn_number",
                    "authority_leverage_score",
                    "insurgent_leverage_score",
                    "opportunist_leverage_score",
                    "third_party_extraction_score",
                    "choke_point_control_score",
                    "visibility_asymmetry",
                    "friction_delta_by_entity",
                    "leverage_delta_authority",
                    "leverage_delta_insurgent",
                    "leverage_delta_opportunist",
                }
            ):
                fv = float(v)
                self.num_sum[k] += fv
                self.num_seen[k] += 1
                self.num_max[k] = fv if k not in self.num_max else max(self.num_max[k], fv)
                self.num_min[k] = fv if k not in self.num_min else min(self.num_min[k], fv)
        raw_label = sr.get("provenance", {}).get("raw_label") if isinstance(sr.get("provenance"), dict) else None
        if isinstance(raw_label, dict):
            out = raw_label.get("outcome") if isinstance(raw_label.get("outcome"), dict) else {}
            if out:
                self.final_scores = out.get("final_scores") or self.final_scores
                self.final_winner_class = out.get("final_winner_class") or self.final_winner_class

    def as_row(self, phase: str) -> dict[str, Any]:
        def top(c: Counter[str]) -> str | None:
            return c.most_common(1)[0][0] if c else None

        means = {k: self.num_sum[k] / self.num_seen[k] for k in self.num_seen}
        return {
            "schema": "lucidota.ahoy.game_composite.v1",
            "phase": phase,
            "game_id": self.game_id,
            "seed": self.seed,
            "rows_adapted": self.rows,
            "faction_rows": dict(self.faction_rows),
            "labels": {
                "primary_dynamic_majority": top(self.primary_dynamic),
                "winning_entity_role_majority": top(self.winning_role),
                "topology_control_majority": top(self.topology),
                "extraction_majority": top(self.extraction),
                "mode_authority_majority": top(self.modes["authority"]),
                "mode_insurgent_majority": top(self.modes["insurgent"]),
                "mode_opportunist_majority": top(self.modes["opportunist"]),
                "threshold_true_counts": dict(self.bool_counts),
                "primary_dynamic_counts": dict(self.primary_dynamic),
                "winning_role_counts": dict(self.winning_role),
            },
            "features": {
                **{f"mean.{k}": v for k, v in means.items()},
                **{f"max.{k}": v for k, v in self.num_max.items()},
                **{f"min.{k}": v for k, v in self.num_min.items()},
            },
            "outcome": {
                "final_winner_class": self.final_winner_class,
                "final_scores": self.final_scores,
            },
        }


def _group_raw_file_rows(dataset: Path) -> Iterable[list[dict[str, Any]]]:
    current_gid: str | None = None
    buf: list[dict[str, Any]] = []
    with (gzip.open(dataset, "rt", encoding="utf-8") if dataset.suffix == ".gz" else dataset.open("r", encoding="utf-8")) as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            gid = str(row.get("game_id", "unknown"))
            if current_gid is None:
                current_gid = gid
            if gid != current_gid and buf:
                yield buf
                buf = []
                current_gid = gid
            buf.append(row)
    if buf:
        yield buf


def adapt_raw_game_rows(raws: list[dict[str, Any]], feature_schema: dict[str, Any], future_window: int) -> tuple[list[dict[str, Any]], list[str]]:
    from ahoy_sim.training.strategy_dataset import raw_row_to_strategy_training_row

    out: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, raw in enumerate(raws):
        prior = raws[i - 1] if i > 0 and raws[i - 1].get("game_id") == raw.get("game_id") else None
        # Look ahead within game only.
        fut = raw
        for j in range(min(len(raws) - 1, i + future_window), i, -1):
            if raws[j].get("game_id") == raw.get("game_id"):
                fut = raws[j]
                break
        try:
            sr = raw_row_to_strategy_training_row(raw, feature_schema, prior_raw_row=prior, future_raw_row=fut, future_window=future_window)
            # Keep final outcome provenance for compact game composites without leaking into features.
            if isinstance(raw.get("label"), dict):
                sr.setdefault("provenance", {})["raw_label"] = raw.get("label")
            out.append(sr)
        except Exception as exc:
            if len(errors) < 20:
                errors.append(f"row_{i}:{type(exc).__name__}:{exc}")
    return out, errors


def process_existing_dataset(
    phase: str,
    dataset: Path,
    out_dir: Path,
    *,
    sample_cap: int,
    ontology_sample_cap: int,
    composite_cap: int | None,
    future_window: int,
    seed: int,
) -> dict[str, Any]:
    from ahoy_sim.training.strategy_dataset import default_feature_schema
    from ahoy_sim.ontology.strategy_adapter import ontology_packet_from_training_row

    feature_schema = default_feature_schema()
    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    samples = Reservoir(sample_cap, seed)
    onto_samples = Reservoir(ontology_sample_cap, seed + 17)
    composite_path = phase_dir / "game_composites.jsonl.gz"
    rows_seen = rows_ok = rows_failed = games = 0
    label_counts: Counter[str] = Counter()
    blockers: list[str] = []
    with gzip.open(composite_path, "wt", encoding="utf-8") as cf:
        for game_rows in _group_raw_file_rows(dataset):
            games += 1
            srows, errs = adapt_raw_game_rows(game_rows, feature_schema, future_window)
            rows_seen += len(game_rows)
            rows_ok += len(srows)
            rows_failed += max(0, len(game_rows) - len(srows))
            blockers.extend(errs[: max(0, 20 - len(blockers))])
            comp = GameComposite(str(game_rows[0].get("game_id", "unknown")))
            for sr in srows:
                label_counts[str(sr.get("labels", {}).get("primary_dynamic_label", "unknown"))] += 1
                comp.add_strategy_row(sr)
                samples.add(sr)
                if onto_samples.n < ontology_sample_cap or ontology_sample_cap > 0:
                    # Reservoir.add decides whether it is kept, but building packets for every row is the adapter wiring proof.
                    onto_samples.add(ontology_packet_from_training_row(sr))
            if composite_cap is None or games <= composite_cap:
                cf.write(json.dumps(comp.as_row(phase), sort_keys=True) + "\n")
    sample_path = phase_dir / "strategy_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as sf:
        for row in samples.rows:
            sf.write(json.dumps(row, sort_keys=True) + "\n")
    onto_path = phase_dir / "ontology_packet_sample.jsonl"
    with onto_path.open("w", encoding="utf-8") as of:
        for row in onto_samples.rows:
            of.write(json.dumps(row, sort_keys=True) + "\n")
    rec = {
        "schema": "lucidota.ahoy.phase_adapter_receipt.v1",
        "phase": phase,
        "source_dataset": str(dataset),
        "games_seen": games,
        "raw_rows_seen": rows_seen,
        "strategy_rows_adapted": rows_ok,
        "ontology_packets_generated": rows_ok,
        "rows_failed": rows_failed,
        "future_window": future_window,
        "sample_path": str(sample_path),
        "sample_rows": len(samples.rows),
        "sample_universe_rows": samples.n,
        "ontology_packet_sample_path": str(onto_path),
        "ontology_packet_sample_rows": len(onto_samples.rows),
        "composite_path": str(composite_path),
        "label_counts_primary_dynamic": dict(label_counts),
        "blockers": blockers,
        "verdict": "PASS" if rows_ok and rows_failed == 0 else "PARTIAL_FAIL" if rows_ok else "FAIL",
    }
    json_dump(phase_dir / "adapter_receipt.json", rec)
    return rec


def worker_replay(
    phase: str,
    start: int,
    stop: int,
    base_seed: int,
    max_rounds: int,
    future_window: int,
    sample_cap: int,
    ontology_sample_cap: int,
    out_dir: Path,
) -> dict[str, Any]:
    from ahoy_sim.cli import build_policies
    from ahoy_sim.engine.game import run_game, winner_summary
    from ahoy_sim.training.strategy_dataset import default_feature_schema
    from ahoy_sim.ontology.strategy_adapter import ontology_packet_from_training_row

    feature_schema = default_feature_schema()
    shard = f"{start:07d}_{stop:07d}"
    phase_dir = out_dir / phase / "worker_parts"
    phase_dir.mkdir(parents=True, exist_ok=True)
    samples = Reservoir(sample_cap, base_seed + start)
    onto_samples = Reservoir(ontology_sample_cap, base_seed + start + 17)
    composite_path = phase_dir / f"composites_{shard}.jsonl.gz"
    wins: Counter[str] = Counter()
    score_sum: Counter[str] = Counter()
    score_min: dict[str, float] = {}
    score_max: dict[str, float] = {}
    rows_seen = rows_ok = rows_failed = illegal = exceptions = 0
    blockers: list[str] = []
    label_counts: Counter[str] = Counter()
    policies = build_policies("river", "strong", "strong", base_seed + start)
    with gzip.open(composite_path, "wt", encoding="utf-8") as cf:
        for game_no in range(start, stop):
            seed = base_seed + game_no
            state, raw_rows_obj, metrics = run_game(seed, policies, max_rounds=max_rounds, training_writer=None)
            summary = winner_summary(state)
            wins[summary["winner_class"]] += 1
            for f, sc in summary.get("scores", {}).items():
                v = float(sc)
                score_sum[f] += v
                score_min[f] = v if f not in score_min else min(score_min[f], v)
                score_max[f] = v if f not in score_max else max(score_max[f], v)
            illegal += int(metrics.get("illegal_action_count", 0))
            exceptions += int(metrics.get("engine_exceptions", 0))
            raw_rows = [r.as_dict() for r in raw_rows_obj]
            srows, errs = adapt_raw_game_rows(raw_rows, feature_schema, future_window)
            rows_seen += len(raw_rows)
            rows_ok += len(srows)
            rows_failed += max(0, len(raw_rows) - len(srows))
            blockers.extend(errs[: max(0, 20 - len(blockers))])
            comp = GameComposite(str(state.game_id), seed=seed)
            for sr in srows:
                label_counts[str(sr.get("labels", {}).get("primary_dynamic_label", "unknown"))] += 1
                comp.add_strategy_row(sr)
                samples.add(sr)
                onto_samples.add(ontology_packet_from_training_row(sr))
            row = comp.as_row(phase)
            row["outcome"]["final_winner_class"] = summary["winner_class"]
            row["outcome"]["final_scores"] = summary.get("scores", {})
            cf.write(json.dumps(row, sort_keys=True) + "\n")
    sample_path = phase_dir / f"strategy_sample_{shard}.jsonl"
    with sample_path.open("w", encoding="utf-8") as sf:
        for row in samples.rows:
            sf.write(json.dumps(row, sort_keys=True) + "\n")
    onto_path = phase_dir / f"ontology_packet_sample_{shard}.jsonl"
    with onto_path.open("w", encoding="utf-8") as of:
        for row in onto_samples.rows:
            of.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "schema": "lucidota.ahoy.replay_worker_receipt.v1",
        "phase": phase,
        "start": start,
        "stop": stop,
        "games_completed": stop - start,
        "raw_rows_seen": rows_seen,
        "strategy_rows_adapted": rows_ok,
        "ontology_packets_generated": rows_ok,
        "rows_failed": rows_failed,
        "illegal_actions": illegal,
        "engine_exceptions": exceptions,
        "wins": dict(wins),
        "score_sum": dict(score_sum),
        "score_min": score_min,
        "score_max": score_max,
        "label_counts_primary_dynamic": dict(label_counts),
        "sample_path": str(sample_path),
        "sample_rows": len(samples.rows),
        "sample_universe_rows": samples.n,
        "ontology_packet_sample_path": str(onto_path),
        "ontology_packet_sample_rows": len(onto_samples.rows),
        "composite_path": str(composite_path),
        "blockers": blockers,
        "river_policy_metadata": {f: getattr(p, "metadata", lambda: {})() for f, p in policies.items()},
        "verdict": "PASS" if rows_ok and rows_failed == 0 and exceptions == 0 else "DEGRADED" if rows_ok else "FAIL",
    }


def run_replay_phase(
    phase: str,
    games: int,
    workers: int,
    seed: int,
    max_rounds: int,
    future_window: int,
    sample_cap: int,
    ontology_sample_cap: int,
    out_dir: Path,
) -> dict[str, Any]:
    phase_dir = out_dir / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(workers, games))
    chunk = math.ceil(games / workers)
    jobs = [(i, min(games, i + chunk)) for i in range(0, games, chunk)]
    per_worker_sample = max(1, math.ceil(sample_cap / len(jobs)))
    per_worker_onto = max(1, math.ceil(ontology_sample_cap / len(jobs)))
    receipts: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(worker_replay, phase, start, stop, seed, max_rounds, future_window, per_worker_sample, per_worker_onto, out_dir)
            for start, stop in jobs
        ]
        for fut in as_completed(futs):
            receipts.append(fut.result())
            json_dump(phase_dir / "progress_receipt.json", summarize_replay_receipts(phase, receipts, games, seed, max_rounds, time.time() - t0))
    rec = summarize_replay_receipts(phase, receipts, games, seed, max_rounds, time.time() - t0)
    rec["worker_receipts"] = receipts
    rec["strategy_sample_path"] = str(concat_files([Path(r["sample_path"]) for r in receipts], phase_dir / "strategy_sample.jsonl"))
    rec["ontology_packet_sample_path"] = str(concat_files([Path(r["ontology_packet_sample_path"]) for r in receipts], phase_dir / "ontology_packet_sample.jsonl"))
    rec["composite_manifest"] = [r["composite_path"] for r in receipts]
    json_dump(phase_dir / "replay_receipt.json", rec)
    return rec


def summarize_replay_receipts(phase: str, receipts: list[dict[str, Any]], games: int, seed: int, max_rounds: int, elapsed: float) -> dict[str, Any]:
    wins: Counter[str] = Counter()
    scores_sum: Counter[str] = Counter()
    scores_min: dict[str, float] = {}
    scores_max: dict[str, float] = {}
    labels: Counter[str] = Counter()
    blockers: list[str] = []
    for r in receipts:
        wins.update(r.get("wins", {}))
        scores_sum.update(r.get("score_sum", {}))
        for f, v in r.get("score_min", {}).items():
            scores_min[f] = v if f not in scores_min else min(scores_min[f], v)
        for f, v in r.get("score_max", {}).items():
            scores_max[f] = v if f not in scores_max else max(scores_max[f], v)
        labels.update(r.get("label_counts_primary_dynamic", {}))
        blockers.extend(r.get("blockers", [])[: max(0, 20 - len(blockers))])
    completed = sum(int(r.get("games_completed", 0)) for r in receipts)
    score_stats = {
        f: {"mean": scores_sum[f] / max(1, completed), "min": scores_min.get(f), "max": scores_max.get(f)}
        for f in sorted(scores_sum)
    }
    return {
        "schema": "lucidota.ahoy.replay_phase_receipt.v1",
        "phase": phase,
        "games_requested": games,
        "games_completed": completed,
        "seed": seed,
        "max_rounds": max_rounds,
        "elapsed_seconds": elapsed,
        "raw_rows_seen": sum(int(r.get("raw_rows_seen", 0)) for r in receipts),
        "strategy_rows_adapted": sum(int(r.get("strategy_rows_adapted", 0)) for r in receipts),
        "ontology_packets_generated": sum(int(r.get("ontology_packets_generated", 0)) for r in receipts),
        "rows_failed": sum(int(r.get("rows_failed", 0)) for r in receipts),
        "illegal_actions": sum(int(r.get("illegal_actions", 0)) for r in receipts),
        "engine_exceptions": sum(int(r.get("engine_exceptions", 0)) for r in receipts),
        "win_rates": {k: v / max(1, completed) for k, v in sorted(wins.items())},
        "score_stats": score_stats,
        "label_counts_primary_dynamic": dict(labels),
        "blockers": blockers,
        "verdict": "PASS" if completed == games and not blockers and sum(int(r.get("rows_failed", 0)) for r in receipts) == 0 else "DEGRADED" if completed else "FAIL",
    }


def concat_files(paths: list[Path], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as of:
        for p in paths:
            with p.open("rb") as inf:
                shutil.copyfileobj(inf, of)
    return out


def merge_phase_samples(phase_recs: list[dict[str, Any]], out_path: Path, max_rows: int, seed: int) -> Path:
    reservoir = Reservoir(max_rows, seed)
    for rec in phase_recs:
        p = Path(rec.get("sample_path") or rec.get("strategy_sample_path", ""))
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        reservoir.add(json.loads(line))
                    except Exception:
                        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as of:
        for row in reservoir.rows:
            of.write(json.dumps(row, sort_keys=True) + "\n")
    return out_path


def train_target_grid(rows_path: Path, target: str, out_dir: Path, *, seed: int, quick: bool = False) -> dict[str, Any]:
    from ahoy_sim.training.train_strategy_xgboost import train_strategy_xgboost
    from ahoy_sim.training.export_strategy_treelite import export_strategy_treelite

    grid = [(32, 3), (64, 3)] if quick else [(32, 3), (64, 4), (96, 3)]
    target_dir = out_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for n_estimators, max_depth in grid:
        stem = f"xgb_n{n_estimators}_d{max_depth}"
        model_path = target_dir / f"{stem}.json"
        receipt_path = target_dir / f"{stem}_train_receipt.json"
        rec = train_strategy_xgboost(rows_path, target, model_path, receipt_path, n_estimators=n_estimators, max_depth=max_depth, seed=seed)
        rec["n_estimators"] = n_estimators
        rec["max_depth"] = max_depth
        candidates.append(rec)
    passed = [r for r in candidates if r.get("verdict") == "PASS"]
    best = max(passed, key=lambda r: (float(r.get("macro_f1", 0.0)), float(r.get("accuracy", 0.0)))) if passed else candidates[-1]
    best_model = Path(best.get("model_path", target_dir / "missing.json"))
    final_model = target_dir / f"{target}_best_xgb.json"
    if best_model.exists():
        shutil.copy2(best_model, final_model)
        for suffix in [".vectorizer.pkl", ".labels.pkl"]:
            src = Path(str(best_model) + suffix)
            if src.exists():
                shutil.copy2(src, Path(str(final_model) + suffix))
    treelite_path = target_dir / f"{target}_best_treelite.so"
    treelite_receipt_path = target_dir / f"{target}_best_treelite_receipt.json"
    trec = export_strategy_treelite(final_model, treelite_path, treelite_receipt_path) if final_model.exists() else {"verdict": "BLOCKED", "blockers": ["best_model_missing"], "treelite_exported": False}
    summary = {
        "schema": "lucidota.ahoy.target_grid_training_summary.v1",
        "target": target,
        "rows_path": str(rows_path),
        "candidates": candidates,
        "best": best,
        "best_model": str(final_model) if final_model.exists() else None,
        "treelite": trec,
        "verdict": "PASS" if best.get("verdict") == "PASS" and trec.get("treelite_exported") else "DEGRADED" if best.get("verdict") == "PASS" else "FAIL",
    }
    json_dump(target_dir / "training_summary.json", summary)
    return summary


def train_model_suite(rows_path: Path, targets: list[str], out_dir: Path, *, seed: int, quick: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for target in targets:
        summaries.append(train_target_grid(rows_path, target, out_dir, seed=seed, quick=quick))
    rec = {
        "schema": "lucidota.ahoy.model_suite_receipt.v1",
        "rows_path": str(rows_path),
        "targets": targets,
        "target_summaries": summaries,
        "passed_targets": [s["target"] for s in summaries if s.get("verdict") == "PASS"],
        "treelites_exported": sum(1 for s in summaries if s.get("treelite", {}).get("treelite_exported")),
        "verdict": "PASS" if summaries and all(s.get("verdict") == "PASS" for s in summaries) else "DEGRADED" if any(s.get("verdict") == "PASS" for s in summaries) else "FAIL",
    }
    json_dump(out_dir / "model_suite_receipt.json", rec)
    return rec


def parse_phase_arg(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(PHASE_DATASETS_DEFAULT)
    out = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--phase-dataset must be PHASE=PATH, got {value!r}")
        k, v = value.split("=", 1)
        out[k] = v
    return out


def cmd_worker(args: argparse.Namespace) -> int:
    rec = worker_replay(
        args.phase,
        args.start,
        args.stop,
        args.seed,
        args.max_rounds,
        args.future_window,
        args.sample_cap,
        args.ontology_sample_cap,
        Path(args.out_dir),
    )
    print(json.dumps(rec, sort_keys=True))
    return 0 if rec["verdict"] in {"PASS", "DEGRADED"} else 4


def main() -> int:
    ap = argparse.ArgumentParser(description="Ahoy million-game replay / ontology adapter / XGBoost->Treelite orchestrator")
    sub = ap.add_subparsers(dest="cmd")
    wp = sub.add_parser("worker")
    wp.add_argument("--phase", required=True)
    wp.add_argument("--start", type=int, required=True)
    wp.add_argument("--stop", type=int, required=True)
    wp.add_argument("--seed", type=int, required=True)
    wp.add_argument("--max-rounds", type=int, default=6)
    wp.add_argument("--future-window", type=int, default=3)
    wp.add_argument("--sample-cap", type=int, default=5000)
    wp.add_argument("--ontology-sample-cap", type=int, default=500)
    wp.add_argument("--out-dir", required=True)

    rp = sub.add_parser("run")
    rp.add_argument("--run-id", default=utc_stamp())
    rp.add_argument("--out-root", type=Path, default=ROOT / "05_OUTPUTS/ahoy/million_replay")
    rp.add_argument("--phase-dataset", action="append", help="Initial dataset as PHASE=PATH")
    rp.add_argument("--skip-existing", action="store_true")
    rp.add_argument("--iterations", type=int, default=10)
    rp.add_argument("--games-per-iteration", type=int, default=100000)
    rp.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    rp.add_argument("--seed", type=int, default=414000000)
    rp.add_argument("--max-rounds", type=int, default=6)
    rp.add_argument("--future-window", type=int, default=3)
    rp.add_argument("--sample-cap", type=int, default=60000)
    rp.add_argument("--ontology-sample-cap", type=int, default=5000)
    rp.add_argument("--train-rows-cap", type=int, default=80000)
    rp.add_argument("--targets", default=",".join(TARGETS_DEFAULT))
    rp.add_argument("--quick-grid", action="store_true")
    rp.add_argument("--replay-only", action="store_true")
    args = ap.parse_args()

    if args.cmd == "worker":
        return cmd_worker(args)
    if args.cmd != "run":
        ap.print_help()
        return 2

    out_dir = args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    phase_receipts: list[dict[str, Any]] = []
    master: dict[str, Any] = {
        "schema": "lucidota.ahoy.million_replay_master.v1",
        "run_id": args.run_id,
        "out_dir": str(out_dir),
        "created_at": utc_stamp(),
        "config": vars(args),
        "phase_receipts": [],
        "model_suites": [],
        "games_completed_total": 0,
        "strategy_rows_adapted_total": 0,
        "ontology_packets_generated_total": 0,
        "blockers": [],
        "verdict": "RUNNING",
    }
    json_dump(out_dir / "master_receipt.json", master)

    if not args.skip_existing:
        for phase, rel in parse_phase_arg(args.phase_dataset).items():
            p = Path(rel)
            if not p.is_absolute():
                p = ROOT / p
            rec = process_existing_dataset(
                phase,
                p,
                out_dir,
                sample_cap=args.sample_cap,
                ontology_sample_cap=args.ontology_sample_cap,
                composite_cap=None,
                future_window=args.future_window,
                seed=args.seed + stable_int(phase) % 100000,
            )
            phase_receipts.append(rec)
            master["phase_receipts"].append(rec)
            master["games_completed_total"] += int(rec.get("games_seen", 0))
            master["strategy_rows_adapted_total"] += int(rec.get("strategy_rows_adapted", 0))
            master["ontology_packets_generated_total"] += int(rec.get("ontology_packets_generated", 0))
            master["blockers"].extend(rec.get("blockers", [])[: max(0, 20 - len(master["blockers"]))])
            if not args.replay_only:
                rows = merge_phase_samples([rec], out_dir / phase / "train_sample.jsonl", args.train_rows_cap, args.seed)
                suite = train_model_suite(rows, targets, out_dir / phase / "models", seed=args.seed, quick=args.quick_grid)
                master["model_suites"].append(suite)
            json_dump(out_dir / "master_receipt.json", master)

    for i in range(1, args.iterations + 1):
        phase = f"iter_{i:02d}_100k"
        rec = run_replay_phase(
            phase,
            args.games_per_iteration,
            args.workers,
            args.seed + i * args.games_per_iteration,
            args.max_rounds,
            args.future_window,
            args.sample_cap,
            args.ontology_sample_cap,
            out_dir,
        )
        phase_receipts.append(rec)
        master["phase_receipts"].append(rec)
        master["games_completed_total"] += int(rec.get("games_completed", 0))
        master["strategy_rows_adapted_total"] += int(rec.get("strategy_rows_adapted", 0))
        master["ontology_packets_generated_total"] += int(rec.get("ontology_packets_generated", 0))
        master["blockers"].extend(rec.get("blockers", [])[: max(0, 20 - len(master["blockers"]))])
        if not args.replay_only:
            # Train on cumulative deterministic composite/sample rows up to this iteration.
            rows = merge_phase_samples(phase_receipts, out_dir / phase / "cumulative_train_sample.jsonl", args.train_rows_cap, args.seed + i)
            suite = train_model_suite(rows, targets, out_dir / phase / "models", seed=args.seed + i, quick=args.quick_grid)
            master["model_suites"].append(suite)
        json_dump(out_dir / "master_receipt.json", master)

    master["completed_at"] = utc_stamp()
    master["verdict"] = "PASS" if not master["blockers"] and master["games_completed_total"] >= args.iterations * args.games_per_iteration else "DEGRADED"
    json_dump(out_dir / "master_receipt.json", master)
    print(json.dumps({"verdict": master["verdict"], "out_dir": str(out_dir), "games_completed_total": master["games_completed_total"], "treelites_exported": sum(s.get("treelites_exported", 0) for s in master["model_suites"]), "receipt": str(out_dir / "master_receipt.json")}, sort_keys=True))
    return 0 if master["verdict"] in {"PASS", "DEGRADED"} else 4


if __name__ == "__main__":
    raise SystemExit(main())
