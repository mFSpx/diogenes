#!/usr/bin/env python3
"""
MALKOVICH SIPHON ORCHESTRATOR — RFC-2501 Execution Engine

Wires the 5-stage pipeline:
  1. SIPHON:  Training data → ElasticOntologyCompressor (Rust) → shape_vectors
  2. ROUTE:   shape_vectors → XGBoost classifiers → Treelite C-code (.so)
  3. TRAIN:   Routed data → 5 LoRA adapters on low-bit Bonsai bases
  4. GAUNTLET: Collisions (fidelity<0.85) → Darwin Hammer tournament → evolved algos
  5. DEPLOY:  Compile Treelite .so, mount 5-head hydra, begin live routing

Each stage writes receipts to 05_OUTPUTS/malkovich_siphon/

Usage:
  python3 scripts/malkovich_siphon_orchestrator.py --pipeline all
  python3 scripts/malkovich_siphon_orchestrator.py --pipeline siphon,route
  python3 scripts/malkovich_siphon_orchestrator.py --pipeline siphon --source-strategy /path/to/strategy_sample.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CRATE_DIR = ROOT / "01_REPOS" / "lucidota_resonance"
ELASTIC_BIN = CRATE_DIR / "target" / "release" / "lucidota_elastic_shape"
EVOLVER_BIN = CRATE_DIR / "target" / "release" / "lucidota_resonance_evolver"
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "malkovich_siphon"
RECEIPT_DIR = OUTPUT_DIR / "receipts"
SHAPE_DIR = OUTPUT_DIR / "shape_vectors"
ROUTE_DIR = OUTPUT_DIR / "routers"
LORA_DIR = OUTPUT_DIR / "lora_heads"
GAUNTLET_DIR = OUTPUT_DIR / "gauntlet"

# ── 5-headed hydra model definitions ──────────────────────────────────────
HYDRA_HEADS = {
    "santa_compliance": {
        "head_id": 0,
        "role": "Compliance / Santa — lawful, rule-following, constraint-respecting",
        "base_model": "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "bit_width": "1-bit",
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
    },
    "krampus_automation": {
        "head_id": 1,
        "role": "Automation / Krampus — task execution, scripting, tool use",
        "base_model": "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "bit_width": "1-bit",
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
    },
    "jekyll_research": {
        "head_id": 2,
        "role": "Research / Jekyll — analysis, synthesis, careful reasoning",
        "base_model": "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "bit_width": "1-bit",
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
    },
    "hyde_adversarial": {
        "head_id": 3,
        "role": "Adversarial / Hyde — edge cases, stress testing, counterfactuals",
        "base_model": "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "bit_width": "1-bit",
        "lora_rank": 8,
        "target_modules": ["q_proj", "v_proj"],
    },
    "central_oracle": {
        "head_id": 4,
        "role": "Central Oracle — ternary 1.58-bit master arbitrator",
        "base_model": "03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf",
        "bit_width": "1.58-bit",
        "lora_rank": 16,
        "target_modules": ["q_proj", "v_proj"],
    },
}


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_receipt(name: str, data: dict[str, Any]) -> Path:
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = RECEIPT_DIR / f"{name}_{now_z().replace(':', '')}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def feature_dict_to_signals(feats: dict[str, Any]) -> list[tuple[str, float]]:
    """Convert a feature dict into token:intensity pairs for the compressor."""
    signals = []
    for k, v in feats.items():
        if isinstance(v, (int, float)) and v != 0:
            signals.append((k, float(v)))
    return signals


def run_elastic_compressor(
    signals: list[tuple[str, float]],
    artifact_uuid: str = "",
    source: str = "TrainingData",
    min_dims: int = 8,
    max_dims: int = 128,
    entropy_hint: float = 0.0,
    threshold: float = 0.1,
    fixed_dims: int = 64,
) -> dict[str, Any]:
    """Run the Rust ElasticOntologyCompressor on a set of signals.

    When fixed_dims is set, min_dims=max_dims=fixed_dims, bypassing the adaptive
    dimension suggestion. This ensures all shape vectors have identical length
    for downstream ML training.
    """
    if not artifact_uuid:
        artifact_uuid = str(uuid.uuid4())

    if fixed_dims > 0:
        min_dims = max_dims = fixed_dims

    cmd = [
        str(ELASTIC_BIN),
        "--artifact-uuid", artifact_uuid,
        "--source", source,
        "--min-dims", str(min_dims),
        "--max-dims", str(max_dims),
        "--entropy-hint", str(entropy_hint),
        "--threshold", str(threshold),
    ]
    for token, value in signals:
        cmd.extend(["--signal", f"{token}={value}"])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"elastic_shape failed: {proc.stderr}")
    return json.loads(proc.stdout)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: SIPHON — Compress all training data through the Elastic Compressor
# ═══════════════════════════════════════════════════════════════════════════

def siphon_ahoy_strategy(source_paths: list[Path], limit: int = 0,
                         fixed_dims: int = 64) -> dict[str, Any]:
    """Feed Ahoy strategy rows through the ElasticOntologyCompressor.

    Each strategy row's 95 features become token:intensity signals.
    The compressor produces: shape_vector, residual_vector, fidelity, collision flag.
    """
    SHAPE_DIR.mkdir(parents=True, exist_ok=True)
    shape_out = SHAPE_DIR / "ahoy_shape_vectors.jsonl"
    collision_out = SHAPE_DIR / "ahoy_collisions.jsonl"

    total = 0
    collisions = 0
    fidelity_sum = 0.0
    lane_labels: Counter = Counter()

    with open(shape_out, "w") as f_shape, open(collision_out, "w") as f_coll:
        for src in source_paths:
            if not src.exists():
                continue
            with open(src) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    feats = row.get("features", {})
                    labels = row.get("labels", {})
                    primary = labels.get("primary_dynamic_label", "unknown")

                    signals = feature_dict_to_signals(feats)
                    if not signals:
                        continue

                    try:
                        receipt = run_elastic_compressor(
                            signals,
                            artifact_uuid=row.get("row_id", str(uuid.uuid4())),
                            source="AhoyStrategy",
                            entropy_hint=float(len(signals)) / 95.0,
                            fixed_dims=fixed_dims,
                        )
                    except RuntimeError:
                        continue

                    total += 1
                    fidelity_sum += receipt["fidelity"]

                    # Route to head based on primary label
                    head_id = _label_to_head(primary)
                    lane_labels[head_id] += 1

                    shape_row = {
                        "id": row.get("row_id", ""),
                        "game_id": row.get("game_id", ""),
                        "shape_vector": receipt["shape_vector"],
                        "residual_vector": receipt["residual_vector"],
                        "fidelity": receipt["fidelity"],
                        "collision": receipt["collision"],
                        "dimensions": receipt["dimensions"],
                        "primary_label": primary,
                        "target_head": head_id,
                        "labels": labels,
                    }
                    f_shape.write(json.dumps(shape_row, sort_keys=True) + "\n")

                    if receipt["collision"]:
                        collisions += 1
                        f_coll.write(json.dumps({
                            "id": row.get("row_id", ""),
                            "fidelity": receipt["fidelity"],
                            "residual_mass": receipt["residual_mass"],
                            "primary_label": primary,
                        }, sort_keys=True) + "\n")

                    if limit and total >= limit:
                        break
            if limit and total >= limit:
                break

    receipt_data = {
        "schema": "lucidota.malkovich.siphon_receipt.v1",
        "stage": "siphon",
        "source": "ahoy_strategy",
        "total_processed": total,
        "collisions": collisions,
        "collision_rate": collisions / total if total else 0,
        "avg_fidelity": fidelity_sum / total if total else 0,
        "lane_distribution": dict(lane_labels),
        "shape_vectors_written": str(shape_out),
        "collisions_written": str(collision_out),
        "timestamp": now_z(),
    }
    receipt_path = write_receipt("siphon_ahoy", receipt_data)
    print(f"  SIPHON: {total} rows → {collisions} collisions ({(collisions/total*100 if total else 0):.1f}%)")
    print(f"  Shape vectors: {shape_out}")
    print(f"  Receipt: {receipt_path}")
    return receipt_data


def _label_to_head(label: str) -> int:
    """Map Ahoy primary dynamic labels to hydra head IDs."""
    mapping = {
        "authority_vs_insurgency": 0,  # Santa — rules/structure
        "mixed": 1,                     # Krampus — action/task execution
        "relationship_dependency": 2,   # Jekyll — analysis/research
        "visibility_control": 3,        # Hyde — adversarial/scouting
    }
    return mapping.get(label, 1)  # default to Krampus


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: ROUTE — XGBoost on shape_vectors → Treelite C-code
# ═══════════════════════════════════════════════════════════════════════════

def train_xgboost_router(shape_vectors_path: Path) -> dict[str, Any]:
    """Train an XGBoost classifier on shape_vectors to predict target_head.

    The shape_vector IS the feature set — each dimension is a feature.
    The target is which hydra head (0-4) should handle this input.
    Varying-length vectors are padded to the max dimension found.
    """
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError:
        return {"verdict": "BLOCKED", "blockers": ["xgboost_not_installed"]}

    ROUTE_DIR.mkdir(parents=True, exist_ok=True)

    raw_X = []
    y = []
    with open(shape_vectors_path) as f:
        for line in f:
            row = json.loads(line)
            raw_X.append(row["shape_vector"])
            y.append(row["target_head"])

    # Pad to uniform length
    max_dim = max(len(v) for v in raw_X)
    min_dim = min(len(v) for v in raw_X)
    X = np.zeros((len(raw_X), max_dim), dtype=np.float32)
    for i, v in enumerate(raw_X):
        X[i, :len(v)] = v

    n = len(X)
    split = int(n * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="multi:softmax",
        num_class=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=414,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val) if len(X_val) > 0 else 0.0

    model_path = ROUTE_DIR / "malkovich_router_xgb.json"
    model.save_model(str(model_path))

    receipt = {
        "schema": "lucidota.malkovich.route_receipt.v1",
        "stage": "route",
        "verdict": "PASS" if val_acc > 0.2 else "PARTIAL",
        "model_type": "xgboost.XGBClassifier",
        "n_estimators": 200,
        "max_depth": 6,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "baseline_majority": float(max(Counter(y).values()) / len(y)),
        "model_path": str(model_path),
        "n_features": max_dim,
        "dim_range": f"{min_dim}-{max_dim}",
        "n_classes": 5,
    }
    write_receipt("route_xgboost", receipt)
    print(f"  ROUTE: train_acc={train_acc:.4f} val_acc={val_acc:.4f} model={model_path}")
    return receipt


def export_treelite_router(model_path: Path) -> dict[str, Any]:
    """Export trained XGBoost to Treelite C-code shared library."""
    try:
        import xgboost as xgb
        import treelite
    except ImportError as e:
        return {"verdict": "BLOCKED", "blockers": [f"import_error:{e}"]}

    if not model_path.exists():
        return {"verdict": "BLOCKED", "blockers": ["model_not_found"]}

    b = xgb.Booster()
    b.load_model(str(model_path))
    tl = treelite.frontend.from_xgboost(b)

    so_path = ROUTE_DIR / "malkovich_router.so"

    try:
        import tl2cgen
        tl2cgen.export_lib(tl, toolchain='gcc', libpath=str(so_path),
                           params={}, nthread=2, verbose=False)
        artifact_kind = "native_shared_library"
    except (ImportError, RuntimeError):
        artifact_kind = "treelite_serialized"
        so_path = so_path.with_suffix(".tl")
        tl.serialize(str(so_path))

    receipt = {
        "schema": "lucidota.malkovich.treelite_receipt.v1",
        "stage": "route_treelite",
        "verdict": "PASS",
        "treelite_path": str(so_path),
        "artifact_kind": artifact_kind,
        "note": "15ns routing gate — compiled C decision tree",
    }
    write_receipt("route_treelite", receipt)
    print(f"  TREELITE: {artifact_kind} → {so_path}")
    return receipt


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: TRAIN — 5 LoRA adapters on low-bit Bonsai bases
# ═══════════════════════════════════════════════════════════════════════════

def configure_lora_training() -> list[dict[str, Any]]:
    """Generate training configs for each hydra head.

    We use llama.cpp's finetune or the project's existing LoRA training
    infrastructure (scripts/lucidota_indy_lora_train.py).

    Each head gets:
    - Its routed subset of the training data
    - A LoRA adapter targeting q_proj and v_proj
    - Frozen base weights (1-bit / 1.58-bit)
    """
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    configs = []

    for head_name, spec in HYDRA_HEADS.items():
        base = ROOT / spec["base_model"]
        lora_out = LORA_DIR / f"{head_name}_lora_r{spec['lora_rank']}.gguf"

        config = {
            "head_name": head_name,
            "head_id": spec["head_id"],
            "role": spec["role"],
            "base_model": str(base),
            "base_exists": base.exists(),
            "bit_width": spec["bit_width"],
            "lora_rank": spec["lora_rank"],
            "target_modules": spec["target_modules"],
            "output_path": str(lora_out),
            "training_config": {
                "frozen_base": True,
                "trainable_params_estimate": spec["lora_rank"] * 512 * 2 * len(spec["target_modules"]),
                "optimizer": "adamw",
                "learning_rate": 1e-4,
                "batch_size": 4,
                "gradient_accumulation": 8,
                "epochs": 3,
                "bf16": True,
            },
        }
        configs.append(config)

    receipt = {
        "schema": "lucidota.malkovich.lora_config_receipt.v1",
        "stage": "train_config",
        "verdict": "PASS" if all(c["base_exists"] for c in configs) else "PARTIAL",
        "heads": configs,
        "total_trainable_params_estimate": sum(
            c["training_config"]["trainable_params_estimate"] for c in configs
        ),
    }
    write_receipt("train_lora_config", receipt)
    for c in configs:
        print(f"  TRAIN CONFIG: {c['head_name']} ({c['bit_width']}) r={c['lora_rank']} "
              f"params~={c['training_config']['trainable_params_estimate']}")
    return configs


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: GAUNTLET — Darwin Hammer on collisions
# ═══════════════════════════════════════════════════════════════════════════

def launch_darwin_gauntlet(collisions_path: Path | None = None,
                           rounds: int = 99) -> dict[str, Any]:
    """Launch the Darwin Hammer tournament on collision artifacts.

    Collisions (fidelity < 0.85) are the "friction" that drives evolution.
    The gauntlet mutates routing thresholds, tree depths, and splitmix64 seeds.
    """
    GAUNTLET_DIR.mkdir(parents=True, exist_ok=True)

    if collisions_path and collisions_path.exists():
        n_collisions = sum(1 for _ in open(collisions_path))
    else:
        n_collisions = 0

    # Launch lucidota_resonance_evolver for continuous spectral evolution
    evolver_mode = "daemon" if rounds > 1 else "once"
    evolver_proc = None
    evolution_log = GAUNTLET_DIR / "resonance_evolution.jsonl"

    if rounds > 1:
        # Continuous mode: spawn in background, will run until killed
        evolver_proc = subprocess.Popen(
            [str(EVOLVER_BIN), "--jsonl-out", str(evolution_log)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        evolver_ok = evolver_proc.poll() is None  # still running = good
        evolver_pid = evolver_proc.pid
    else:
        try:
            proc = subprocess.run(
                [str(EVOLVER_BIN), "--once", "--jsonl-out", str(evolution_log)],
                capture_output=True, text=True, timeout=30,
            )
            evolver_ok = proc.returncode == 0
            evolver_pid = None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            evolver_ok = False
            evolver_pid = None

    # Also launch Darwin Hammer if collisions exist and rounds > 0
    hammer_proc = None
    if n_collisions > 0 and rounds > 0:
        hammer_cmd = [
            sys.executable, str(ROOT / "scripts" / "darwin_hammer_tournament.py"),
            "--rounds", str(rounds),
        ]
        hammer_proc = subprocess.Popen(
            hammer_cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    receipt = {
        "schema": "lucidota.malkovich.gauntlet_receipt.v1",
        "stage": "gauntlet",
        "verdict": "PASS" if (evolver_ok or n_collisions > 0) else "IDLE",
        "collisions_available": n_collisions,
        "resonance_evolver": {
            "ran": evolver_ok,
            "mode": evolver_mode,
            "pid": evolver_pid,
            "log": str(evolution_log),
        },
        "darwin_hammer": {
            "launched": hammer_proc is not None,
            "pid": hammer_proc.pid if hammer_proc else None,
            "rounds": rounds,
        },
        "gauntlet_dir": str(GAUNTLET_DIR),
        "note": "Dual-track evolution: spectral (Rust) + algorithmic (Python/LLM). "
                "Collisions fuel both tracks. Kill evolver PID to stop.",
    }
    write_receipt("gauntlet", receipt)
    print(f"  GAUNTLET: {n_collisions} collisions → {'evolving' if evolver_ok else 'idle'}")
    return receipt


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Execute pipeline stages
# ═══════════════════════════════════════════════════════════════════════════

def load_strategy_sources(source_paths: list[str] | None = None) -> list[Path]:
    """Find strategy training data sources."""
    if source_paths:
        return [Path(p) for p in source_paths]

    # Default sources — all Ahoy strategy samples
    ahoy_base = Path("/home/mfspx/BOARDGAMES/BOARD_GAMES/AHOY/05_OUTPUTS/ahoy/million_replay")
    candidates = sorted(ahoy_base.rglob("strategy_sample.jsonl"))
    if not candidates:
        # Fall back to extracted TRM training data
        candidates = sorted((ROOT / "05_OUTPUTS/trm_training/ahoy").glob("*.jsonl"))
    return candidates


def run_pipeline(stages: set[str], source_paths: list[Path], limit: int,
                 gauntlet_rounds: int) -> dict[str, Any]:
    """Execute selected pipeline stages in sequence."""
    results: dict[str, Any] = {"started": now_z(), "stages_run": []}

    shape_path = SHAPE_DIR / "ahoy_shape_vectors.jsonl"
    collision_path = SHAPE_DIR / "ahoy_collisions.jsonl"

    # Stage 1: SIPHON
    if "siphon" in stages:
        print("=" * 60)
        print("STAGE 1: MALKOVICH SIPHON — Compressing training data")
        print("=" * 60)
        results["siphon"] = siphon_ahoy_strategy(source_paths, limit=limit,
                                                     fixed_dims=64)
        results["stages_run"].append("siphon")
    else:
        print("  SIPHON skipped (use --pipeline siphon,all to include)")

    # Stage 2: ROUTE
    if "route" in stages:
        print("\n" + "=" * 60)
        print("STAGE 2: ROUTE — XGBoost → Treelite C-code")
        print("=" * 60)
        if shape_path.exists():
            results["route_xgb"] = train_xgboost_router(shape_path)
            model_path = ROUTE_DIR / "malkovich_router_xgb.json"
            if model_path.exists():
                results["route_treelite"] = export_treelite_router(model_path)
            results["stages_run"].append("route")
        else:
            print("  ROUTE skipped: no shape vectors. Run siphon first.")
    else:
        print("  ROUTE skipped")

    # Stage 3: TRAIN
    if "train" in stages:
        print("\n" + "=" * 60)
        print("STAGE 3: TRAIN — 5 LoRA adapters on low-bit Bonsai bases")
        print("=" * 60)
        results["lora_configs"] = configure_lora_training()
        results["stages_run"].append("train")
    else:
        print("  TRAIN skipped")

    # Stage 4: GAUNTLET
    if "gauntlet" in stages:
        print("\n" + "=" * 60)
        print("STAGE 4: DARWIN GAUNTLET — Evolutionary tournament on collisions")
        print("=" * 60)
        results["gauntlet"] = launch_darwin_gauntlet(collision_path, rounds=gauntlet_rounds)
        results["stages_run"].append("gauntlet")
    else:
        print("  GAUNTLET skipped")

    # Stage 5: WATCH — 3-plane shape observation (River + Bytewax)
    if "watch" in stages:
        print("\n" + "=" * 60)
        print("STAGE 5: SHAPE WATCHER — River fidelity drift + Bytewax cross-lane correlation")
        print("=" * 60)
        try:
            watcher_script = ROOT / "scripts" / "malkovich_shape_watcher.py"
            proc = subprocess.run(
                [sys.executable, str(watcher_script), "--oneshot", "--json", "--batch-size", "25000"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode == 0:
                results["shape_watcher"] = json.loads(proc.stdout.strip().splitlines()[-1])
                print(f"  WATCH: {results['shape_watcher'].get('verdict', 'UNKNOWN')} "
                      f"— {results['shape_watcher'].get('batches_ingested', 0)} batches, "
                      f"{results['shape_watcher'].get('drift_events_emitted', 0)} drifts, "
                      f"{results['shape_watcher'].get('cross_lane_pairs_computed', 0)} cross-lane pairs")
            else:
                results["shape_watcher"] = {"verdict": "FAILED", "stderr": proc.stderr[:500]}
                print(f"  WATCH FAILED: {proc.stderr[:200]}")
            results["stages_run"].append("watch")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            results["shape_watcher"] = {"verdict": "FAILED", "error": str(e)}
            print(f"  WATCH FAILED: {e}")
    else:
        print("  WATCH skipped (use --watch or --pipeline ...,watch)")

    results["completed"] = now_z()
    master_receipt = write_receipt("master_pipeline", {
        "schema": "lucidota.malkovich.master_receipt.v1",
        **results,
    })
    print(f"\n  MASTER RECEIPT: {master_receipt}")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="MALKOVICH SIPHON — RFC-2501 5-Headed Hydra Orchestrator"
    )
    ap.add_argument("--pipeline", default="all",
                    help="Comma-separated stages: siphon,route,train,gauntlet,watch,all (default: all)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max rows to siphon (0=all, no limit)")
    ap.add_argument("--source-strategy", nargs="+", default=[],
                    help="Path(s) to strategy_sample.jsonl")
    ap.add_argument("--gauntlet-rounds", type=int, default=99,
                    help="Darwin Hammer tournament rounds (default=99, always daemon mode)")
    ap.add_argument("--watch", action="store_true",
                    help="Enable 3-plane shape watcher (River + Bytewax observation after siphon)")
    ap.add_argument("--entropy-hint", type=float, default=0.0)
    ap.add_argument("--min-dims", type=int, default=8)
    ap.add_argument("--max-dims", type=int, default=128)
    args = ap.parse_args()

    if args.pipeline == "all":
        stages = {"siphon", "route", "train", "gauntlet"}
    else:
        stages = set(args.pipeline.split(","))

    if args.watch:
        stages.add("watch")
    else:
        stages = set(args.pipeline.split(","))

    source_paths = load_strategy_sources(args.source_strategy)
    print(f"MALKOVICH SIPHON — Pipeline: {stages}")
    print(f"  Sources: {len(source_paths)} file(s)")
    print(f"  Limit: {args.limit or 'all'}")
    print(f"  Elastic Shape binary: {ELASTIC_BIN} (exists={ELASTIC_BIN.exists()})")
    print(f"  Resonance Evolver: {EVOLVER_BIN} (exists={EVOLVER_BIN.exists()})")
    print()

    if not ELASTIC_BIN.exists():
        print("ERROR: lucidota_elastic_shape binary not found. Build with:")
        print(f"  cd {CRATE_DIR} && cargo build --release")
        sys.exit(1)

    results = run_pipeline(stages, source_paths, args.limit, args.gauntlet_rounds)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Stages run: {results['stages_run']}")
    print(f"  Outputs: {OUTPUT_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
