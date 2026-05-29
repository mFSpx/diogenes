#!/usr/bin/env python3
"""polycareer_treelite_gate.py — reusable treelite routing gate module.

5-feature vector contract (DB-canonical, gate version v001, schema route_lane):
  [token_count_norm, mutation_requested, needs_cloud_model, needs_graph_write, risk_of_slop]

Lane enum: fast | slow | audit | external | dead_letter

Graceful degradation: if treelite is not installed, defaults to lane="slow", score=0.5.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "03_VAULT/router/treelite_router_v0.tl"

BINARY_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".bin", ".exe", ".so", ".dylib", ".dll",
    ".pkl", ".pickle", ".npy", ".npz",
}

LANES = {"fast", "slow", "audit", "external", "dead_letter"}

_TREELITE_AVAILABLE: bool | None = None


def _check_treelite() -> bool:
    global _TREELITE_AVAILABLE
    if _TREELITE_AVAILABLE is None:
        try:
            import treelite.gtil  # noqa: F401
            _TREELITE_AVAILABLE = True
        except ImportError:
            _TREELITE_AVAILABLE = False
    return _TREELITE_AVAILABLE


def compute_features(artifact_path: str | None, payload: dict) -> np.ndarray:
    """Compute 5-feature vector for treelite routing gate.

    Features (float32, in order):
      0. token_count_norm: file size / 100000, clamped 0..1
      1. mutation_requested: 1.0 if payload.get('mutation') else 0.0
      2. needs_cloud_model: 1.0 if file is image/PDF/binary, 0.0 if plain text
      3. needs_graph_write: 0.0 (workers never write graph directly)
      4. risk_of_slop: 1.0 if source_type=='web_scrape', 0.5 if 'email', 0.0 otherwise
    """
    # feature 0: token_count_norm
    if artifact_path is not None:
        p = Path(artifact_path)
        try:
            size = p.stat().st_size
        except (OSError, FileNotFoundError):
            size = 0
    else:
        size = 0
    token_count_norm = float(min(size / 100000.0, 1.0))

    # feature 1: mutation_requested
    mutation_requested = 1.0 if payload.get("mutation") else 0.0

    # feature 2: needs_cloud_model
    if artifact_path is not None:
        ext = Path(artifact_path).suffix.lower()
        needs_cloud_model = 1.0 if ext in BINARY_EXTENSIONS else 0.0
    else:
        needs_cloud_model = 0.0

    # feature 3: needs_graph_write (workers never write graph directly)
    needs_graph_write = 0.0

    # feature 4: risk_of_slop
    source_type = payload.get("source_type", "")
    if source_type == "web_scrape":
        risk_of_slop = 1.0
    elif source_type == "email":
        risk_of_slop = 0.5
    else:
        risk_of_slop = 0.0

    return np.array(
        [[token_count_norm, mutation_requested, needs_cloud_model, needs_graph_write, risk_of_slop]],
        dtype=np.float32,
    )


def _score_from_model(features: np.ndarray, model_path: str) -> float:
    """Load compiled treelite model and return prediction score. Raises on failure."""
    import treelite
    import treelite.gtil as gtil

    resolved = Path(model_path) if Path(model_path).is_absolute() else ROOT / model_path
    model = treelite.Model.deserialize(str(resolved))
    score = float(gtil.predict(model, features).reshape(-1)[0])
    return score


def _score_fallback(features: np.ndarray) -> float:
    """Pure-numpy fallback replicating the single-split tree:
    split on feature[0] (token_count_norm) at 0.5 → left=0.25, right=0.90.
    """
    return float(0.90 if features[0, 0] >= 0.5 else 0.25)


def _lane_from_score(score: float, features: np.ndarray) -> str:
    """Map score + features to lane enum."""
    mutation_requested = float(features[0, 1])
    needs_graph_write = float(features[0, 3])
    risk_of_slop = float(features[0, 4])

    # dead_letter: very low confidence and mutation risk
    if score < 0.20:
        return "dead_letter"

    # audit: graph write involved or high slop + mutation
    if needs_graph_write > 0.5 or (risk_of_slop >= 1.0 and mutation_requested > 0.5):
        return "audit"

    # external: cloud model required (needs_cloud_model feature[2] >= 1.0)
    if float(features[0, 2]) >= 1.0:
        return "external"

    # fast vs slow threshold
    if score >= 0.5:
        return "fast"

    return "slow"


def route(
    artifact_path: str | None,
    payload: dict,
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """Route an artifact+payload through the treelite gate.

    Returns {"lane": str, "score": float, "features": list[float]}.
    """
    features = compute_features(artifact_path, payload)

    if _check_treelite():
        try:
            score = _score_from_model(features, model_path)
        except Exception:
            score = _score_fallback(features)
    else:
        score = _score_fallback(features)

    lane = _lane_from_score(score, features)

    return {
        "lane": lane,
        "score": round(score, 6),
        "features": features[0].tolist(),
    }


def lane_to_strategy(lane: str) -> dict:
    """Map lane enum to execution strategy flags.

    Returns {"use_groq": bool, "use_gliner": bool, "regex_only": bool, "audit_required": bool}.
    """
    strategies: dict[str, dict[str, Any]] = {
        "fast": {
            "use_groq": False,
            "use_gliner": False,
            "regex_only": True,
            "audit_required": False,
        },
        "slow": {
            "use_groq": False,
            "use_gliner": True,
            "regex_only": False,
            "audit_required": False,
        },
        "audit": {
            "use_groq": False,
            "use_gliner": True,
            "regex_only": False,
            "audit_required": True,
        },
        "external": {
            "use_groq": True,
            "use_gliner": False,
            "regex_only": False,
            "audit_required": False,
        },
        "dead_letter": {
            "use_groq": False,
            "use_gliner": False,
            "regex_only": False,
            "audit_required": True,
        },
    }
    if lane not in strategies:
        raise ValueError(f"Unknown lane: {lane!r}. Must be one of {sorted(LANES)}")
    return strategies[lane]


def main() -> int:
    ap = argparse.ArgumentParser(prog="polycareer-treelite-gate")
    ap.add_argument("--artifact", default=None, help="Path to artifact file (optional)")
    ap.add_argument("--payload", default="{}", help="JSON payload dict")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to compiled treelite model (default: {DEFAULT_MODEL_PATH})",
    )
    args = ap.parse_args()

    try:
        payload = json.loads(args.payload)
    except json.JSONDecodeError as exc:
        print(json.dumps({"ok": False, "error": f"Invalid payload JSON: {exc}"}))
        return 1

    result = route(args.artifact, payload, model_path=args.model)
    strategy = lane_to_strategy(result["lane"])

    output = {
        "lane": result["lane"],
        "score": result["score"],
        "features": result["features"],
        "strategy": strategy,
        "treelite_available": _check_treelite(),
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
