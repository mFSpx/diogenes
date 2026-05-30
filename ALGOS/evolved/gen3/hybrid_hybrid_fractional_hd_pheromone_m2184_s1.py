# DARWIN HAMMER — match 2184, survivor 1
# gen: 3
# parent_a: hybrid_fractional_hdc_hybrid_endpoint_circ_m119_s0.py (gen2)
# parent_b: pheromone.py (gen0)
# born: 2026-05-29T23:41:17Z

"""Hybrid Fractional HDC + Pheromone Modulated Health Score.

Parents:
- hybrid_fractional_hdc_hybrid_endpoint_circ_m119_s0.py
  provides fractional hypervector binding and endpoint geometry vectors.
- pheromone.py
  provides a surface‑pheromone signalling model (signal_kind, signal_value,
  half‑life) that can be used as a scalar weight.

Mathematical bridge:
Both parents expose a scalar (or vector) that can modulate a similarity
measure.  The fractional binding yields a similarity s = ⟨hv, g⟩ (dot
product of a complex hypervector *hv* with a real geometry vector *g*).
The pheromone model supplies a decayed weight w ∈ ℝ⁺ derived from
signal_value and half_life_seconds:

    w = signal_value * exp(-Δt / half_life_seconds)

where Δt is the elapsed time since signal creation (Δt = 0 for a fresh
signal in this module).  The unified health score is therefore

    H = w * (s / ‖hv‖‖g‖)

i.e. a pheromone‑scaled cosine similarity.  This fuses the algebraic
structure of fractional HDC with the temporal decay dynamics of the
pheromone system.

The module implements three core functions demonstrating the hybrid
operation and a lightweight JSON logger that mimics the original
pheromone.py behaviour without external database dependencies.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Core primitives from Parent A (fractional HDC)
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: int | None = None) -> np.ndarray:
    """Generate a random hypervector.

    Parameters
    ----------
    d : int
        Dimension of the hypervector.
    kind : {"complex", "bipolar", "real"}
        Type of hypervector.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (d,).  dtype=complex128 for ``kind=="complex"``,
        float64 otherwise.
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * math.pi, size=d)
        hv = np.exp(1j * theta).astype(np.complex128)
    elif kind == "bipolar":
        hv = rng.choice([-1.0, 1.0], size=d).astype(np.float64)
    elif kind == "real":
        hv = rng.normal(size=d)
        hv /= np.linalg.norm(hv)  # unit L2 norm
    else:
        raise ValueError(f"Unsupported kind={kind!r}")
    return hv

def fractional_bind(hv_a: np.ndarray, hv_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Fractional binding of two hypervectors.

    The operation is defined as element‑wise multiplication of ``hv_a`` with
    ``hv_b`` raised to the fractional power ``alpha`` (complex exponentiation
    works component‑wise).

    Parameters
    ----------
    hv_a, hv_b : np.ndarray
        Input hypervectors of identical shape.
    alpha : float
        Fractional exponent (0 < alpha ≤ 1).

    Returns
    -------
    np.ndarray
        The bound hypervector.
    """
    if hv_a.shape != hv_b.shape:
        raise ValueError("Hypervectors must have the same shape")
    # Complex exponentiation works for both real and complex inputs.
    bound = hv_a * np.power(hv_b, alpha)
    return bound

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit‑norm version of *v* (L2 norm)."""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Zero‑norm vector cannot be normalized")
    return v / norm

# ---------------------------------------------------------------------------
# Core primitives from Parent B (pheromone signalling)
# ---------------------------------------------------------------------------

@dataclass
class PheromoneSignal:
    surface_key: str
    signal_kind: str
    signal_value: float
    half_life_seconds: float
    created_at: str = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    detail: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.detail is None:
            self.detail = {"script": "ALGOS/hybrid_pheromone_hdc.py"}

def decay_weight(signal: PheromoneSignal, now_iso: str | None = None) -> float:
    """Compute the decayed weight of a pheromone signal.

    w = signal_value * exp(-Δt / half_life)

    Parameters
    ----------
    signal : PheromoneSignal
        The signal metadata.
    now_iso : str | None
        Current time as ISO‑8601 string; if None, uses utcnow().

    Returns
    -------
    float
        Decayed weight (≥0).
    """
    now = datetime.fromisoformat((now_iso or datetime.now(timezone.utc).isoformat()).replace('Z', '+00:00'))
    created = datetime.fromisoformat(signal.created_at.replace('Z', '+00:00'))
    delta_seconds = (now - created).total_seconds()
    # Guard against non‑positive half‑life
    if signal.half_life_seconds <= 0:
        return signal.signal_value
    weight = signal.signal_value * math.exp(-delta_seconds / signal.half_life_seconds)
    return max(weight, 0.0)

def log_pheromone(signal: PheromoneSignal, out_dir: Path | None = None) -> Path:
    """Persist a pheromone signal as a JSON report (mirrors parent B's write())."""
    out_dir = out_dir or Path(__file__).resolve().parents[1] / "05_OUTPUTS" / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = f"pheromone_{signal.signal_kind}_{timestamp}.json"
    report_path = out_dir / filename
    payload = asdict(signal)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'))
    payload["report_path"] = str(report_path.relative_to(Path(__file__).resolve().parents[1]))
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"REPORT_PATH={report_path}")
    return report_path

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def endpoint_geometry_vector(d: int = 10000, seed: int | None = None) -> np.ndarray:
    """Create a geometry vector representing endpoint morphology.

    For demonstration, we use a normalized real Gaussian vector.
    """
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=d)
    return normalize_vector(vec)

def hybrid_health_score(
    hv: np.ndarray,
    geom_vec: np.ndarray,
    signal: PheromoneSignal,
) -> float:
    """Compute the pheromone‑modulated cosine similarity between *hv* and *geom_vec*.

    The steps are:
    1. Normalize both vectors (L2 norm).
    2. Compute cosine similarity = ⟨hv_norm, geom_norm⟩.
    3. Scale by the decayed pheromone weight.

    Returns
    -------
    float
        The unified health score H.
    """
    hv_norm = normalize_vector(hv.astype(np.complex128).view(np.float64))  # treat complex as 2‑real dims
    geom_norm = normalize_vector(geom_vec)
    # Cosine similarity (real part because hv may be complex)
    similarity = float(np.real(np.vdot(hv_norm, geom_norm)))
    weight = decay_weight(signal)
    # Normalized similarity already in [-1, 1]; we keep sign to reflect antagonism.
    health = weight * similarity
    return health

def hybrid_bind_and_score(
    dim: int = 10000,
    alpha: float = 0.5,
    seed: int | None = None,
) -> Tuple[np.ndarray, float]:
    """End‑to‑end demo: generate hypervectors, bind fractionally,
    create a pheromone signal, and return the health score.

    Returns
    -------
    bound_hv : np.ndarray
        Result of fractional binding.
    health   : float
        Pheromone‑scaled similarity.
    """
    # 1. Generate base hypervectors
    hv_a = random_hv(d=dim, kind="complex", seed=seed)
    hv_b = random_hv(d=dim, kind="complex", seed=(seed or 0) + 1)

    # 2. Fractional binding
    bound_hv = fractional_bind(hv_a, hv_b, alpha=alpha)

    # 3. Geometry vector (endpoint morphology)
    geom = endpoint_geometry_vector(d=dim, seed=(seed or 0) + 2)

    # 4. Simulated pheromone signal
    signal = PheromoneSignal(
        surface_key="demo_surface",
        signal_kind="health",
        signal_value=1.0,          # baseline strength
        half_life_seconds=300.0,   # 5 minutes
    )
    # Log the signal (side‑effect only)
    log_pheromone(signal)

    # 5. Compute health score
    health = hybrid_health_score(bound_hv, geom, signal)
    return bound_hv, health

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run a quick sanity check; should execute without external dependencies.
    bound, score = hybrid_bind_and_score(dim=2048, alpha=0.7, seed=42)
    print(f"Bound hypervector norm: {np.linalg.norm(bound):.4f}")
    print(f"Hybrid health score: {score:.6f}")
    sys.exit(0)