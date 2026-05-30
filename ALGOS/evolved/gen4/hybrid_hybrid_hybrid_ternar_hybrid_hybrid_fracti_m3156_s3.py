# DARWIN HAMMER — match 3156, survivor 3
# gen: 4
# parent_a: hybrid_hybrid_ternary_lens__hybrid_path_signatur_m34_s0.py (gen2)
# parent_b: hybrid_hybrid_fractional_hd_pheromone_m2184_s2.py (gen3)
# born: 2026-05-29T23:48:11Z

import numpy as np
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CLASSIFICATIONS = {
    "usable_now": 1.0,
    "research_only": 0.8,
    "needs_conversion": 0.5,
    "unsafe_for_fastpath": 0.2,
    "unsupported": 0.1,
}
LOCAL_PATTERNS = [
    "*bitnet*",
    "*BitNet*",
    "*fairyfuse*",
    "*FairyFuse*",
    "*lora*",
    "*LoRA*",
    "*adapter*",
]

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format (no sub‑second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_manifest(path: Path) -> Dict[str, Any]:
    """Load a JSON manifest and validate classifications."""
    data = json.loads(path.read_text(encoding="utf-8"))
    for cand in data.get("vendors", []):
        cls = cand.get("classification")
        if cls not in CLASSIFICATIONS:
            raise SystemExit(f"invalid classification {cls!r} for {cand.get('candidate_key')}")
    return data


def enforce_fast_path_rule(candidate: Dict[str, Any]) -> List[str]:
    """Check fast‑path constraints; return human‑readable findings."""
    findings: List[str] = []
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    if re.search(r"standard.*lora|peft|qlora", f"{key} {family}", re.I):
        if candidate.get("classification") != "unsafe_for_fastpath":
            findings.append("Fast path rule enforced")
    return findings


def _seed_from_string(s: str) -> int:
    """Derive a deterministic 64‑bit seed from an arbitrary string."""
    # Simple but reproducible hash – not cryptographic
    return abs(hash(s)) % (2**63)


def random_hv(
    d: int = 10_000,
    kind: str = "complex",
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a random hypervector.

    Parameters
    ----------
    d: dimensionality.
    kind: "complex", "bipolar", or "real".
    seed: optional deterministic seed.
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


def fractional_bind(
    hv_a: np.ndarray,
    hv_b: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Perform fractional binding of two hypervectors.

    The operation is element‑wise multiplication of ``hv_a`` with ``hv_b`` raised
    to the power ``alpha``.  For complex vectors this respects the polar form.
    """
    if hv_a.shape != hv_b.shape:
        raise ValueError("Hypervectors must have the same shape")
    # Preserve dtype – np.power works for real, complex and bipolar arrays
    bound = hv_a * np.power(hv_b, alpha)
    return bound


def exponential_decay(
    initial: float,
    elapsed_seconds: float,
    half_life_seconds: float,
) -> float:
    """Standard exponential decay based on half‑life."""
    if half_life_seconds <= 0:
        raise ValueError("half_life_seconds must be positive")
    decay_factor = 2 ** (-elapsed_seconds / half_life_seconds)
    return initial * decay_factor


def classification_weight(classification: str) -> float:
    """Map a classification string to its base weight."""
    try:
        return CLASSIFICATIONS[classification]
    except KeyError as exc:
        raise ValueError(f"Unsupported classification {classification!r}") from exc


def pheromone_signal(
    classification: str,
    base_signal: float,
    half_life_seconds: float,
    timestamp: datetime,
) -> float:
    """
    Compute the pheromone signal value after decay, scaled by classification weight.

    The decay is computed from the supplied ``timestamp`` to ``utc_now()``.
    """
    now = datetime.now(timezone.utc)
    elapsed = (now - timestamp).total_seconds()
    decayed = exponential_decay(base_signal, elapsed, half_life_seconds)
    return decayed * classification_weight(classification)


def candidate_lens_hv(candidate: Dict[str, Any], d: int = 10_000) -> np.ndarray:
    """
    Produce a deterministic hypervector representing the lens candidate.

    The seed is derived from a stable combination of candidate fields.
    """
    key = candidate.get("candidate_key", "")
    family = candidate.get("family", "")
    notes = candidate.get("notes", "")
    seed_str = f"{key}|{family}|{notes}"
    seed = _seed_from_string(seed_str)
    return random_hv(d=d, kind="complex", seed=seed)


def hybrid_operation(candidate: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    """
    Execute the deep hybrid algorithm.

    Returns
    -------
    bound_hv : np.ndarray
        The fractional binding of the lens hypervector with a pheromone hypervector.
    signal   : float
        The final pheromone signal after classification‑aware decay.
    """
    classification = candidate.get("classification")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"Invalid classification {classification!r}")

    # 1. Deterministic lens hypervector
    lens_hv = candidate_lens_hv(candidate)

    # 2. Stochastic pheromone hypervector (independent seed)
    pheromone_hv = random_hv(d=lens_hv.shape[0], kind="complex")

    # 3. Classification‑driven binding exponent
    #    Higher trust → stronger binding (alpha close to 1)
    base_alpha = 0.5
    alpha = base_alpha + 0.4 * classification_weight(classification)  # range ≈[0.5,0.9]

    # 4. Fractional binding
    bound_hv = fractional_bind(lens_hv, pheromone_hv, alpha=alpha)

    # 5. Pheromone signal with temporal decay
    signal = pheromone_signal(
        classification=classification,
        base_signal=1.0,
        half_life_seconds=3600.0,
        timestamp=datetime.now(timezone.utc),  # start time = now for demo purposes
    )

    return bound_hv, signal


# ----------------------------------------------------------------------
# Demo / CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo_candidate = {
        "classification": "usable_now",
        "candidate_key": "example_lens",
        "family": "example_family",
        "notes": "demo entry",
    }
    hv, sig = hybrid_operation(demo_candidate)
    print("Bound hypervector (first 5 components):", hv[:5])
    print("Pheromone signal:", sig)