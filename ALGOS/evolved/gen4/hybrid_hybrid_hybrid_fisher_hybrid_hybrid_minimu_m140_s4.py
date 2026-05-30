# DARWIN HAMMER — match 140, survivor 4
# gen: 4
# parent_a: hybrid_hybrid_fisher_locali_hybrid_sketches_rlct_m33_s2.py (gen3)
# parent_b: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s4.py (gen2)
# born: 2026-05-29T23:27:14Z

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – statistical core
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def count_min_sketch(items: Iterable[Any], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Very simple count‑min sketch using SHA‑256 as hash."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def _loglog_regression(losses: np.ndarray, ns: np.ndarray) -> float:
    """Core of the RLCT estimator – ordinary least squares in log‑log space."""
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(np.maximum(ns, math.e)))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def estimate_rlct_from_losses(
    train_losses_per_n: Iterable[float],
    n_values: Iterable[float],
) -> float:
    """Public wrapper that validates inputs and forwards to the regression."""
    losses = np.asarray(list(train_losses_per_n), dtype=np.float64)
    ns = np.asarray(list(n_values), dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    if losses.shape != ns.shape:
        raise ValueError("train_losses_per_n and n_values must have the same length")
    return _loglog_regression(losses, ns)


# ----------------------------------------------------------------------
# Parent B – epistemic certainty core
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """Immutable container for an epistemic certainty statement."""
    label: str
    confidence_bps: int  # basis points, 0..10000  (1 bps = 0.01%)
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", "2024-01-01T00:00:00Z")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    """Factory for :class:`CertaintyFlag`."""
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    """A concrete FACT flag for a file that has been hashed."""
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    """Produces a PROBABLE flag for normal extraction or BULLSHIT when injection is detected."""
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=9000,
            authority_class="prompt_injection_signature",
            rationale="Untrusted source text matched instruction‑injection signatures; preserve bytes but treat embedded directives as hostile data.",
            evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
        )
    return certainty(
        "PROBABLE",
        confidence_bps=8000,
        authority_class="parser_extraction",
        rationale="Extraction succeeded without detected injection.",
        evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
    )


# ----------------------------------------------------------------------
# Fusion layer – three public functions
# ----------------------------------------------------------------------
def weighted_fisher_score(
    data: Iterable[float],
    *,
    center: float,
    width: float,
    certainty_flag: CertaintyFlag,
) -> float:
    """
    Compute the average Fisher information of *data* (interpreted as angles)
    and weight it by the epistemic confidence.

    The confidence is interpreted as a probability in ``[0, 1]``:
    ``p = confidence_bps / 10000``.
    """
    conf = certainty_flag.confidence_bps / 10000.0
    if not (0.0 <= conf <= 1.0):
        raise ValueError("confidence must be in [0,1]")
    scores = [fisher_score(theta, center, width) for theta in data]
    avg_score = float(np.mean(scores)) if scores else 0.0
    return conf * avg_score


def weighted_rlct_estimate(
    cert_flags: Iterable[CertaintyFlag],
    train_losses_per_n: Iterable[float],
    n_values: Iterable[float],
) -> float:
    """
    RLCT regression where each loss contribution is multiplied by the
    corresponding certainty weight.

    The two iterables ``cert_flags``, ``train_losses_per_n`` and ``n_values``
    must have the same length.
    """
    if not (len(cert_flags) == len(train_losses_per_n) == len(n_values)):
        raise ValueError("cert_flags, train_losses_per_n and n_values must have the same length")
    losses = np.asarray(list(train_losses_per_n), dtype=np.float64)
    ns = np.asarray(list(n_values), dtype=np.float64)
    confs = np.asarray([flag.confidence_bps / 10000.0 for flag in cert_flags], dtype=np.float64)
    if np.any(ns <= math.e):
        raise ValueError("all n_values must be > e for log(log(n)) to be positive")
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(np.maximum(ns, math.e)))
    x_c = x - x.mean()
    y_c = (y * confs) - (y * confs).mean()
    var_x = (x_c ** 2).sum()
    if var_x < 1e-15:
        raise ValueError("n_values have no variance in log(log(n)) space")
    return float((x_c * y_c).sum() / var_x)


def hybrid_metric(
    data: Iterable[float],
    cert_flags: Iterable[CertaintyFlag],
    train_losses_per_n: Iterable[float],
    n_values: Iterable[float],
    *,
    center: float,
    width: float,
) -> Tuple[float, float, float]:
    """
    A consolidated metric that combines Fisher scoring, epistemic certainty
    and RLCT regression.

    Returns a tuple of:
    - weighted Fisher score
    - weighted RLCT estimate
    - count-min sketch
    """
    weighted_fisher = weighted_fisher_score(data, center=center, width=width, certainty_flag=list(cert_flags)[0])
    weighted_rlct = weighted_rlct_estimate(cert_flags, train_losses_per_n, n_values)
    sketch = count_min_sketch(data)
    return weighted_fisher, weighted_rlct, sketch