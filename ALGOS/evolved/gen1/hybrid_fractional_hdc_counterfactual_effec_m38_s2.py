# DARWIN HAMMER — match 38, survivor 2
# gen: 1
# parent_a: fractional_hdc.py (gen0)
# parent_b: counterfactual_effects.py (gen0)
# born: 2026-05-29T23:23:24Z

"""Hybrid hyperdimensional-causal module.

This module fuses the fractional binding algebra from *fractional_hdc.py* with the
scalar causal effect estimates from *counterfactual_effects.py*.  The bridge is
the fractional exponent `alpha` used in `fractional_power`.  A causal average
treatment effect (ATE) is first normalised to a bounded range and then used as
the exponent that scales the phase of the outcome hypervector before it is
bound to the treatment hypervector.  Confounders are bundled and finally bound
to the treatment‑outcome pair, yielding a single hypervector that simultaneously
encodes the symbolic identifiers and the magnitude of the causal effect.

The resulting hypervector can be queried (unbound) to recover approximate
identities of treatment, outcome and a numeric estimate of the ATE, demonstrating
a mathematically unified representation of both parent algorithms.
"""

from __future__ import annotations

import math
import random
import sys
import uuid
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Re‑implementation of core primitives from fractional_hdc.py
# ---------------------------------------------------------------------------

def random_hv(d: int = 10000, kind: str = "complex", seed: Optional[int] = None) -> np.ndarray:
    """Generate a random hypervector.

    Parameters
    ----------
    d: dimension
    kind: "complex", "bipolar", or "real"
    seed: optional RNG seed
    """
    rng = np.random.default_rng(seed)
    if kind == "complex":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=d)
        return np.exp(1j * theta)
    if kind == "bipolar":
        return rng.choice(np.array([-1.0, 1.0]), size=d)
    if kind == "real":
        v = rng.standard_normal(d)
        return v / (np.linalg.norm(v) + 1e-30)
    raise ValueError(f"Unsupported kind {kind!r}")


def bind(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Circular convolution binding."""
    return np.fft.ifft(np.fft.fft(X) * np.fft.fft(Y))


def unbind(Z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Inverse of bind using division in the Fourier domain."""
    FY = np.fft.fft(Y)
    mag = np.abs(FY)
    inv_FY = np.conj(FY) / (mag ** 2 + 1e-30)
    return np.fft.ifft(np.fft.fft(Z) * inv_FY)


def fractional_power(Y: np.ndarray, alpha: float) -> np.ndarray:
    """Raise hypervector Y to a fractional power by scaling its Fourier phase."""
    F = np.fft.fft(Y.astype(complex))
    magnitude = np.abs(F)
    phase = np.angle(F)
    F_frac = magnitude * np.exp(1j * alpha * phase)
    return np.fft.ifft(F_frac)


def bundle(hvs: Iterable[np.ndarray],
           weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Superpose hypervectors and normalise to unit L2 norm."""
    arr = np.stack(list(hvs), axis=0)
    if weights is not None:
        arr = arr * weights[:, None]
    summed = arr.sum(axis=0)
    norm = np.linalg.norm(summed) + 1e-30
    return summed / norm


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (real part for complex vectors)."""
    a = a.ravel()
    b = b.ravel()
    dot = np.real(np.vdot(a, b))
    return dot / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)


# ---------------------------------------------------------------------------
# Re‑implementation of core primitives from counterfactual_effects.py
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CausalEffect:
    effect_id: str
    treatment: str
    outcome: str
    confounders: Tuple[str, ...]
    ate_estimate: Optional[float]
    ate_confidence_interval: Optional[Tuple[float, float]]
    refutation_passed: bool
    refutation_methods: Tuple[str, ...]
    heterogeneous_effects: Dict[str, float]


def _extract_arrays(data: dict, key: str) -> List[float]:
    """Helper to pull a list of floats from the data dict."""
    raw = data.get(key, [])
    return [float(v) for v in raw]


def estimate_causal_effect(treatment: str,
                           outcome: str,
                           confounders: List[str],
                           data: dict) -> CausalEffect:
    """Very lightweight ATE estimator (identical to parent implementation)."""
    t = _extract_arrays(data, treatment)
    y = _extract_arrays(data, outcome)
    if not t or len(t) != len(y):
        ate = None
        ci = None
    else:
        yt = [yy for tt, yy in zip(t, y) if tt >= 0.5]
        yc = [yy for tt, yy in zip(t, y) if tt < 0.5]
        ate = (statistics.mean(yt) - statistics.mean(yc)) if yt and yc else None
        spread = statistics.pstdev(y) if len(y) > 1 else 0.0
        ci = None if ate is None else (ate - spread, ate + spread)
    return CausalEffect(
        effect_id=str(uuid.uuid4()),
        treatment=treatment,
        outcome=outcome,
        confounders=tuple(confounders),
        ate_estimate=ate,
        ate_confidence_interval=ci,
        refutation_passed=ate is not None,
        refutation_methods=("placebo_treatment", "data_subset", "random_common_cause"),
        heterogeneous_effects={},
    )


# ---------------------------------------------------------------------------
# Hybrid layer: mapping causal objects ↔ hypervectors
# ---------------------------------------------------------------------------

# Global cache for deterministic symbol → hypervector mapping
_SYMBOL_CACHE: Dict[str, np.ndarray] = {}
_SYMBOL_DIM = 4096  # modest dimension for demo (still holographic)


def _symbol_hv(symbol: str) -> np.ndarray:
    """Return a deterministic hypervector for a given symbol string."""
    if symbol not in _SYMBOL_CACHE:
        # deterministic seed from the symbol's hash
        seed = int.from_bytes(symbol.encode("utf-8"), "little", signed=False) % (2 ** 32)
        _SYMBOL_CACHE[symbol] = random_hv(d=_SYMBOL_DIM, kind="complex", seed=seed)
    return _SYMBOL_CACHE[symbol]


def _normalize_ate(ate: Optional[float],
                   scale: float = 1.0) -> float:
    """Map raw ATE to a bounded exponent in [-1, 1].

    The tanh function provides smooth saturation; the optional `scale` can be
    tuned to the expected magnitude of ATEs.
    """
    if ate is None or math.isnan(ate):
        return 0.0
    return math.tanh(ate / scale)


def encode_causal_effect(effect: CausalEffect,
                         ate_scale: float = 1.0) -> np.ndarray:
    """Encode a CausalEffect into a single hypervector.

    The encoding follows:

        T = hv(treatment)
        O = hv(outcome) ^ alpha        (fractional power)
        C = bundle( hv(conf_i) )       (confounder bundle)

        Z = bind( bind(T, O), C )

    where `alpha = tanh(ATE/scale)` and the bundle of confounders is optional.
    """
    T = _symbol_hv(effect.treatment)
    O = _symbol_hv(effect.outcome)

    alpha = _normalize_ate(effect.ate_estimate, scale=ate_scale)
    O_frac = fractional_power(O, alpha)

    if effect.confounders:
        C_vecs = [_symbol_hv(c) for c in effect.confounders]
        C = bundle(C_vecs)
    else:
        # identity hypervector for "no confounders"
        C = np.ones(_SYMBOL_DIM, dtype=complex) / np.sqrt(_SYMBOL_DIM)

    # Two successive bindings: (T * O_frac) then bind with confounder bundle
    TO = bind(T, O_frac)
    Z = bind(TO, C)
    # Normalise to unit magnitude for stability
    return Z / (np.linalg.norm(Z) + 1e-30)


def decode_causal_effect(encoded: np.ndarray,
                         candidate_treatments: List[str],
                         candidate_outcomes: List[str],
                         candidate_confounders: List[str]) -> dict:
    """Approximate reverse mapping from a hypervector to causal components.

    The function searches the provided candidate lists for the symbols whose
    hypervectors maximise similarity after appropriate unbinding steps.

    Returns a dictionary with keys:
        'treatment', 'outcome', 'alpha_estimate', 'confounders' (list)
    """
    # First, try to recover the confounder bundle by unbinding with each
    # candidate treatment/outcome pair and measuring residual similarity.
    best = {"score": -1.0}
    for t_sym in candidate_treatments:
        T = _symbol_hv(t_sym)
        # Unbind treatment
        Z1 = unbind(encoded, T)
        for o_sym in candidate_outcomes:
            O = _symbol_hv(o_sym)
            # Since O was raised to alpha, we cannot directly unbind.
            # Instead we search for the alpha that maximises similarity.
            # We sample a small set of alphas.
            for alpha in np.linspace(-1.0, 1.0, 21):
                O_alpha = fractional_power(O, alpha)
                Z2 = unbind(Z1, O_alpha)
                # The remaining vector should be close to the confounder bundle.
                # Compare against a bundle of all candidate confounders.
                conf_vecs = [_symbol_hv(c) for c in candidate_confounders]
                C_bundle = bundle(conf_vecs)
                sim = similarity(Z2, C_bundle)
                if sim > best["score"]:
                    best.update({
                        "score": sim,
                        "treatment": t_sym,
                        "outcome": o_sym,
                        "alpha_estimate": alpha,
                        "confounders": candidate_confounders,
                    })
    # Convert alpha back to an approximate ATE using the inverse tanh scaling.
    alpha = best.get("alpha_estimate", 0.0)
    ate_est = math.atanh(alpha)  # inverse of tanh, yields raw ATE estimate (scaled)
    best["ate_estimate"] = ate_est
    return best


def hybrid_effect_pipeline(treatment: str,
                           outcome: str,
                           confounders: List[str],
                           data: dict,
                           ate_scale: float = 1.0) -> Tuple[CausalEffect, np.ndarray]:
    """Full pipeline: estimate causal effect, encode to hypervector, and return both."""
    effect = estimate_causal_effect(treatment, outcome, confounders, data)
    hv = encode_causal_effect(effect, ate_scale=ate_scale)
    return effect, hv


# ---------------------------------------------------------------------------
# Demonstration functions (three required)
# ---------------------------------------------------------------------------

def demo_encoding_decoding():
    """Show encoding of a synthetic causal effect and its approximate decoding."""
    # Synthetic data: binary treatment, continuous outcome
    data = {
        "drug": [1, 0, 1, 0, 1, 0],
        "recovery": [0.9, 0.4, 0.85, 0.3, 0.95, 0.2],
    }
    treatment = "drug"
    outcome = "recovery"
    confs = ["age", "sex"]
    effect, hv = hybrid_effect_pipeline(treatment, outcome, confs, data, ate_scale=0.5)

    # Candidate pools (in a real system they would be larger)
    cand_treat = ["drug", "placebo"]
    cand_outcome = ["recovery", "mortality"]
    cand_confs = ["age", "sex", "income"]

    decoded = decode_causal_effect(hv, cand_treat, cand_outcome, cand_confs)

    print("Original effect:")
    print(effect)
    print("\nDecoded approximation:")
    print(decoded)


def demo_fractional_binding_vs_ate():
    """Illustrate how the ATE magnitude controls the fractional exponent."""
    # Create two simple hypervectors
    a = _symbol_hv("treatment_A")
    b = _symbol_hv("outcome_B")

    # Simulated ATE values
    for ate in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        alpha = _normalize_ate(ate, scale=1.0)
        bound = bind(a, fractional_power(b, alpha))
        sim = similarity(bound, a)  # should be low, showing binding
        print(f"ATE={ate: .2f}, alpha={alpha: .3f}, similarity to treatment={sim:.3f}")


def demo_refutation_hypervector():
    """Encode the refutation methods of a CausalEffect as a hypervector."""
    effect = estimate_causal_effect(
        treatment="intervention",
        outcome="result",
        confounders=[],
        data={"intervention": [1, 0, 1], "result": [0.7, 0.4, 0.8]},
    )
    # Encode each refutation method and bind them together
    method_hvs = [_symbol_hv(m) for m in effect.refutation_methods]
    methods_bundle = bundle(method_hvs)
    # Bind with the effect's treatment hypervector for a composite representation
    composite = bind(_symbol_hv(effect.treatment), methods_bundle)
    print(f"Composite refutation hypervector norm: {np.linalg.norm(composite):.3f}")
    # Query similarity to each individual method
    for m in effect.refutation_methods:
        sim = similarity(composite, _symbol_hv(m))
        print(f"Similarity to method '{m}': {sim:.3f}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running hybrid hyperdimensional-causal demos...\n")
    demo_encoding_decoding()
    print("\n---\n")
    demo_fractional_binding_vs_ate()
    print("\n---\n")
    demo_refutation_hypervector()
    print("\nAll demos executed successfully.")