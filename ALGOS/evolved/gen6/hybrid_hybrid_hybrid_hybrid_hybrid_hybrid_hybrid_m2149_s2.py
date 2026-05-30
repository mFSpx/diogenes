# DARWIN HAMMER — match 2149, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_hybrid_m976_s5.py (gen5)
# parent_b: hybrid_hybrid_hybrid_model__hybrid_hard_truth_ma_m23_s1.py (gen3)
# born: 2026-05-29T23:41:06Z

import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
_EPS = 1e-12                     # numerical stability guard
_LEARNING_RATE = 0.05            # base step size for weight updates
_ITERATIONS = 20                 # number of integration steps
_DIM = 10                        # dimensionality of stylometry / weight space

# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    """Physical description of an object used for curvature‑based modulation."""
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class VramSlotPlan:
    artifact_id: str
    artifact_kind: str
    action: str
    estimated_mb: int
    reason: str
    detail: Dict[str, Any]

# ----------------------------------------------------------------------
# Mathematical primitives
# ----------------------------------------------------------------------
def fisher_score(m: Morphology) -> float:
    """
    A smooth, bounded Fisher‑like score.
    The denominator contains a small epsilon to avoid division by zero
    and to keep the score well‑scaled for extreme morphologies.
    """
    numerator = m.length * m.width * m.height * m.mass
    denominator = 1.0 + m.length + m.width + m.height + m.mass + _EPS
    return numerator / denominator


def ollivier_ricci_curvature(v_src: np.ndarray, v_tgt: np.ndarray) -> float:
    """
    Compute a cosine‑style Ollivier–Ricci curvature.
    Returns a value in [-1, 1] with safe handling of zero‑norm vectors.
    """
    norm_src = np.linalg.norm(v_src) + _EPS
    norm_tgt = np.linalg.norm(v_tgt) + _EPS
    return float(np.dot(v_src, v_tgt) / (norm_src * norm_tgt))


def hybrid_transport(
    m: Morphology,
    v_src: np.ndarray,
    v_tgt: np.ndarray,
) -> np.ndarray:
    """
    Produce a transport vector that blends geometric curvature with the
    Fisher score of the morphology.  The result is a direction that can be
    added to a stylometric embedding before it influences the weight matrix.
    """
    w_f = fisher_score(m)
    kappa = ollivier_ricci_curvature(v_src, v_tgt)
    # (1 + kappa) stays in [0, 2]; scaling by w_f keeps magnitude sensible.
    return (1.0 + kappa) * w_f * (v_tgt - v_src)


# ----------------------------------------------------------------------
# Feature extraction – richer than the original two‑feature version
# ----------------------------------------------------------------------
def stylometry_features(text: str) -> np.ndarray:
    """
    Extract a 10‑dimensional stylometric embedding:
        0 – first‑person singular pronouns count
        1 – second‑person pronouns count
        2 – average word length
        3 – total token count
        4 – number of punctuation marks
        5 – ratio of uppercase letters
        6 – lexical diversity (unique / total)
        7 – count of digits
        8 – presence of exclamation marks (binary)
        9 – average sentence length (in words)
    All features are normalised to [0, 1] where appropriate.
    """
    tokens = text.split()
    n_tokens = max(len(tokens), 1)

    # pronoun counts
    first_person = sum(tok.lower() in {"i", "me", "my", "mine", "myself"} for tok in tokens)
    second_person = sum(tok.lower() in {"you", "your", "yours", "yourself"} for tok in tokens)

    # basic lexical stats
    word_lengths = [len(tok.strip(".,!?;:")) for tok in tokens if tok.strip(".,!?;:")]
    avg_word_len = sum(word_lengths) / max(len(word_lengths), 1)

    # punctuation
    punct = sum(1 for ch in text if ch in ".,!?;:")

    # uppercase ratio
    uppercase = sum(1 for ch in text if ch.isupper())
    uppercase_ratio = uppercase / max(len(text), 1)

    # lexical diversity
    unique = len(set(tok.lower() for tok in tokens))
    lexical_div = unique / n_tokens

    # digits
    digits = sum(1 for ch in text if ch.isdigit())

    # exclamation presence
    exclamation = 1.0 if "!" in text else 0.0

    # sentence length (naïve split on .!?)
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)

    feats = np.array([
        first_person,
        second_person,
        avg_word_len,
        n_tokens,
        punct,
        uppercase_ratio,
        lexical_div,
        digits,
        exclamation,
        avg_sent_len,
    ], dtype=float)

    # Normalise the count‑based features to keep magnitudes comparable.
    # The normalisation constants are simple heuristics; they can be tuned.
    feats[0:2] = feats[0:2] / max(n_tokens, 1)          # pronoun ratios
    feats[2] = min(feats[2] / 20.0, 1.0)                # avg word length capped at 20
    feats[3] = min(feats[3] / 200.0, 1.0)               # token count capped at 200
    feats[4] = min(feats[4] / 50.0, 1.0)                # punctuation capped at 50
    feats[7] = min(feats[7] / 20.0, 1.0)                # digits capped at 20
    feats[9] = min(feats[9] / 40.0, 1.0)                # avg sentence length capped at 40

    return feats


# ----------------------------------------------------------------------
# Weight‑matrix dynamics – a deeper integration of geometry and statistics
# ----------------------------------------------------------------------
def _euler_step(
    W: np.ndarray,
    v: np.ndarray,
    delta: np.ndarray,
    lr: float,
) -> np.ndarray:
    """
    One explicit Euler integration step for the differential equation:
        dW/dt = -lr * (W - outer(v + delta, v + delta)) * fisher_score
    The outer product encodes the current stylometric direction perturbed by
    the transport vector `delta`.  The Fisher score acts as a global modulation.
    """
    embed = v + delta
    outer = np.outer(embed, embed)                     # rank‑1 contribution
    grad = (W - outer) * fisher_score(current_morph)   # shape (D, D)
    return W - lr * grad


def update_weight_matrix_deep(
    W: np.ndarray,
    v: np.ndarray,
    delta: np.ndarray,
    lr: float,
) -> np.ndarray:
    """
    Wrapper that performs a single stable update using the Euler step.
    """
    return _euler_step(W, v, delta, lr)


def hybrid_algorithm(
    m: Morphology,
    text: str,
    *,
    iterations: int = _ITERATIONS,
    base_lr: float = _LEARNING_RATE,
) -> np.ndarray:
    """
    End‑to‑end hybrid procedure:
        1. Extract stylometric embedding `v`.
        2. Initialise a random symmetric weight matrix `W`.
        3. For each iteration:
            a) Compute a transport vector `delta` using the current `v`
               and a reference vector (here we reuse `v` shifted by a small
               random perturbation to emulate a target state).
            b) Update `W` with a geometry‑aware Euler step.
            c) Optionally decay the learning rate.
        4. Return the final weight matrix.
    """
    global current_morph
    current_morph = m                     # needed inside the Euler step

    v = stylometry_features(text)        # shape (D,)
    # Ensure symmetry of the initial matrix for interpretability
    W = np.random.rand(_DIM, _DIM)
    W = (W + W.T) / 2.0

    lr = base_lr
    for step in range(iterations):
        # Create a pseudo‑target vector by adding a tiny random jitter.
        jitter = np.random.normal(scale=0.01, size=v.shape)
        v_target = v + jitter

        delta = hybrid_transport(m, v, v_target)   # transport direction
        W = update_weight_matrix_deep(W, v, delta, lr)

        # Simple learning‑rate decay to improve convergence
        lr *= 0.98

    # Enforce symmetry again to counter numerical drift
    W = (W + W.T) / 2.0
    return W


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_morph = Morphology(length=1.2, width=0.8, height=2.5, mass=3.4)
    example_text = (
        "I think you should consider the implications of this test. "
        "Your feedback is valuable! Let's improve together."
    )
    final_W = hybrid_algorithm(example_morph, example_text)
    np.set_printoptions(precision=4, suppress=True)
    print("Final weight matrix (symmetric):")
    print(final_W)