# DARWIN HAMMER — match 88, survivor 0
# gen: 4
# parent_a: hybrid_fold_change_detectio_hybrid_hybrid_bandit_m103_s0.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_geomet_m18_s3.py (gen3)
# born: 2026-05-29T23:26:51Z

"""
Hybrid Fold‑Change Stylometric‑Geometric Bandit

Parents
-------
* Parent A: ``fold_change_detection`` + hybrid bandit router.
  Provides a dynamical system that converts a scalar input stream into a
  2‑dimensional state (x, y) via the ``step`` update and a ``response_series``.

* Parent B: Stylometric feature extractor + geometric Voronoi/Clifford toolkit.
  Turns a text into a high‑dimensional point (frequency fingerprint) and
  supplies Euclidean‑based Voronoi assignment and a minimal Clifford‑algebra
  blade representation.

Mathematical Bridge
-------------------
Both parents ultimately operate on vectors in ℝⁿ:

* The fold‑change detector yields a 2‑D vector **s** = (x, y) that captures
  temporal change.
* The stylometric extractor yields an n‑dimensional vector **v** that captures
  linguistic style.

By concatenating these vectors we obtain a unified point  

    **h** = (x, y, v₁, …, vₙ) ∈ ℝⁿ⁺² .

This point can be fed to the geometric subsystem (Voronoi partition, Clifford
blade construction).  The index of the nearest Voronoi seed is then used as a
bias term when updating the bandit policy, thereby letting the temporal signal
influence action selection through a spatial‑geometric channel.

The module below implements this fusion, exposing three core hybrid functions:
``hybrid_state_vector``, ``assign_region`` and ``hybrid_select_action``.
"""

import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Fold‑change detection and bandit policy
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear the internal bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Average reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been selected."""
    return _POLICY.get(action, [0.0, 0.0])[1]

def update_policy(updates: list[tuple[str, float]]) -> None:
    """Incorporate a batch of (action, reward) observations."""
    for action, reward in updates:
        total, n = _POLICY.get(action, [0.0, 0.0])
        _POLICY[action] = [total + reward, n + 1]

def step(u: float, x: float, y: float,
         dt: float = 1.0,
         gain: float = 1.0,
         decay_x: float = 1.0,
         decay_y: float = 1.0,
         eps: float = 1e-12) -> tuple[float, float]:
    """
    Euler integration of the feed‑forward fold‑change dynamics.

    Parameters
    ----------
    u : float
        Current input stimulus.
    x, y : float
        Current state variables.
    dt : float
        Integration timestep (non‑negative).
    gain, decay_x, decay_y : float
        Model parameters.
    eps : float
        Small constant to avoid division by zero.
    """
    if dt < 0:
        raise ValueError('dt must be non‑negative')
    ratio = u / max(abs(x), eps)
    dy = gain * ratio - decay_y * y
    dx = u - decay_x * x
    return x + dt * dx, y + dt * dy

def response_series(inputs: list[float],
                    x0: float = 1.0,
                    y0: float = 0.0,
                    **kw) -> list[tuple[float, float]]:
    """
    Apply ``step`` sequentially to *inputs* and collect the trajectory.
    """
    x, y = x0, y0
    out = []
    for u in inputs:
        x, y = step(u, x, y, **kw)
        out.append((x, y))
    return out

def final_response_state(inputs: list[float],
                         x0: float = 1.0,
                         y0: float = 0.0,
                         **kw) -> tuple[float, float]:
    """
    Return the terminal (x, y) after processing *inputs*.
    """
    if not inputs:
        return x0, y0
    return response_series(inputs, x0, y0, **kw)[-1]

# ----------------------------------------------------------------------
# Parent B – Stylometric fingerprint & geometric utilities
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set("no not never none neither cannot".split()),
}

def _tokenize(text: str) -> list[str]:
    """Very simple whitespace / punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())

def stylometric_vector(text: str) -> np.ndarray:
    """
    Produce a normalized frequency vector over the defined function categories.
    The vector length equals ``len(FUNCTION_CATS)``.
    """
    tokens = _tokenize(text)
    total = len(tokens) or 1
    counts = []
    for cat in FUNCTION_CATS.values():
        cnt = sum(1 for t in tokens if t in cat)
        counts.append(cnt / total)
    return np.array(counts, dtype=float)

# Minimal Clifford‑algebra blade representation --------------------------------
def _canonical_blade(categories: frozenset[str]) -> int:
    """
    Encode a set of categories as a bitmask (each category → one bit).
    """
    mapping = {name: i for i, name in enumerate(FUNCTION_CATS.keys())}
    mask = 0
    for cat in categories:
        idx = mapping.get(cat)
        if idx is not None:
            mask |= 1 << idx
    return mask

def _multiply_blades(b1: int, b2: int) -> int:
    """
    Blade multiplication via XOR (geometric product modulo sign is ignored for brevity).
    """
    return b1 ^ b2

# ----------------------------------------------------------------------
# Hybrid Layer: combine temporal response with stylometric geometry
# ----------------------------------------------------------------------
def hybrid_state_vector(text: str,
                        inputs: list[float],
                        x0: float = 1.0,
                        y0: float = 0.0,
                        **kw) -> np.ndarray:
    """
    Concatenate the final fold‑change state (x, y) with the stylometric
    fingerprint of *text* to obtain a single hybrid point.
    """
    x, y = final_response_state(inputs, x0, y0, **kw)
    v = stylometric_vector(text)
    return np.concatenate((np.array([x, y], dtype=float), v))

def assign_region(point: np.ndarray,
                  seed_points: np.ndarray) -> int:
    """
    Euclidean Voronoi assignment: return the index of the nearest seed.
    """
    if seed_points.ndim != 2:
        raise ValueError("seed_points must be a 2‑D array (num_seeds, dim)")
    dists = np.linalg.norm(seed_points - point, axis=1)
    return int(np.argmin(dists))

def hybrid_select_action(actions: list[str],
                         inputs: list[float],
                         text: str,
                         seed_points: np.ndarray,
                         x0: float = 1.0,
                         y0: float = 0.0,
                         **kw) -> str:
    """
    Choose an action by blending bandit expectations with a bias derived
    from the Voronoi region of the hybrid state vector.

    Parameters
    ----------
    actions : list[str]
        Candidate actions.
    inputs : list[float]
        Temporal stimulus stream for the fold‑change detector.
    text : str
        Text whose stylometric fingerprint participates in the hybrid point.
    seed_points : np.ndarray
        Pre‑computed Voronoi seeds (shape = (num_seeds, dim)).
    """
    # 1. Build the hybrid point and locate its region.
    h_point = hybrid_state_vector(text, inputs, x0, y0, **kw)
    region_idx = assign_region(h_point, seed_points)

    # 2. Compute a small bias proportional to the region index.
    region_bias = region_idx * 1e-3

    # 3. Expected rewards from the bandit policy.
    exp_rewards = np.array([_reward(a) for a in actions], dtype=float)

    # 4. Combine and pick the best.
    scores = exp_rewards + region_bias
    best_idx = int(np.argmax(scores))
    chosen = actions[best_idx]

    # 5. (Optional) Update policy with a synthetic reward that reflects the bias.
    #    In a real system the reward would come from the environment.
    update_policy([(chosen, region_bias)])

    return chosen

# ----------------------------------------------------------------------
# Demonstration / smoke test
# ----------------------------------------------------------------------
def _demo() -> None:
    """Run a quick sanity‑check of the hybrid pipeline."""
    # 1. Define a tiny seed set for Voronoi (2‑D + 6‑dim stylometry = 8‑D)
    seed_texts = [
        "the quick brown fox jumps over the lazy dog",
        "i love programming in python and data science",
        "quantum mechanics and relativity are fascinating",
    ]
    seed_inputs = [
        [0.1, 0.2, 0.3],
        [1.0, 0.5, 0.2],
        [0.0, 0.0, 0.0],
    ]
    seeds = np.stack([
        hybrid_state_vector(t, inp) for t, inp in zip(seed_texts, seed_inputs)
    ])

    actions = ["explore", "exploit", "wait"]
    reset_policy()

    # Simulate a few decision steps.
    for step_id in range(5):
        txt = f"Sample text number {step_id} with some random words."
        inp = [random.random() for _ in range(3)]
        chosen = hybrid_select_action(actions, inp, txt, seeds)
        print(f"Step {step_id}: chosen = {chosen}")

if __name__ == "__main__":
    _demo()