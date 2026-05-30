# DARWIN HAMMER — match 818, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_bandit_router_m111_s2.py (gen3)
# parent_b: nlms.py (gen0)
# born: 2026-05-29T23:31:09Z

import re
import math
import random
import sys
import pathlib
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regular expressions (kept from original for possible future extensions)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
                         re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
                         re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
                      re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
                        re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
                         re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
                        re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
                          re.I)
SCARCITY_RE = re.compile(r"\b(?:limit|short|shortage|shortages|limited|limited|scarcity)\b",
                         re.I)


# ----------------------------------------------------------------------
# Utility: Count‑Min Sketch (lightweight, deterministic hash functions)
# ----------------------------------------------------------------------
@dataclass
class CountMinSketch:
    width: int = 2000
    depth: int = 5
    seed: int = 0
    table: np.ndarray = field(init=False)
    _hashes: List[np.random.Generator] = field(init=False)

    def __post_init__(self) -> None:
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        self._hashes = [rng.integers(1, 2**31 - 1, size=self.width, dtype=np.int64)
                        for _ in range(self.depth)]

    def _indices(self, key: int) -> List[int]:
        """Deterministic indices for a given integer key."""
        return [(key * h) % self.width for h in self._hashes]

    def update(self, key: int, increment: int = 1) -> None:
        for row, col in enumerate(self._indices(key)):
            self.table[row, col] += increment

    def estimate(self, key: int) -> int:
        return min(self.table[row, col] for row, col in enumerate(self._indices(key)))


# ----------------------------------------------------------------------
# Core mathematical bridge
# ----------------------------------------------------------------------
def shannon_entropy(counts: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute Shannon entropy of a non‑negative count vector.
    The vector is first normalised to a probability distribution.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Guard against log(0) by adding eps inside the log
    return -np.sum(probs * np.log2(probs + eps))


def predict(weights: np.ndarray, x: np.ndarray) -> float:
    """Linear prediction used by NLMS."""
    return float(np.dot(weights, x))


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                target: float,
                mu: float = 0.5,
                eps: float = 1e-9,
                entropy_factor: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Normalized Least Mean Squares update whose step size is modulated
    by the entropy of the feature vector. Higher entropy (more uncertainty)
    reduces the effective learning rate, yielding a deeper integration of
    the information‑theoretic signal.

    Returns the updated weight vector and the instantaneous error.
    """
    y = predict(weights, x)
    error = target - y
    power = np.dot(x, x) + eps

    # Entropy‑aware scaling: 1/(1 + entropy) keeps the factor in (0,1]
    scale = mu / (1.0 + entropy_factor)

    delta = scale * error * x / power
    new_weights = weights + delta
    return new_weights, error


def entropy_aware_softmax(values: np.ndarray, entropy: float, temperature: float = 1.0) -> np.ndarray:
    """
    Convert a value vector to a probability distribution.
    The temperature is increased with entropy, encouraging more exploration
    when the feature distribution is uncertain.
    """
    temp = temperature * (1.0 + entropy)
    # Numerical stability
    shifted = values - np.max(values)
    exp_vals = np.exp(shifted / temp)
    probs = exp_vals / exp_vals.sum()
    return probs


def select_action(weights: np.ndarray,
                  x: np.ndarray,
                  sketch: CountMinSketch,
                  action_space: List[int]) -> int:
    """
    Choose an action using a bandit‑style softmax that incorporates:
      * the linear model's estimated reward (weights·x)
      * a count‑min sketch estimate of how often each action has been taken
      * the entropy of the current feature vector (more uncertainty → more exploration)

    The function returns the index of the selected action within ``action_space``.
    """
    # Estimate reward for each possible action as a linear function of the same features
    # (in many practical settings each action would have its own weight vector;
    # here we reuse the shared weight vector for simplicity).
    rewards = np.array([predict(weights, x) for _ in action_space])

    # Entropy of the current feature counts
    ent = shannon_entropy(x)

    # Exploration bonus inversely proportional to how often an action was taken
    # (the sketch gives an approximate count; we add 1 to avoid division by zero)
    counts = np.array([sketch.estimate(a) + 1 for a in action_space])
    exploration_bonus = 1.0 / np.sqrt(counts)

    # Combine reward and exploration, then apply entropy‑aware softmax
    combined = rewards + exploration_bonus
    probs = entropy_aware_softmax(combined, ent)

    chosen = np.random.choice(len(action_space), p=probs)
    return action_space[chosen]


# ----------------------------------------------------------------------
# Demonstration / entry point
# ----------------------------------------------------------------------
def main() -> None:
    # Initial model (3 decision‑hygiene features)
    weights = np.array([0.5, 0.3, 0.2], dtype=np.float64)

    # Example feature counts (could come from any upstream process)
    x = np.array([10, 20, 30], dtype=np.float64)

    # Target value we would like the linear model to predict
    target = 50.0

    # Hyper‑parameters
    mu = 0.5
    eps = 1e-9

    # Compute entropy once (used both for update and action selection)
    entropy = shannon_entropy(x)

    # NLMS update with entropy‑aware scaling
    new_weights, err = nlms_update(weights, x, target, mu=mu, eps=eps, entropy_factor=entropy)

    print("=== NLMS Update ===")
    print(f"Previous weights : {weights}")
    print(f"Updated weights  : {new_weights}")
    print(f"Error            : {err:.6f}")
    print(f"Feature entropy  : {entropy:.4f}")

    # Initialise a Count‑Min sketch to track action frequencies
    sketch = CountMinSketch(width=1024, depth=4, seed=42)

    # Define a small discrete action space (e.g., three possible policies)
    actions = [0, 1, 2]

    # Simulate a few selections to illustrate the adaptive behaviour
    print("\n=== Action Selection ===")
    for step in range(5):
        act = select_action(new_weights, x, sketch, actions)
        sketch.update(act)                     # record the choice
        print(f"Step {step + 1}: selected action {act}")

    # Show sketch estimates (approximate counts)
    print("\nSketch estimates after selection:")
    for a in actions:
        print(f"Action {a}: approx. count = {sketch.estimate(a)}")


if __name__ == "__main__":
    main()