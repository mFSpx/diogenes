# DARWIN HAMMER — match 5153, survivor 3
# gen: 6
# parent_a: hybrid_hybrid_sketches_hybr_hybrid_hybrid_hdc_hy_m561_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2682_s1.py (gen5)
# born: 2026-05-30T00:00:08Z

"""Hybrid Algorithm combining Count‑Min Sketch frequency estimation (Parent A) with
regex‑based textual feature extraction (Parent B).

Mathematical bridge:
- Parent A provides a count‑min sketch `S ∈ ℕ^{d×w}` for a multiset of items.
  The sketch rows are hashed count vectors; flattening `S` yields a high‑dimensional
  frequency feature vector `f_s`.
- Parent B extracts a low‑dimensional categorical count vector `f_t ∈ ℕ^{k}` from
  free‑form text using regular expressions.
- The hybrid algorithm concatenates `f = [f_s , f_t]` and treats it as a modulation
  vector for a surrogate model `M = {centers_i, weight_i}`.
  Each center `c_i ∈ ℝ^{|f|}` is compared to `f` using the inner‑product similarity,
  then passed through a Gaussian kernel `g(r)=exp(-(ε r)^2)`. The weighted sum of the
  kernel outputs yields a surrogate prediction (e.g., expected reward).

Thus the core topologies of both parents are fused through a shared vector space
and a common similarity‑kernel operation.
"""

import hashlib
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from collections import defaultdict
import re
import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# ----------------------------------------------------------------------
# Global policy store (Parent A)
# ----------------------------------------------------------------------
_POLICY: dict[str, list[float]] = {}

def reset_policy() -> None:
    """Clear all stored rewards."""
    _POLICY.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Accumulate rewards per action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])  # [total_reward, count]
        stats[0] += float(u.reward)
        stats[1] += 1.0

def _reward(action_id: str) -> float:
    """Mean reward for an action, or 0 if unseen."""
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items: list[str], width: int = 64, depth: int = 4) -> np.ndarray:
    """Return a depth×width integer sketch matrix."""
    table = np.zeros((depth, width), dtype=int)
    for item in items:
        for d in range(depth):
            h = int(hashlib.sha256(f'{d}:{item}'.encode()).hexdigest(), 16)
            table[d, h % width] += 1
    return table

# ----------------------------------------------------------------------
# Regex feature extraction (Parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|pa)\b", re.I)

def extract_text_features(text: str) -> np.ndarray:
    """Return a 7‑dim integer vector counting regex matches."""
    return np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
        len(OUTCOME_RE.findall(text)),
        len(IMPULSIVE_RE.findall(text)),
    ], dtype=int)

# ----------------------------------------------------------------------
# Core similarity / kernel utilities (shared)
# ----------------------------------------------------------------------
def inner_product(a: np.ndarray, b: np.ndarray) -> float:
    """Standard inner product; vectors must have equal length."""
    if a.shape != b.shape:
        raise ValueError("vectors must have the same shape")
    return float(np.dot(a, b))

def gaussian_kernel(r: float, epsilon: float = 1.0) -> float:
    """Radial basis function kernel."""
    return math.exp(-((epsilon * r) ** 2))

# ----------------------------------------------------------------------
# Surrogate model (adapted from Parent A)
# ----------------------------------------------------------------------
def init_surrogate(num_centers: int, dim: int) -> dict:
    """Create a surrogate with random centers and uniform weights."""
    centers = [np.random.randn(dim) for _ in range(num_centers)]
    weights = np.random.rand(num_centers).tolist()
    return {"centers": centers, "weights": weights}

def modulate_surrogate(surrogate: dict, modulation_vec: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Compute a surrogate prediction:
        y = Σ_i w_i * g( <c_i, v> )
    where g is the Gaussian kernel, <·,·> is inner product, and v is the modulation vector.
    """
    total = 0.0
    for center, weight in zip(surrogate["centers"], surrogate["weights"]):
        sim = inner_product(center, modulation_vec)
        total += weight * gaussian_kernel(sim, epsilon)
    return total

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_modulation_vector(items: list[str], context_text: str,
                            sketch_width: int = 64, sketch_depth: int = 4) -> np.ndarray:
    """
    Produce a single modulation vector by concatenating:
        - flattened count‑min sketch of `items`
        - regex feature vector of `context_text`
    """
    sketch = count_min_sketch(items, width=sketch_width, depth=sketch_depth)
    sketch_flat = sketch.flatten().astype(float)  # shape (depth*width,)
    text_feat = extract_text_features(context_text).astype(float)  # shape (7,)
    return np.concatenate([sketch_flat, text_feat])

def predict_action_reward(action: BanditAction,
                          items: list[str],
                          context_text: str,
                          surrogate: dict,
                          epsilon: float = 1.0) -> float:
    """
    Hybrid prediction for a given action:
        1. Build modulation vector from the observed items and textual context.
        2. Pass it through the surrogate to obtain a raw score.
        3. Blend with the empirical mean reward from the policy store.
    """
    mod_vec = build_modulation_vector(items, context_text)
    surrogate_score = modulate_surrogate(surrogate, mod_vec, epsilon)
    empirical = _reward(action.action_id)
    # Simple linear blend; coefficients can be tuned.
    return 0.7 * surrogate_score + 0.3 * empirical

def batch_predict(actions: list[BanditAction],
                  items_per_action: dict[str, list[str]],
                  context_per_action: dict[str, str],
                  surrogate: dict,
                  epsilon: float = 1.0) -> dict[str, float]:
    """
    Compute predictions for a batch of actions.
    Returns a mapping action_id → predicted value.
    """
    results = {}
    for act in actions:
        items = items_per_action.get(act.action_id, [])
        ctx = context_per_action.get(act.action_id, "")
        results[act.action_id] = predict_action_reward(act, items, ctx, surrogate, epsilon)
    return results

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a tiny policy with some historic updates
    reset_policy()
    updates = [
        BanditUpdate(context_id="c1", action_id="a1", reward=1.0, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="a1", reward=0.0, propensity=0.5),
        BanditUpdate(context_id="c3", action_id="a2", reward=1.0, propensity=0.8),
    ]
    update_policy(updates)

    # Define actions
    actions = [
        BanditAction(action_id="a1", propensity=0.5, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="a2", propensity=0.8, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="a3", propensity=0.3, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
    ]

    # Synthetic items and contexts
    items_per_action = {
        "a1": ["apple", "banana", "apple"],
        "a2": ["orange", "banana", "kiwi", "banana"],
        "a3": ["pear", "pear", "apple"],
    }
    context_per_action = {
        "a1": "We have verified the source and have a plan to ship the product.",
        "a2": "Need to wait before we can confirm the outcome; possible delay.",
        "a3": "User expressed rage and impulsive behavior, need support.",
    }

    # Initialise surrogate (dimension = depth*width + 7)
    dim = 4 * 64 + 7
    surrogate = init_surrogate(num_centers=10, dim=dim)

    # Run batch prediction
    preds = batch_predict(actions, items_per_action, context_per_action, surrogate)

    for aid, val in preds.items():
        print(f"Action {aid}: predicted value = {val:.4f}")

    # Ensure no exception
    sys.exit(0)