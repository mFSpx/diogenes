# DARWIN HAMMER — match 3387, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1645_s2.py (gen6)
# born: 2026-05-29T23:49:41Z

"""Hybrid algorithm combining MinHash‑style binary hashing and Bayesian updating
(from hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1012_s1.py) with
weekday‑based sheaf‑cohomology weight vectors and linear action utility
projection (from hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1645_s2.py).

Mathematical bridge:
- The binary hash produced by ``compute_phash`` yields a 64‑bit integer.
- ``hamming_distance`` between two hashes supplies a similarity measure
  ``sim = 1 - hd/64``.
- This similarity is interpreted as a likelihood in a Bayesian update.
- The prior for each action is taken from the weekday weight vector
  ``weekday_weight_vector`` which distributes probability mass over a
  configurable group list.
- The posterior is then multiplied by a linear utility projection that
  combines an action's expected value, cost, risk and a feature‑based
  hygiene score (positive/negative weight dot‑product).
Thus the two parent topologies are fused through a single matrix‑style
computation:

posterior_i = BayesUpdate(prior_i, likelihood_i, marginal)
utility_i   = posterior_i * (ev_i - cost_i - risk_i) * hygiene_i
"""

import math
import random
import sys
from pathlib import Path
from datetime import date
from dataclasses import dataclass
from typing import List, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Parent A components (MinHash / Bayesian)
# ----------------------------------------------------------------------
def compute_phash(values: List[float]) -> int:
    """Compute a 64‑bit binary hash by thresholding values against their mean."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    bits = 0
    for v in values[:64]:
        bits = (bits << 1) | int(v >= avg)
    return bits

def hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two integers."""
    return (a ^ b).bit_count()

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Marginal probability for a binary hypothesis."""
    if not (0.0 <= prior <= 1.0 and 0.0 <= likelihood <= 1.0 and 0.0 <= false_positive <= 1.0):
        raise ValueError("All probabilities must lie in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

def bayes_update(prior: float, likelihood: float, marginal: float) -> float:
    """Bayesian posterior given prior, likelihood and marginal."""
    if marginal <= 0.0:
        raise ValueError("Marginal probability must be > 0")
    return prior * likelihood / marginal

def extract_features(text: str) -> np.ndarray:
    """Simple regex‑based feature extraction returning a 9‑dim integer vector."""
    import re
    counts = []
    FEATURE_REGEXES = [
        ("evidence", r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"),
        ("planning", r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"),
        ("delay", r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|b)\b"),
        ("quality", r"\b(?:quality|high|low|grade|rating)\b"),
        ("security", r"\b(?:security|secure|vulnerability|exploit)\b"),
        ("performance", r"\b(?:performance|fast|slow|latency)\b"),
        ("compliance", r"\b(?:compliance|regulation|standard)\b"),
        ("cost", r"\b(?:cost|price|budget|expense)\b"),
        ("generic", r"\b\w{7,}\b"),
    ]
    for _, pattern in FEATURE_REGEXES:
        matches = re.findall(pattern, text, re.I)
        counts.append(len(matches))
    return np.array(counts, dtype=int)

def hybrid_hygiene_score(v: np.ndarray,
                         w_pos: np.ndarray,
                         w_neg: np.ndarray) -> float:
    """
    Compute a hygiene score = (v·w_pos) - (v·w_neg) normalized to [0,1].
    v : feature vector (9,)
    w_pos, w_neg : weight vectors (9,)
    """
    if v.shape != (9,) or w_pos.shape != (9,) or w_neg.shape != (9,):
        raise ValueError("All vectors must have shape (9,)")
    pos = float(v @ w_pos)
    neg = float(v @ w_neg)
    raw = pos - neg
    # Normalization: map raw from its possible range to [0,1]
    # Assuming each component ≤ max count (here we conservatively bound by 10)
    max_possible = 10 * (w_pos.sum() + w_neg.sum())
    return max(0.0, min(1.0, (raw + max_possible) / (2 * max_possible)))

# ----------------------------------------------------------------------
# Parent B components (weekday weight vector, actions)
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")

def weekday_weight_vector(groups: tuple, dow: int) -> np.ndarray:
    """
    Produce a normalized weight vector for *groups* based on the weekday ``dow``.
    dow: 0=Sunday … 6=Saturday
    """
    n = len(groups)
    base_angles = np.arange(n) * (2.0 * math.pi) / n
    phase = (2.0 * math.pi) * (dow % 7) / 7.0
    amplitude = 0.2
    raw = 1.0 + amplitude * np.sin(base_angles + phase)
    weight_vec = raw / raw.sum()
    return weight_vec.astype(np.float64)

@dataclass(frozen=True)
class MathAction:
    """Represent a potential action with economic attributes."""
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

# ----------------------------------------------------------------------
# Hybrid functions
# ----------------------------------------------------------------------
def similarity_likelihood(hash_a: int, hash_b: int) -> float:
    """
    Transform normalized Hamming similarity into a likelihood in [0,1].
    """
    hd = hamming_distance(hash_a, hash_b)
    similarity = 1.0 - hd / 64.0   # 0 (completely different) … 1 (identical)
    # Clip to avoid pathological zero marginal later
    return max(0.01, min(0.99, similarity))

def compute_action_posteriors(actions: List[MathAction],
                              prior_weights: np.ndarray,
                              likelihood: float) -> List[float]:
    """
    Given a list of actions, a vector of priors (summing to 1) and a common
    likelihood, compute Bayesian posteriors for each action.
    """
    if len(actions) != len(prior_weights):
        raise ValueError("Number of actions must match length of prior vector")
    marginal = bayes_marginal(prior_weights.mean(), likelihood, 1 - likelihood)
    posteriors = [bayes_update(p, likelihood, marginal) for p in prior_weights]
    # Re‑normalize to ensure they sum to 1 (numerical safety)
    total = sum(posteriors)
    return [p / total for p in posteriors]

def hybrid_utility_scores(text: str,
                          reference_text: str,
                          actions: List[MathAction],
                          groups: tuple = GROUPS,
                          dow: int = None) -> List[Tuple[str, float]]:
    """
    End‑to‑end hybrid scoring:
    1. Extract feature vectors from *text* and *reference_text*.
    2. Produce binary hashes and a similarity‑derived likelihood.
    3. Build weekday‑based priors for each action.
    4. Bayesian update to obtain posteriors.
    5. Compute a hygiene score from the feature vector.
    6. Combine posterior, hygiene, and linear utility (ev‑cost‑risk).
    Returns a list of (action_id, utility) sorted descending.
    """
    # 1. Feature extraction
    feats = extract_features(text)
    ref_feats = extract_features(reference_text)

    # 2. Hashes and likelihood
    h1 = compute_phash(feats.tolist())
    h2 = compute_phash(ref_feats.tolist())
    likelihood = similarity_likelihood(h1, h2)

    # 3. Weekday priors
    if dow is None:
        dow = (date.today().weekday() + 1) % 7  # align with parent B convention
    priors = weekday_weight_vector(groups, dow)
    # If there are more actions than groups, repeat the pattern cyclically
    if len(actions) > len(priors):
        repeats = (len(actions) + len(priors) - 1) // len(priors)
        priors = np.tile(priors, repeats)[:len(actions)]
    else:
        priors = priors[:len(actions)]

    # 4. Bayesian posteriors
    posteriors = compute_action_posteriors(actions, priors, likelihood)

    # 5. Hygiene score (same for all actions, derived from *text*)
    # Simple positive/negative weight vectors – could be learned; here static.
    w_pos = np.array([1.0, 0.8, 0.2, 0.5, 0.9, 0.6, 0.4, 0.3, 0.7])
    w_neg = np.array([0.5, 0.3, 0.7, 0.2, 0.1, 0.4, 0.6, 0.8, 0.2])
    hygiene = hybrid_hygiene_score(feats, w_pos, w_neg)

    # 6. Final utility composition
    utilities = []
    for act, post in zip(actions, posteriors):
        linear_utility = act.expected_value - act.cost - act.risk
        utility = post * linear_utility * hygiene
        utilities.append((act.id, utility))

    utilities.sort(key=lambda kv: kv[1], reverse=True)
    return utilities

def random_action_set(n: int) -> List[MathAction]:
    """Generate *n* synthetic actions with random economic parameters."""
    actions = []
    for i in range(n):
        ev = random.uniform(0, 100)
        cost = random.uniform(0, 30)
        risk = random.uniform(0, 20)
        actions.append(MathAction(id=f"act_{i}", expected_value=ev, cost=cost, risk=risk))
    return actions

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "The verification log shows that the security patch was applied. "
        "We have a detailed plan with steps and a timeline. "
        "Performance metrics indicate low latency."
    )
    reference = (
        "Audit records confirm the patch deployment. The roadmap includes "
        "testing phases and cost estimates. Security is high."
    )
    actions = random_action_set(6)

    scores = hybrid_utility_scores(sample_text, reference, actions)
    for aid, util in scores:
        print(f"{aid}: {util:.4f}")