# DARWIN HAMMER — match 3963, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s1.py (gen4)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_sketch_m983_s0.py (gen4)
# born: 2026-05-29T23:52:49Z

"""Hybrid Darwin-Capybara Sketch Algorithm
This module fuses the core topologies of:

* **Parent A** – stylometry analysis and geometric‑product calculations.
* **Parent B** – count‑min sketch, VRAM budgeting and contextual bandit routing.

Mathematical bridge:
The frequency distribution of linguistic categories (Parent A) is compressed with a
count‑min sketch (Parent B).  The sketch provides a probabilistic estimate of each
category’s count which is used as a weighting factor in the geometric product of
the stylometric feature vectors.  The resulting weighted product feeds a contextual
bandit whose reward estimates are further scaled by the sketch‑derived frequency
factor, while the VRAM budget is estimated from the sketch size.  Thus the
sketch links the two algorithmic worlds both at the data‑compression level and
at the decision‑making level."""


import math
import random
import sys
import pathlib
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared linguistic resources (from Parent A)
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, set[str]] = {
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
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


def _clean_token(tok: str) -> str:
    """Lower‑case token stripped of surrounding punctuation."""
    return tok.strip(PUNCT).lower()


def tokenize(text: str) -> List[str]:
    """Very simple whitespace tokenizer with punctuation stripping."""
    return [_clean_token(t) for t in text.split() if _clean_token(t)]


# ----------------------------------------------------------------------
# Count‑Min Sketch utilities (from Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: List[str], width: int = 64, depth: int = 4) -> List[List[int]]:
    """Return a count‑min sketch matrix for *items*."""
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def estimate_frequency(sketch: List[List[int]], query: str, width: int = 64) -> int:
    """Probabilistic upper‑bound on the frequency of *query* using the sketch."""
    mins = []
    for d, row in enumerate(sketch):
        h = hashlib.sha256(f"{d}:{query}".encode()).hexdigest()
        idx = int(h, 16) % width
        mins.append(row[idx])
    return min(mins)


def estimate_vram_usage(sketch: List[List[int]], budget_mb: int, reserve_mb: int) -> int:
    """
    Rough VRAM usage estimate:
    total count in sketch * (reserve / 100) + static budget.
    """
    total_counts = sum(sum(row) for row in sketch)
    estimated = total_counts * reserve_mb // 100 + budget_mb
    return int(estimated)


# ----------------------------------------------------------------------
# Bandit data structures (from Parent B)
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
# Hybrid core – stylometry + sketch + geometric product + bandit
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}  # action_id -> [cumulative_reward, count]


def reset_policy() -> None:
    """Clear the internal bandit policy."""
    _POLICY.clear()


def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate a batch of bandit updates."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0


def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0


def stylometric_category_vector(text: str, sketch_width: int = 64) -> np.ndarray:
    """
    Compute a normalized vector of category frequencies.
    The raw counts are obtained via a count‑min sketch to keep the
    operation consistent with the hybrid bridge.
    """
    tokens = tokenize(text)
    sketch = count_min_sketch(tokens, width=sketch_width, depth=4)

    # Estimate frequency for each category by summing estimates of its words
    cat_counts = []
    for cat, vocab in FUNCTION_CATS.items():
        est = sum(estimate_frequency(sketch, w, width=sketch_width) for w in vocab)
        cat_counts.append(est)

    vec = np.array(cat_counts, dtype=float)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def weighted_geometric_product(vec_a: np.ndarray, vec_b: np.ndarray, weight: float) -> np.ndarray:
    """
    Perform a geometric‑product‑like operation (outer product followed by
    contraction) and scale the result by *weight*.
    """
    outer = np.outer(vec_a, vec_b)               # shape (d, d)
    # Contract to a vector by summing the diagonal (a simple inner‑product analogue)
    diag_sum = np.trace(outer)
    result = outer * weight + diag_sum
    return result


def hybrid_action_estimate(
    action: BanditAction,
    sketch: List[List[int]],
    budget_mb: int,
    reserve_mb: int,
    query_token: str,
) -> float:
    """
    Combine the bandit’s expected reward with a frequency‑derived scaling factor.
    The scaling factor is the estimated count of *query_token* divided by the
    total sketch mass, providing a smooth bridge between linguistic statistics
    and decision‑making.
    """
    # Base reward from the bandit
    base = _reward(action.action_id) if _reward(action.action_id) else action.expected_reward

    # Frequency factor
    total = sum(sum(row) for row in sketch) + 1e-6
    freq = estimate_frequency(sketch, query_token) / total

    # VRAM‑adjusted penalty (larger usage reduces reward)
    vram_est = estimate_vram_usage(sketch, budget_mb, reserve_mb)
    vram_penalty = 1.0 / (1.0 + math.log1p(vram_est))

    # Final hybrid estimate
    return base * (1.0 + freq) * vram_penalty


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def hybrid_process(text: str, action: BanditAction) -> Tuple[np.ndarray, float]:
    """
    Full hybrid pipeline:
    1. Build a sketch from *text*.
    2. Derive a stylometric vector.
    3. Compute a weighted geometric product with the action’s propensity.
    4. Return the product and a hybrid reward estimate.
    """
    tokens = tokenize(text)
    sketch = count_min_sketch(tokens)

    # Stylometric vector (category space)
    styl_vec = stylometric_category_vector(text)

    # Weight derived from action propensity (clamped to [0,1])
    w = max(0.0, min(1.0, action.propensity))

    # Geometric product
    product = weighted_geometric_product(styl_vec, np.full_like(styl_vec, w), weight=w)

    # Hybrid reward estimate using the most frequent token as query
    query = max(set(tokens), key=tokens.count) if tokens else ""
    reward_est = hybrid_action_estimate(
        action, sketch, budget_mb=256, reserve_mb=64, query_token=query
    )
    return product, reward_est


def sample_bandit_action() -> BanditAction:
    """Create a deterministic dummy bandit action for testing."""
    return BanditAction(
        action_id="stylometry_boost",
        propensity=0.7,
        expected_reward=0.5,
        confidence_bound=0.1,
        algorithm="HybridDarwinCapybara",
    )


def run_smoke_test() -> None:
    """Execute a minimal end‑to‑end run to verify that all components interoperate."""
    reset_policy()
    txt = (
        "The quick brown fox jumps over the lazy dog while the cat watches. "
        "I think therefore I am. You and I are together."
    )
    action = sample_bandit_action()
    product, reward = hybrid_process(txt, action)

    # Perform a trivial policy update
    upd = BanditUpdate(context_id="test", action_id=action.action_id, reward=reward, propensity=action.propensity)
    update_policy([upd])

    # Print shapes / values to prove execution (no external I/O constraints)
    print("Product shape:", product.shape)
    print("Sample product entry:", product.flat[0])
    print("Hybrid reward estimate:", reward)
    print("Policy entry:", _POLICY.get(action.action_id))


if __name__ == "__main__":
    run_smoke_test()