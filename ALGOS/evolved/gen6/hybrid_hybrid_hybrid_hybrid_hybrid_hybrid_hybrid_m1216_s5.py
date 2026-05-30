# DARWIN HAMMER — match 1216, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m1192_s0.py (gen5)
# born: 2026-05-29T23:34:26Z

"""Hybrid Fusion of Bandit‑Router Store Dynamics with Workshare Allocation via Text‑Signature Features.

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hard_truth_ma_m551_s0.py (Bandit‑Router / Store update)
- hybrid_hybrid_hybrid_hybrid_hybrid_hard_t_m1192_s0.py (Stylometry‑driven Workshare allocator)

Mathematical Bridge:
Both parents expose a *vector‑valued* intermediate:
* Parent A* produces a signature vector `σ ∈ ℝ^k` from a context (e.g. a B‑spline projected path signature) and uses it in the store update

ℓ_{t+1} = ℓ_t + dt·(α·ℓ_t + β·⟨w, σ⟩)          (1)

where `w` are learned coefficients.

* Parent B* consumes a *feature vector* derived from stylometric counts to compute a work‑share distribution across LLM groups.

The fusion treats the signature `σ` from A as the stylometric feature vector for B.  
Equation (1) updates the honey‑bee store, then the updated store level is normalised to obtain a probability simplex `π`.  
These probabilities directly drive the work‑share allocation:

share_g = π_g · total_units                       (2)

Thus the store dynamics modulate the allocation, while the allocation feeds back into bandit propensities (via a rescaling factor).

The module implements three core functions that demonstrate this closed‑loop hybrid system:
1. `compute_signature(text)` – builds a low‑dimensional signature from stylometric categories.
2. `store_update_and_allocate(sig, store, total_units)` – performs (1) and (2) in one step.
3. `adjust_bandit_propensities(updates, allocation)` – rescales bandit propensities using the allocation outcome.

All operations rely only on NumPy and the Python standard library.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A (Bandit‑Router)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Result of an action selection."""
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    """Single observation used to update the policy."""
    context_id: str
    action_id: str
    reward: float
    propensity: float


@dataclass
class StoreState:
    """Honeybee‑style store and derived control signal."""
    level: float = 0.0          # ℓ_t
    alpha: float = 1.0          # decay / reinforcement coefficient
    beta: float = 1.0           # scaling of signature influence
    dt: float = 1.0             # time step
    base: float = 1.0           # reference level (unused in this fusion)


# ----------------------------------------------------------------------
# Data structures from Parent B (Stylometry + Workshare)
# ----------------------------------------------------------------------


FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

GROUPS = ("codex", "groq", "cohere", "local_models")


def words(text: str) -> List[str]:
    """Simple tokeniser returning alphabetic lower‑case words."""
    return [w.lower() for w in text.split() if w.isalpha()]


@dataclass
class WorkshareLane:
    group: str
    llm_units: float
    llm_share_pct: float
    proof_required: bool = False


# ----------------------------------------------------------------------
# Fusion Core Functions
# ----------------------------------------------------------------------


def compute_signature(text: str) -> np.ndarray:
    """
    Build a low‑dimensional signature vector from stylometric category counts.

    The vector has length ``len(FUNCTION_CATS)``.  Each entry is the
    normalised frequency of words belonging to the corresponding category.
    This mimics the B‑spline projected signature used in Parent A while
    reusing the linguistic features of Parent B.

    Parameters
    ----------
    text: str
        Input text whose stylometric signature is required.

    Returns
    -------
    np.ndarray
        Normalised signature vector ``σ`` with shape ``(k,)``.
    """
    token_list = words(text)
    total = max(len(token_list), 1)  # avoid division by zero
    sig = np.zeros(len(FUNCTION_CATS), dtype=float)

    for idx, (cat, vocab) in enumerate(FUNCTION_CATS.items()):
        count = sum(1 for w in token_list if w in vocab)
        sig[idx] = count / total

    # Optional: smooth with a simple B‑spline‑like kernel (here a moving average)
    kernel = np.ones(3) / 3.0
    sig = np.convolve(sig, kernel, mode="same")
    return sig


def store_update_and_allocate(
    signature: np.ndarray,
    store: StoreState,
    total_units: float,
) -> Tuple[StoreState, Dict[str, WorkshareLane]]:
    """
    Perform the hybrid dynamics:
    1. Update the honeybee store using the signature (eq. 1).
    2. Normalise the new store level into a probability simplex.
    3. Allocate ``total_units`` across LLM groups proportionally
       to the simplex (eq. 2).

    Parameters
    ----------
    signature : np.ndarray
        Signature vector σ from ``compute_signature``.
    store : StoreState
        Current store state (will be mutated).
    total_units : float
        Total LLM computational units to distribute.

    Returns
    -------
    Tuple[StoreState, Dict[str, WorkshareLane]]
        Updated store and a dict mapping group names to ``WorkshareLane`` objects.
    """
    # --- 1. Store update -------------------------------------------------
    # w is a simple weight vector emphasising early signature components.
    w = np.linspace(1.0, 0.5, len(signature))
    inflow = store.beta * np.dot(w, signature)   # ⟨w, σ⟩ term
    dlevel = store.dt * (store.alpha * store.level + inflow)
    store.level = max(store.level + dlevel, 0.0)  # store level stays non‑negative

    # --- 2. Normalise to simplex -----------------------------------------
    # Use a softmax to guarantee positivity and sum‑to‑one.
    logits = np.array([store.level, store.level * 0.8, store.level * 0.6, store.level * 0.4])
    max_log = np.max(logits)
    exp_logits = np.exp(logits - max_log)
    probs = exp_logits / exp_logits.sum()  # π_g for each group

    # --- 3. Allocation ----------------------------------------------------
    allocation: Dict[str, WorkshareLane] = {}
    for i, group in enumerate(GROUPS):
        share = probs[i] * total_units
        lane = WorkshareLane(
            group=group,
            llm_units=share,
            llm_share_pct=round(100.0 * probs[i], 2),
            proof_required=False,
        )
        allocation[group] = lane

    return store, allocation


def adjust_bandit_propensities(
    updates: List[BanditUpdate],
    allocation: Dict[str, WorkshareLane],
) -> List[BanditAction]:
    """
    Rescale bandit propensities based on the workshare allocation.
    The idea is that groups receiving more units are considered more
    “promising”; their associated actions receive a proportional boost.

    Parameters
    ----------
    updates : List[BanditUpdate]
        Observations collected during the current round.
    allocation : Dict[str, WorkshareLane]
        Result of ``store_update_and_allocate``.

    Returns
    -------
    List[BanditAction]
        New actions with adjusted propensities.
    """
    # Build a mapping from group to a boost factor (units normalised).
    total_units = sum(l.llm_units for l in allocation.values())
    boost = {g: (l.llm_units / total_units) if total_units > 0 else 0.0 for g, l in allocation.items()}

    actions: List[BanditAction] = []
    for upd in updates:
        # Derive a pseudo‑group from the context_id (hash modulo number of groups)
        idx = hash(upd.context_id) % len(GROUPS)
        group = GROUPS[idx]

        # Original propensity is given; we apply the boost multiplicatively.
        new_propensity = upd.propensity * (1.0 + boost.get(group, 0.0))

        # Simple expected reward estimate: reward * new_propensity
        exp_reward = upd.reward * new_propensity

        action = BanditAction(
            action_id=upd.action_id,
            propensity=new_propensity,
            expected_reward=exp_reward,
            confidence_bound=math.sqrt(new_propensity),  # placeholder
            algorithm=f"Hybrid-{group}",
        )
        actions.append(action)

    return actions


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text to generate a stylometric signature.
    sample_text = (
        "The quick brown fox jumps over the lazy dog while the AI models "
        "evaluate their own performance and allocate resources accordingly."
    )

    # 1. Compute signature.
    sigma = compute_signature(sample_text)
    print("Signature vector:", sigma)

    # 2. Initialise store.
    store = StoreState(level=5.0, alpha=0.9, beta=0.4, dt=1.0)

    # 3. Perform hybrid update + allocation.
    updated_store, lanes = store_update_and_allocate(sigma, store, total_units=100.0)
    print("\nUpdated store level:", updated_store.level)
    print("Workshare allocation:")
    for lane in lanes.values():
        print(f"  {lane.group}: {lane.llm_units:.2f} units ({lane.llm_share_pct}%)")

    # 4. Mock bandit updates.
    mock_updates = [
        BanditUpdate(context_id="ctx1", action_id="a1", reward=1.2, propensity=0.3),
        BanditUpdate(context_id="ctx2", action_id="a2", reward=0.5, propensity=0.6),
        BanditUpdate(context_id="ctx3", action_id="a3", reward=0.8, propensity=0.4),
    ]

    # 5. Adjust propensities based on allocation.
    actions = adjust_bandit_propensities(mock_updates, lanes)
    print("\nAdjusted Bandit Actions:")
    for act in actions:
        print(
            f"Action {act.action_id} (alg={act.algorithm}) -> propensity={act.propensity:.3f}, "
            f"exp_reward={act.expected_reward:.3f}"
        )

    sys.exit(0)