# DARWIN HAMMER — match 5368, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2597_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s2.py (gen4)
# born: 2026-05-30T00:01:35Z

"""Hybrid Fractional‑Memory & Regex‑Bandit Fusion Module
====================================================

This module fuses the two parent algorithms:

* **Parent A – Hybrid Fractional‑Memory Phantasmagoria**  
  Provides a deterministic/LLM split, an input‑dependent effective time constant,
  and a Caputo fractional‑memory term that modulates allocations together with a
  pheromone‑infused priority matrix.

* **Parent B – Regex‑Feature + Fisher‑Info + Bandit‑Capybara Optimiser**  
  Scores textual evidence using regexes, builds a Fisher‑information‑like diagonal
  matrix from those scores, and runs a simple Thompson‑sampling bandit whose
  `propensity` and `confidence_bound` drive a stochastic optimisation primitive.

**Mathematical Bridge**

The bridge is built on two shared constructs:

1. **Feature‑derived scalar `s_t`** – the total weighted regex score for a given
   time step *t*.  Parent B supplies `s_t`; Parent A treats it as the external
   stimulus that determines the *effective time constant* `τ_t`.

2. **Bandit‑modulated memory kernel** – the bandit yields a `propensity p_t ∈ [0,1]`
   and a `confidence_bound c_t`.  These are injected into the Caputo fractional
   derivative kernel `K_{t,k}` as a multiplicative factor
   `K_{t,k}=p_t·(t‑k)^{‑α}·exp(‑c_t·(t‑k))`.  Thus the bandit directly shapes the
   memory contribution that modulates the pheromone‑driven allocation.

The resulting allocation for group *g* on day *t* is


a_{g,t} =  λ_g· (1‑τ_t)·L_t  +  (1‑λ_g)·τ_t·M_{g,t}


where

* `λ_g` is the deterministic/LLM split (from Parent A),
* `L_t` is the raw LLM share (baseline),
* `M_{g,t}` is the pheromone‑infused priority,
* `τ_t = sigmoid(β·s_t)` is the effective time constant,
* `M_{g,t}` is further scaled by the fractional‑memory term
  `F_t = Σ_{k<t} K_{t,k}·a_{g,k}`.

The code below implements this fusion, exposing three public functions
`init_hybrid_system`, `hybrid_allocate`, and `update_bandit_memory`,
plus a small smoke test. """

import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Regex feature definitions (Parent B)
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
    r"\b(?:boundary|boundaries|walk away|no contact|do not|block|ignore|distance|safe|safe distance|no talk|no communication|stop|stop talking|stop interaction|no interaction|leave)\b",
    re.I,
)

REGEX_PATTERNS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
}

# ----------------------------------------------------------------------
# Bandit core (Parent B)
# ----------------------------------------------------------------------
@dataclass
class ThompsonBandit:
    """Simple Thompson‑sampling bandit with Beta priors."""
    successes: float = 1.0
    failures: float = 1.0
    # dynamic confidence bound (upper‑confidence index)
    confidence_scale: float = 1.0

    def sample(self) -> float:
        """Draw a propensity sample from the posterior Beta distribution."""
        return random.betavariate(self.successes, self.failures)

    @property
    def propensity(self) -> float:
        """Mean of the Beta posterior – used as deterministic propensity."""
        return self.successes / (self.successes + self.failures)

    @property
    def confidence_bound(self) -> float:
        """A simple confidence interval width."""
        var = (self.successes * self.failures) / (
            (self.successes + self.failures) ** 2 * (self.successes + self.failures + 1)
        )
        return self.confidence_scale * math.sqrt(var)

    def update(self, reward: float) -> None:
        """Update posterior with binary reward (1 = success, 0 = failure)."""
        if reward >= 0.5:
            self.successes += 1
        else:
            self.failures += 1


# ----------------------------------------------------------------------
# Fractional‑memory utilities (Parent A)
# ----------------------------------------------------------------------
def _caputo_kernel(alpha: float, delta: int, propensity: float, conf: float) -> float:
    """
    Fractional‑memory kernel for a lag `delta` (t‑k) with bandit modulation.
    K = propensity * (delta)^{-alpha} * exp(-conf * delta)
    """
    if delta <= 0:
        return 0.0
    return propensity * (delta ** (-alpha)) * math.exp(-conf * delta)


def fractional_memory(
    past_alloc: np.ndarray,
    alpha: float,
    propensity: float,
    confidence: float,
) -> float:
    """
    Compute the Caputo‑type memory contribution for the current step.
    `past_alloc` is a 1‑D array of previous allocations for a single group.
    """
    if past_alloc.size == 0:
        return 0.0
    deltas = np.arange(past_alloc.size, 0, -1)  # t‑k = 1,2,…,n
    kernels = np.vectorize(_caputo_kernel)(alpha, deltas, propensity, confidence)
    return float(np.dot(kernels, past_alloc))


# ----------------------------------------------------------------------
# Hybrid system data container
# ----------------------------------------------------------------------
@dataclass
class HybridState:
    """Holds all mutable state required for the hybrid algorithm."""
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
    # deterministic/LLM split λ_g for each group (sum to 1 across groups)
    lambda_split: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.3, 0.2, 0.1]))
    # pheromone priority matrix M_{g,t} (initialized uniformly)
    pheromone: np.ndarray = field(init=False)  # shape (n_groups, )
    # allocation history per group (list of lists)
    history: List[List[float]] = field(default_factory=lambda: [[] for _ in range(4)])
    # bandit instance
    bandit: ThompsonBandit = field(default_factory=ThompsonBandit)
    # fractional order α (0<α<1)
    alpha: float = 0.7
    # scaling factor β for effective time constant τ_t = sigmoid(β·s_t)
    beta: float = 0.05

    def __post_init__(self):
        self.pheromone = np.ones(len(self.groups)) / len(self.groups)


# ----------------------------------------------------------------------
# Core hybrid functions
# ----------------------------------------------------------------------
def init_hybrid_system() -> HybridState:
    """Create a fresh HybridState with sensible defaults."""
    return HybridState()


def score_features(text: str) -> Dict[str, int]:
    """
    Apply the regex catalogue to `text` and return a dict of hit counts.
    Mirrors Parent B's feature extraction.
    """
    scores = {}
    for name, pattern in REGEX_PATTERNS.items():
        scores[name] = len(pattern.findall(text))
    return scores


def effective_time_constant(total_score: float, beta: float) -> float:
    """
    Map the aggregated feature score `total_score` to an effective time constant τ∈(0,1)
    using a logistic sigmoid.
    """
    return 1.0 / (1.0 + math.exp(-beta * total_score))


def hybrid_allocate(state: HybridState, text: str, baseline_llm_share: float = 0.6) -> Dict[str, float]:
    """
    Compute per‑group allocation for a single time step.

    Steps:
    1. Feature scoring → total score `s_t`.
    2. τ_t = sigmoid(β·s_t) (effective time constant).
    3. Bandit draws propensity `p_t` and confidence `c_t`.
    4. Fractional memory `F_{g,t}` for each group via past history.
    5. Pheromone matrix is updated proportionally to Fisher‑information‑like
       diagonal derived from feature counts.
    6. Final allocation follows the hybrid equation:
       a_{g,t} = λ_g·(1‑τ_t)·L_t + (1‑λ_g)·τ_t·(M_{g,t}+F_{g,t})
    """
    # 1. Feature scoring
    feats = score_features(text)
    total_score = sum(feats.values())

    # 2. Effective time constant
    tau = effective_time_constant(total_score, state.beta)

    # 3. Bandit draw
    p = state.bandit.propensity
    c = state.bandit.confidence_bound

    # 4. Fractional memory per group
    mem_terms = np.zeros(len(state.groups))
    for idx, hist in enumerate(state.history):
        past = np.array(hist, dtype=float)
        mem_terms[idx] = fractional_memory(past, state.alpha, p, c)

    # 5. Fisher‑information‑like diagonal (variance proxy) → update pheromone
    #    Use counts as a surrogate for information; normalize to sum 1.
    info_vec = np.array([feats.get(k, 0) for k in REGEX_PATTERNS.keys()], dtype=float)
    if info_vec.sum() > 0:
        info_norm = info_vec / info_vec.sum()
        # Broadcast to groups (simple repeat) and blend with existing pheromone
        blended = 0.7 * state.pheromone + 0.3 * np.resize(info_norm, state.pheromone.shape)
        state.pheromone = blended / blended.sum()  # re‑normalize

    # 6. Allocation computation
    lambda_arr = state.lambda_split
    L = baseline_llm_share
    M = state.pheromone
    allocations = {}
    for idx, grp in enumerate(state.groups):
        a = (
            lambda_arr[idx] * (1.0 - tau) * L
            + (1.0 - lambda_arr[idx]) * tau * (M[idx] + mem_terms[idx])
        )
        allocations[grp] = max(a, 0.0)  # ensure non‑negative
        # store history
        state.history[idx].append(allocations[grp])

    # Optional: reward the bandit – simple heuristic: higher total allocation → success
    reward = sum(allocations.values()) / len(state.groups)
    state.bandit.update(reward)

    return allocations


def summarize_state(state: HybridState) -> Dict[str, float]:
    """
    Produce a quick diagnostic summary:
    * average allocation per group,
    * current pheromone distribution,
    * bandit propensity & confidence.
    """
    avg_alloc = np.array([np.mean(hist) if hist else 0.0 for hist in state.history])
    summary = {
        "avg_allocation_codex": avg_alloc[0],
        "avg_allocation_groq": avg_alloc[1],
        "avg_allocation_cohere": avg_alloc[2],
        "avg_allocation_local_models": avg_alloc[3],
        "pheromone_codex": state.pheromone[0],
        "pheromone_groq": state.pheromone[1],
        "pheromone_cohere": state.pheromone[2],
        "pheromone_local_models": state.pheromone[3],
        "bandit_propensity": state.bandit.propensity,
        "bandit_confidence": state.bandit.confidence_bound,
    }
    return summary


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise system
    hy_state = init_hybrid_system()

    # Dummy daily texts (could be logs, emails, etc.)
    daily_texts = [
        "The evidence was verified and the source hash is 123abc. Please plan the next steps.",
        "We need to pause the rollout, wait for the audit, and maybe schedule a meeting tomorrow.",
        "Support from the team is ready. The documentation is complete and the checklist is clear.",
        "Boundary conditions have been set. No further interaction is allowed until review.",
    ]

    for day, txt in enumerate(daily_texts, 1):
        alloc = hybrid_allocate(hy_state, txt)
        print(f"Day {day} allocation:")
        for grp, val in alloc.items():
            print(f"  {grp:15s}: {val:.6f}")
        print("-" * 40)

    # Final diagnostic
    summary = summarize_state(hy_state)
    print("\nFinal Summary:")
    for k, v in summary.items():
        print(f"{k:30s}: {v:.6f}")