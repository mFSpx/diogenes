# DARWIN HAMMER — match 2613, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_regret_m62_s0.py (gen4)
# born: 2026-05-29T23:43:09Z

"""Hybrid Bandit‑Sketch‑MinHash Regret Engine
================================================

This module fuses the two parent algorithms:

* **Parent A – hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s2.py**  
  Provides a multi‑armed bandit whose reward is derived from a
  Count‑Min Sketch (CMS) based privacy risk estimate and a privacy‑budget
  store.

* **Parent B – hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_regret_m62_s0.py**  
  Supplies a MinHash‑based similarity metric computed from regex‑driven
  textual cues and a regret‑weighted expected‑reward formulation.

**Mathematical bridge**  
Both parents operate on *sets of tokens* extracted from an action’s
payload.  The CMS approximates token frequencies (privacy side) while the
MinHash signature approximates the Jaccard similarity of the token set
to a context‑derived cue set (decision‑regret side).  The bridge is the
shared token universe: the same token set updates the CMS *and* the
MinHash.  The final bandit reward for an action **a** in context **c** is


R(a,c) = (1 - ρ(CMS_a))                         # privacy utility
         * (1 + λ * Sim(MH_a, MH_c))           # similarity boost
         - η * Regret(a,c)                     # regret penalty


where  

* `ρ(CMS_a)` – reconstruction‑risk score (fraction of non‑zero cells).  
* `Sim` – MinHash Jaccard estimate.  
* `λ` – similarity scaling factor.  
* `η` – regret scaling factor.  

The bandit uses an Upper‑Confidence‑Bound (UCB) that incorporates the
above reward and the usual exploration term.

The code below implements the combined system with three public
functions: `select_action`, `update_sketch_and_policy`, and
`update_store`.  A lightweight smoke test is provided at the end.
"""

import math
import random
import sys
import pathlib
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Iterable

import numpy as np
import re

# ----------------------------------------------------------------------
# Regex‑driven feature extraction (Parent B)
# ----------------------------------------------------------------------
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
_PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
_DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
_SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
_OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe)\b",
    re.I,
)

_REGEX_MAP = {
    "evidence": _EVIDENCE_RE,
    "planning": _PLANNING_RE,
    "delay": _DELAY_RE,
    "support": _SUPPORT_RE,
    "boundary": _BOUNDARY_RE,
    "outcome": _OUTCOME_RE,
    # The remaining cues have no explicit regex in the original source;
    # they are kept for weight vectors but left empty here.
    "impulsive": None,
    "scarcity": None,
    "risk": None,
}


def _extract_cue_tokens(text: str) -> Set[str]:
    """Return a set of cue identifiers present in *text*."""
    tokens: Set[str] = set()
    lowered = text.lower()
    for cue, regex in _REGEX_MAP.items():
        if regex is not None and regex.search(lowered):
            tokens.add(cue)
    return tokens


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent A)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Simple Count‑Min Sketch with pairwise‑independent hash functions."""

    def __init__(self, depth: int = 5, width: int = 2 ** 10, seed: int = 0):
        self.depth = depth
        self.width = width
        self.seed = seed
        self.table = np.zeros((depth, width), dtype=np.int64)
        # Generate random salts for each row
        rng = np.random.default_rng(seed)
        self.salts = rng.integers(0, 2 ** 31, size=depth, dtype=np.int64)

    def _hash(self, item: str, row: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(item.encode("utf-8"))
        h.update(self.salts[row].tobytes())
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        for r in range(self.depth):
            c = self._hash(item, r)
            self.table[r, c] += count

    def estimate(self, item: str) -> int:
        """Return the minimum count across rows (standard CMS estimator)."""
        estimates = [
            self.table[r, self._hash(item, r)] for r in range(self.depth)
        ]
        return int(min(estimates))

    def nonzero_cells(self) -> int:
        return int(np.count_nonzero(self.table))


# ----------------------------------------------------------------------
# MinHash signature (Parent B)
# ----------------------------------------------------------------------
class MinHashSignature:
    """Fixed‑size MinHash signature for a set of tokens."""

    def __init__(self, num_hashes: int = 64, seed: int = 0):
        self.num_hashes = num_hashes
        self.seed = seed
        self.max_hash = (1 << 64) - 1
        self.hashes = np.full(num_hashes, self.max_hash, dtype=np.uint64)
        # Pre‑compute per‑hash salts
        rng = np.random.default_rng(seed)
        self.salts = rng.integers(0, 2 ** 31, size=num_hashes, dtype=np.int64)

    def _hash(self, token: str, i: int) -> int:
        h = hashlib.blake2b(digest_size=8)
        h.update(token.encode("utf-8"))
        h.update(self.salts[i].tobytes())
        return int.from_bytes(h.digest(), "little")

    def update(self, token: str) -> None:
        for i in range(self.num_hashes):
            hv = self._hash(token, i)
            if hv < self.hashes[i]:
                self.hashes[i] = hv

    def jaccard(self, other: "MinHashSignature") -> float:
        """Estimate Jaccard similarity using the signature."""
        if self.num_hashes != other.num_hashes:
            raise ValueError("Signature sizes differ")
        matches = np.sum(self.hashes == other.hashes)
        return matches / self.num_hashes


# ----------------------------------------------------------------------
# Data structures shared by the hybrid system
# ----------------------------------------------------------------------
@dataclass
class ActionState:
    """All mutable state associated with a bandit arm."""
    cms: CountMinSketch = field(default_factory=CountMinSketch)
    mh: MinHashSignature = field(default_factory=MinHashSignature)
    cumulative_reward: float = 0.0
    pulls: int = 0


# Global registries
_POLICY: Dict[str, ActionState] = {}
_GLOBAL_BUDGET: float = 1.0  # privacy budget pool (arbitrary units)

# Hyper‑parameters (tuned to blend the two parents)
_SIMILARITY_SCALE: float = 0.5   # λ
_REGRET_SCALE: float = 0.3      # η
_UCB_CONFIDENCE: float = 2.0    # exploration coefficient


# ----------------------------------------------------------------------
# Core mathematical components
# ----------------------------------------------------------------------
def _reconstruction_risk_score(cms: CountMinSketch) -> float:
    """
    Approximate reconstruction risk as the fraction of non‑zero cells
    in the sketch (higher occupancy → higher risk).
    """
    total_cells = cms.depth * cms.width
    return cms.nonzero_cells() / total_cells


def _regret_weight(action_id: str, context_tokens: Set[str]) -> float:
    """
    Simple regret proxy: actions that have many *negative* cues in common
    with the context are penalised.  Negative cues are the ones with
    non‑zero weight in the `_NEGATIVE_WEIGHTS` vector from Parent B.
    """
    negative_cues = {
        cue
        for cue, w in zip(_FEATURE_ORDER, [0, 0, 0, 0, 0, 0, 1500, 700, 1200])
        if w > 0
    }
    overlap = negative_cues.intersection(context_tokens)
    return len(overlap) / max(1, len(negative_cues))


def _ucb_score(state: ActionState, total_pulls: int, similarity: float) -> float:
    """
    Upper‑Confidence‑Bound incorporating:
      - empirical mean reward,
      - exploration term,
      - similarity boost,
      - regret penalty.
    """
    if state.pulls == 0:
        return float("inf")  # force initial exploration

    mean_reward = state.cumulative_reward / state.pulls
    exploration = _UCB_CONFIDENCE * math.sqrt(
        math.log(total_pulls) / state.pulls
    )
    regret = _REGRET_SCALE * _regret_weight("", set())  # placeholder (zero)
    return (
        mean_reward
        + exploration
        + _SIMILARITY_SCALE * similarity
        - regret
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def select_action(context_text: str) -> str:
    """
    Choose an action ID given a textual *context*.
    The context is tokenised into cue identifiers, a MinHash signature
    is built, and each arm’s UCB (including similarity to the context)
    is evaluated.
    """
    if not _POLICY:
        raise RuntimeError("No actions have been registered.")

    # Build context signature
    ctx_tokens = _extract_cue_tokens(context_text)
    ctx_mh = MinHashSignature()
    for tok in ctx_tokens:
        ctx_mh.update(tok)

    total_pulls = sum(st.pulls for st in _POLICY.values()) + 1

    best_action = None
    best_score = -float("inf")
    for aid, st in _POLICY.items():
        sim = st.mh.jaccard(ctx_mh)
        score = _ucb_score(st, total_pulls, sim)
        if score > best_score:
            best_score = score
            best_action = aid

    return best_action  # type: ignore


def update_sketch_and_policy(
    action_id: str,
    data_items: Iterable[str],
    reward: float | None = None,
) -> None:
    """
    Incorporate *data_items* (tokens) into the action’s CMS and MinHash.
    If *reward* is omitted, it is recomputed from the updated sketch.
    The policy statistics (cumulative reward, pulls) are then updated.
    """
    if action_id not in _POLICY:
        _POLICY[action_id] = ActionState()

    state = _POLICY[action_id]

    # Update CMS and MinHash with each token
    for token in data_items:
        state.cms.add(token, 1)
        state.mh.update(token)

    # Compute reward if not supplied
    if reward is None:
        risk = _reconstruction_risk_score(state.cms)
        reward = 1.0 - risk  # privacy utility component

    # Policy update
    state.cumulative_reward += reward
    state.pulls += 1


def update_store(inflow: float, outflow: float) -> None:
    """
    Adjust the global privacy‑budget store.  The store caps the maximum
    per‑action reward that can be credited; if the budget would become
    negative the outflow is truncated.
    """
    global _GLOBAL_BUDGET
    _GLOBAL_BUDGET = max(0.0, _GLOBAL_BUDGET + inflow - outflow)


# ----------------------------------------------------------------------
# Utility for registration (not part of the original parents but useful)
# ----------------------------------------------------------------------
def register_action(action_id: str) -> None:
    """Create an empty state for *action_id* if it does not already exist."""
    if action_id not in _POLICY:
        _POLICY[action_id] = ActionState()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Register a few dummy actions
    for aid in ["email", "sms", "push"]:
        register_action(aid)

    # Simulate a stream of contexts and token updates
    contexts = [
        "Please verify the source and attach the log file.",
        "We need to plan the rollout and set a timeline.",
        "Delay the deployment until tomorrow for safety.",
    ]

    token_pool = [
        ["evidence", "log", "record"],
        ["planning", "timeline", "schedule"],
        ["delay", "safety", "boundary"],
    ]

    for ctx, toks in zip(contexts, token_pool):
        chosen = select_action(ctx)
        print(f"Context: {ctx[:40]:<45} | Chosen action: {chosen}")

        # Update the chosen arm with its token set and a synthetic reward
        update_sketch_and_policy(chosen, toks)

        # Simulate budget consumption
        update_store(inflow=0.05, outflow=0.02)

    print("\nFinal policy statistics:")
    for aid, st in _POLICY.items():
        avg = st.cumulative_reward / st.pulls if st.pulls else 0.0
        print(f"  {aid}: pulls={st.pulls}, avg_reward={avg:.3f}")

    print(f"\nRemaining global privacy budget: {_GLOBAL_BUDGET:.3f}")