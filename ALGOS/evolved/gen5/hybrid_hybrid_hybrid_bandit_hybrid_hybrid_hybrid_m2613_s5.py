# DARWIN HAMMER — match 2613, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_bandit_router_hybrid_privacy_sketc_m275_s2.py (gen2)
# parent_b: hybrid_hybrid_hybrid_endpoi_hybrid_hybrid_regret_m62_s0.py (gen4)
# born: 2026-05-29T23:43:09Z

"""Hybrid Bandit‑Sketch‑Regret Engine
===================================

This module fuses the mathematics of the two parent algorithms:

* **Hybrid Bandit‑Sketch Privacy Store** – a multi‑armed bandit whose reward is
  derived from a privacy‑risk score computed from a Count‑Min Sketch (CMS).
* **Hybrid Endpoint Regret with MinHash similarity** – a decision‑hygiene engine
  that extracts a binary feature vector from textual cues, builds a MinHash
  signature, and weights the expected reward by a regret‑adjusted factor.

**Mathematical bridge**

For each candidate *action* we maintain:

1. A private CMS `S_a` that records quasi‑identifiers observed when the action
   is taken.
2. A binary feature vector `f_a` (size 9) obtained by matching the context
   against the nine regexes defined in the endpoint‑regret parent.

From the global sketch `S_g` (the union of all CMS cells) we derive a set
`U_g` of *active* cells (non‑zero counters).  
From `f_a` we derive a set `U_a` of active feature indices.

A MinHash signature `h(·)` with `P` permutations yields an estimate of the
Jaccard similarity  


J_a = |U_a ∩ U_g| / |U_a ∪ U_g|   ≈   similarity = 1/P Σ_i 1[ h_i(U_a) == h_i(U_g) ]


The **privacy‑risk score** of an action is


risk_a = reconstruction_risk(S_a) = 1 - (unique_quasi_identifiers / depth)


where `unique_quasi_identifiers` is the number of non‑zero rows in `S_a`
(divided by the sketch depth).

The **regret‑adjusted weight** is


w_a = 1 / (1 + R_a)          # R_a = cumulative regret for action a


Finally the bandit’s *effective reward* used for selection is


reward_a = (1 - risk_a) * similarity * w_a


The algorithm therefore prefers actions that (i) reveal few distinct
identifiers (low privacy risk), (ii) are semantically aligned with the current
context (high MinHash similarity), and (iii) have incurred little regret so
far.

The three public functions below demonstrate the hybrid operation:
`select_action`, `update_sketch_and_policy`, and `update_privacy_budget`.
"""

import math
import random
import sys
import pathlib
import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterable

import numpy as np

# ----------------------------------------------------------------------
# Feature extraction (parent B)
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

_REGEXES = [
    _EVIDENCE_RE,
    _PLANNING_RE,
    _DELAY_RE,
    _SUPPORT_RE,
    _BOUNDARY_RE,
    _OUTCOME_RE,
    re.compile(r"\b(?:impulsive|rash|hasty)\b", re.I),   # impulsive placeholder
    re.compile(r"\b(?:scarcity|limited|rare)\b", re.I),   # scarcity placeholder
    re.compile(r"\b(?:risk|danger|threat)\b", re.I),      # risk placeholder
]

def extract_feature_vector(text: str) -> np.ndarray:
    """Return a binary (0/1) vector indicating presence of each regex."""
    text = text.lower()
    vec = np.zeros(len(_FEATURE_ORDER), dtype=np.int8)
    for i, rgx in enumerate(_REGEXES):
        if rgx.search(text):
            vec[i] = 1
    return vec

# ----------------------------------------------------------------------
# MinHash similarity (bridge)
# ----------------------------------------------------------------------
class MinHash:
    """Simple MinHash with a fixed number of random hash functions."""
    def __init__(self, num_perm: int = 64):
        self.num_perm = num_perm
        # random seeds for hash functions
        self._seeds = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]

    def _hash(self, x: int, seed: int) -> int:
        # combine integer x with seed using a stable hash
        return int(hashlib.sha256(f"{x}_{seed}".encode()).hexdigest(), 16)

    def signature(self, indices: Iterable[int]) -> List[int]:
        """Compute MinHash signature for a set of integer indices."""
        sig = [sys.maxsize] * self.num_perm
        for idx in indices:
            for i, seed in enumerate(self._seeds):
                h = self._hash(idx, seed)
                if h < sig[i]:
                    sig[i] = h
        return sig

    @staticmethod
    def estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from two signatures."""
        assert len(sig1) == len(sig2)
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

# ----------------------------------------------------------------------
# Count‑Min Sketch (parent A)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Two‑dimensional CMS with depth d and width w."""
    def __init__(self, depth: int = 5, width: int = 1000, seed: int = 0):
        self.depth = depth
        self.width = width
        self.tables = np.zeros((depth, width), dtype=np.int64)
        random.seed(seed)
        self._hash_seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: bytes, i: int) -> int:
        h = hashlib.blake2b(item, digest_size=8, person=bytes([i]))
        return int.from_bytes(h.digest(), "little") % self.width

    def add(self, item: str, count: int = 1) -> None:
        b = item.encode()
        for i, seed in enumerate(self._hash_seeds):
            idx = self._hash(b, i)
            self.tables[i, idx] += count

    def estimate(self, item: str) -> int:
        b = item.encode()
        estimates = []
        for i, seed in enumerate(self._hash_seeds):
            idx = self._hash(b, i)
            estimates.append(self.tables[i, idx])
        return min(estimates)

    def nonzero_rows(self) -> int:
        """Number of rows that contain at least one non‑zero cell."""
        return int(np.count_nonzero(np.any(self.tables > 0, axis=1)))

    def merge(self, other: "CountMinSketch") -> None:
        assert self.depth == other.depth and self.width == other.width
        self.tables += other.tables

    def active_indices(self) -> set:
        """Return a set of (row, col) tuples where the count is > 0."""
        rows, cols = np.nonzero(self.tables)
        return { (int(r), int(c)) for r, c in zip(rows, cols) }

# ----------------------------------------------------------------------
# Bandit structures (parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    action_id: str
    description: str = ""

@dataclass
class PolicyEntry:
    cumulative_reward: float = 0.0
    count: int = 0
    cumulative_regret: float = 0.0

# Global containers ----------------------------------------------------
_POLICY: Dict[str, PolicyEntry] = {}
_SKETCHES: Dict[str, CountMinSketch] = {}
_GLOBAL_SKETCH = CountMinSketch(depth=5, width=2000, seed=42)
_MINHASH = MinHash(num_perm=64)

_PRIVACY_BUDGET: float = 1.0   # starts with full budget (arbitrary unit)

# ----------------------------------------------------------------------
# Core hybrid mathematics
# ----------------------------------------------------------------------
def _reconstruction_risk(skm: CountMinSketch, total_records: int) -> float:
    """Risk = 1 - (unique_quasi_identifiers / depth)."""
    unique = skm.nonzero_rows()
    return 1.0 - (unique / skm.depth)

def _minhash_similarity(action_vec: np.ndarray) -> float:
    """Similarity between action feature set and global sketch feature set."""
    # Action feature set = indices where vector == 1
    action_set = {int(i) for i, v in enumerate(action_vec) if v}
    # Global sketch feature set = active row indices (ignore column)
    sketch_set = {row for row, _ in _GLOBAL_SKETCH.active_indices()}
    sig_a = _MINHASH.signature(action_set)
    sig_g = _MINHASH.signature(sketch_set)
    return MinHash.estimate_jaccard(sig_a, sig_g)

def _combined_reward(action_id: str, context: str) -> float:
    """Compute the hybrid reward for a given action and context."""
    # 1. privacy component
    sk = _SKETCHES.get(action_id, _GLOBAL_SKETCH)
    total_records = sk.tables.sum()
    privacy_score = 1.0 - _reconstruction_risk(sk, total_records)  # higher is better

    # 2. similarity component
    feat_vec = extract_feature_vector(context)
    similarity = _minhash_similarity(feat_vec)

    # 3. regret component
    entry = _POLICY.get(action_id, PolicyEntry())
    regret_weight = 1.0 / (1.0 + entry.cumulative_regret)

    # Combine multiplicatively (any zero will zero out the reward)
    return privacy_score * similarity * regret_weight

def _ucb_score(action_id: str, context: str, total_selections: int) -> float:
    """Upper Confidence Bound using the hybrid reward."""
    entry = _POLICY.get(action_id, PolicyEntry())
    if entry.count == 0:
        # force exploration
        return float('inf')
    avg_reward = entry.cumulative_reward / entry.count
    confidence = math.sqrt(2 * math.log(total_selections + 1) / entry.count)
    # blend UCB with hybrid reward (treated as a bias term)
    hybrid = _combined_reward(action_id, context)
    return avg_reward + confidence + hybrid

# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def select_action(context: str, candidates: List[Action]) -> Action:
    """Select an action using UCB enriched with privacy‑risk, MinHash similarity
    and regret weighting."""
    total_selections = sum(_POLICY.get(a.action_id, PolicyEntry()).count for a in candidates) + 1
    scores = {a.action_id: _ucb_score(a.action_id, context, total_selections) for a in candidates}
    best_id = max(scores, key=scores.get)
    # Return the full Action object
    for a in candidates:
        if a.action_id == best_id:
            return a
    # Fallback (should never happen)
    return candidates[0]

def update_sketch_and_policy(action: Action, context: str, reward: float) -> None:
    """Update the CMS for the chosen action, the global sketch, and the bandit
    policy with the observed reward."""
    # Update sketches
    sk = _SKETCHES.setdefault(action.action_id, CountMinSketch(depth=5, width=1000, seed=hash(action.action_id) & 0xffffffff))
    tokens = re.findall(r"\w+", context.lower())
    for t in tokens:
        sk.add(t, 1)
        _GLOBAL_SKETCH.add(t, 1)

    # Update policy entry
    entry = _POLICY.setdefault(action.action_id, PolicyEntry())
    entry.cumulative_reward += reward
    entry.count += 1

    # Regret update: regret = (max possible reward - actual reward)
    # For simplicity we treat max possible reward as 1.0
    regret = max(0.0, 1.0 - reward)
    entry.cumulative_regret += regret

def update_privacy_budget(inflow: float, outflow: float) -> None:
    """Adjust the global privacy budget pool."""
    global _PRIVACY_BUDGET
    _PRIVACY_BUDGET = max(0.0, _PRIVACY_BUDGET + inflow - outflow)

# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        Action(action_id="A1", description="Send verification email"),
        Action(action_id="A2", description="Log user activity"),
        Action(action_id="A3", description="Store audit record"),
    ]

    # Simulated contexts
    contexts = [
        "Please verify the receipt and attach the screenshot.",
        "We need to plan the rollout and schedule the test.",
        "User requested support, call the doctor and log the outcome.",
    ]

    # Run a few selection‑update cycles
    for i in range(10):
        ctx = random.choice(contexts)
        chosen = select_action(ctx, actions)
        # Simulated reward: higher when context contains "verify" or "audit"
        base = 0.6 if "verify" in ctx.lower() or "audit" in ctx.lower() else 0.3
        noise = random.uniform(-0.05, 0.05)
        reward = max(0.0, min(1.0, base + noise))
        update_sketch_and_policy(chosen, ctx, reward)
        update_privacy_budget(inflow=0.01, outflow=reward * 0.005)

    # Print final policy statistics
    print("Final policy entries:")
    for aid, entry in _POLICY.items():
        print(f"{aid}: reward={entry.cumulative_reward:.3f}, count={entry.count}, regret={entry.cumulative_regret:.3f}")
    print(f"Remaining privacy budget: {_PRIVACY_BUDGET:.4f}")