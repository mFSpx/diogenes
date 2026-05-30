# DARWIN HAMMER — match 4103, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1496_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_indy_learning_m816_s0.py (gen4)
# born: 2026-05-29T23:53:41Z

"""Hybrid Privacy‑Bandit & Ternary‑Lens Vectorized Learning Engine

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1496_s1.py (DP‑budget bandit with
  hyper‑vector generation driven by a min‑hash signature)
- hybrid_hybrid_hybrid_ternar_hybrid_indy_learning_m816_s0.py (ternary lens
  audit that feeds a vectorized‑learning pipeline)

Mathematical Bridge
------------------
Both parents manipulate sets of integer identifiers derived from textual data.

*Parent A* creates a min‑hash signature **S** ⊂ ℤ, computes its cardinality
|S| and defines a privacy‑risk term  

\[
r = \frac{|S|}{N},
\]

where **N** is the total number of records.  The same signature seeds a
deterministic hyper‑vector  

\[
\mathbf{h} \in \{-1,+1\}^{d}.
\]

*Parent B* audits a manifest and produces a dictionary **F** of findings.
These findings are hashed together with each token to generate stable chunk
identifiers; the collection of tokens is then processed in a vectorized
learning routine.

The fusion treats the audit findings **F** as a *context* that modulates the
utility term of the bandit.  Concretely, for a token set **T** we build a
binary token‑vector **v**∈{-1,+1}^d (derived from hashes of the tokens) and
measure geometric alignment with the hyper‑vector **h** via the dot product

\[
u = \frac{1}{d}\,\mathbf{h}^{\!\top}\mathbf{v}\in[-1,1],
\]

which serves as the base utility.  An audit‑driven bonus  

\[
b = \beta\;\frac{1}{|T|}\sum_{t\in T}\mathbf{1}\bigl(\text{family}(t)\in F\bigr)
\]

is added, yielding the total expected reward for a DP‑budget ε

\[
\hat{R}(\varepsilon)=u + b - \alpha\,r(\varepsilon).
\]

A regret‑weighted bandit policy then selects ε by sampling from a
soft‑max distribution over the cumulative average rewards of each ε.  The
resulting system couples privacy risk, geometric similarity of hyper‑vectors,
and audit‑derived semantic weighting in a single unified algorithm.
"""

import math
import random
import sys
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Immutable data structures (shared with Parent A)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BanditAction:
    """Concrete DP‑budget choice produced by the bandit."""
    epsilon: float
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridPrivacyBanditTernaryLens"


@dataclass(frozen=True)
class BanditUpdate:
    """Feedback used to update the bandit policy."""
    context_id: str
    epsilon: float
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Global mutable state – bandit policy & simple KV store (Parent A)
# ----------------------------------------------------------------------


_POLICY: Dict[float, List[float]] = {}   # ε → [total_reward, count]
_STORE: Dict[str, float] = {}           # arbitrary key‑value store
DEFAULT_EPSILONS = [0.1, 0.5, 1.0]
DEFAULT_BUDGET_MB = 1024 * 4
_DIMENSION = 128          # dimension of hyper‑vectors
_ALPHA = 0.5              # privacy‑risk weight
_BETA = 0.3               # audit‑bonus weight


def reset_policy() -> None:
    """Clear the bandit policy and KV store."""
    _POLICY.clear()
    _STORE.clear()


# ----------------------------------------------------------------------
# Helper primitives – min‑hash, hyper‑vector, audit, tokenisation
# ----------------------------------------------------------------------


def _minhash_signature(tokens: List[str], num_hashes: int = 64) -> set[int]:
    """Create a simple min‑hash signature from a list of tokens."""
    signature = set()
    for i in range(num_hashes):
        min_hash = min(
            int(hashlib.sha256(f"{t}|{i}".encode()).hexdigest(), 16)
            for t in tokens
        )
        signature.add(min_hash)
    return signature


def _hash_to_seed(value: Any) -> int:
    """Deterministic integer seed from any JSON‑serialisable value."""
    json_bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(json_bytes).hexdigest(), 16) % (2**32)


def generate_hypervector(seed_value: Any, d: int = _DIMENSION) -> np.ndarray:
    """Deterministic hyper‑vector ∈{-1,+1}^d seeded from *seed_value*."""
    seed = _hash_to_seed(seed_value)
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=d)


def ternary_lens_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    """Extract audit findings from a vendor manifest."""
    findings: dict[str, Any] = {}
    for candidate in manifest.get("vendors", []):
        key = candidate.get("candidate_key", "")
        family = candidate.get("family", "")
        notes = candidate.get("notes", "")
        findings[key] = {"family": family, "notes": notes}
    return findings


def token_vector(tokens: List[str], d: int = _DIMENSION) -> np.ndarray:
    """Map tokens to a binary vector in {-1,+1}^d via deterministic hashing."""
    vec = np.empty(d, dtype=int)
    for i in range(d):
        if i < len(tokens):
            h = int(hashlib.sha256(tokens[i].encode()).hexdigest(), 16)
            vec[i] = 1 if (h % 2) == 0 else -1
        else:
            # pad with deterministic pattern
            vec[i] = 1 if (i % 2) == 0 else -1
    return vec


def _audit_bonus(tokens: List[str], findings: dict[str, Any]) -> float:
    """Proportion of tokens whose string contains any audited family."""
    if not tokens:
        return 0.0
    families = {info["family"] for info in findings.values() if info.get("family")}
    match_count = sum(
        any(fam.lower() in token.lower() for fam in families) for token in tokens
    )
    return _BETA * (match_count / len(tokens))


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------


def compute_privacy_risk(signature: set[int], total_records: int) -> float:
    """Risk term r = |S| / N."""
    if total_records <= 0:
        raise ValueError("total_records must be positive")
    return len(signature) / total_records


def bandit_select_epsilon(
    expected_reward: float,
    context_id: str,
    epsilon_candidates: List[float] = DEFAULT_EPSILONS,
) -> BanditAction:
    """Regret‑weighted selection of a DP budget ε."""
    # Initialise policy entries if missing
    for eps in epsilon_candidates:
        _POLICY.setdefault(eps, [0.0, 0.0])

    # Compute average rewards
    avg_rewards = np.array([
        (_POLICY[eps][0] / max(_POLICY[eps][1], 1.0)) for eps in epsilon_candidates
    ])

    # Soft‑max over averages to obtain probabilities
    # Adding a small constant for numerical stability
    exp_vals = np.exp(avg_rewards - np.max(avg_vals := avg_rewards))
    probs = exp_vals / exp_vals.sum()

    # Sample ε according to the soft‑max distribution
    chosen_eps = random.choices(epsilon_candidates, weights=probs, k=1)[0]
    propensity = probs[epsilon_candidates.index(chosen_eps)]

    # Confidence bound (simple Hoeffding‑type placeholder)
    count = max(_POLICY[chosen_eps][1], 1)
    conf_bound = math.sqrt( (2 * math.log(1 / 0.05)) / count )  # 95% confidence

    return BanditAction(
        epsilon=chosen_eps,
        propensity=propensity,
        expected_reward=expected_reward,
        confidence_bound=conf_bound,
    )


def update_bandit(update: BanditUpdate) -> None:
    """Incorporate observed reward into the policy."""
    if update.epsilon not in _POLICY:
        _POLICY[update.epsilon] = [0.0, 0.0]
    total, cnt = _POLICY[update.epsilon]
    _POLICY[update.epsilon] = [total + update.reward, cnt + 1]


def hybrid_process(
    text: str,
    manifest: dict[str, Any],
    total_records: int,
    d: int = _DIMENSION,
    alpha: float = _ALPHA,
) -> Tuple[BanditAction, float]:
    """
    End‑to‑end hybrid routine:

    1. Audit the manifest → findings.
    2. Tokenise *text* → tokens.
    3. Build min‑hash signature from tokens → S.
    4. Compute privacy risk r.
    5. Generate deterministic hyper‑vector h seeded by findings.
    6. Build token vector v.
    7. Compute geometric utility u = (h·v)/d.
    8. Add audit‑bonus b.
    9. Form expected reward  R̂ = u + b – α·r.
    10. Select ε via regret‑weighted bandit.
    11. Return the chosen action and the raw reward value.
    """
    # 1‑2
    findings = ternary_lens_audit(manifest)
    tokens = re.findall(r"\S+", text)

    # 3‑4
    signature = _minhash_signature(tokens)
    risk = compute_privacy_risk(signature, total_records)

    # 5‑6
    hypervector = generate_hypervector(findings, d)
    token_vec = token_vector(tokens, d)

    # 7‑8
    geometric_utility = float(np.dot(hypervector, token_vec) / d)  # ∈[-1,1]
    audit_bonus = _audit_bonus(tokens, findings)

    # 9
    expected_reward = geometric_utility + audit_bonus - alpha * risk

    # 10
    context_id = hashlib.sha256(text.encode()).hexdigest()
    action = bandit_select_epsilon(expected_reward, context_id)

    # 11 – immediate feedback (simulated as the expected reward itself)
    update = BanditUpdate(
        context_id=context_id,
        epsilon=action.epsilon,
        reward=expected_reward,
        propensity=action.propensity,
    )
    update_bandit(update)

    return action, expected_reward


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Minimal manifest example
    sample_manifest = {
        "vendors": [
            {"candidate_key": "v1", "family": "Alpha", "notes": "first"},
            {"candidate_key": "v2", "family": "Beta", "notes": "second"},
        ]
    }

    sample_text = (
        "Alpha systems provide robust performance, while Beta solutions excel in flexibility."
    )

    # Assume we are operating over a dataset of 10 000 records
    action, reward = hybrid_process(
        text=sample_text,
        manifest=sample_manifest,
        total_records=10_000,
    )

    print("Chosen DP budget (ε):", action.epsilon)
    print("Propensity:", round(action.propensity, 4))
    print("Expected reward:", round(action.expected_reward, 4))
    print("Confidence bound:", round(action.confidence_bound, 4))
    print("Policy snapshot:", {k: asdict(v) for k, v in _POLICY.items()})