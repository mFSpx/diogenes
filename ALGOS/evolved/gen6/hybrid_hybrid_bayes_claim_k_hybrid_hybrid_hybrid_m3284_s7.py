# DARWIN HAMMER — match 3284, survivor 7
# gen: 6
# parent_a: hybrid_bayes_claim_kernel_hybrid_ternary_lens__m26_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m560_s3.py (gen5)
# born: 2026-05-29T23:48:59Z

"""Hybrid Bayesian-Pheromone Evidence Fusion

This module combines the core mathematics of two parent algorithms:

* **Parent A (bayes_claim_kernel + pruning schedule)** – updates a hypothesis posterior
  using a Bayesian odds‑ratio update. Evidence is optionally pruned according to a
  decreasing exponential schedule `p(t) = λ·exp(-α·t)`.

* **Parent B (pheromone decay + stylometric features)** – maintains a pheromone entry
  that decays exponentially with a half‑life (`factor = 0.5^{age/half_life}`) and
  extracts stylometric feature vectors from raw text.

The mathematical bridge is the **exponential decay** that appears in both
pruning (`exp(-α·t)`) and pheromone decay (`0.5^{age/half_life} = exp(-ln2·age/half_life)`).
We therefore define a unified `exp_decay(t, rate)` function and let the pheromone
signal modulate the Bayesian likelihood ratio. The hybrid algorithm proceeds as:

1. Each piece of evidence carries a `PheromoneEntry`. Its decayed signal value
   influences the likelihood ratio (`LR = base_lr * pheromone_signal`).
2. At each time step `t` we optionally prune evidence according to the same
   exponential decay rate used for pheromones.
3. Hypotheses are updated with the modified likelihood ratios, preserving the
   original Bayesian odds‑ratio update.

The three public functions below illustrate this workflow:
`exp_decay`, `prune_and_decay_evidence`, `update_hypothesis_with_pheromone`,
and `stylometric_feature_extraction` (used to compute a base likelihood ratio).

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import uuid
import numpy as np
from typing import List, Tuple, Mapping, Any

# ----------------------------------------------------------------------
# Core data structures (from both parents)
# ----------------------------------------------------------------------


class MathClaim:
    def __init__(self, id: str):
        self.id = id


class MathEvidence:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text  # raw text used for stylometric features


class MathHypothesis:
    def __init__(self, id: str, prior: float, posterior: float, evidence_ids: List[str]):
        self.id = id
        self.prior = prior
        self.posterior = posterior
        self.evidence_ids = evidence_ids


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    label: str
    score: float
    backend: str = "literal_fallback"


class PheromoneEntry:
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid4())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay factor based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        # 0.5 ** (age / half_life) == exp(-ln(2) * age / half_life)
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


# ----------------------------------------------------------------------
# Unified exponential decay (mathematical bridge)
# ----------------------------------------------------------------------


def exp_decay(t: float, rate: float) -> float:
    """
    General exponential decay function.
    For Parent A: rate = α  (pruning schedule λ·exp(-α·t) with λ handled externally)
    For Parent B: rate = ln(2)/half_life (decay factor = exp(-rate·t))
    """
    if t < 0 or rate < 0:
        raise ValueError("t and rate must be non‑negative")
    return math.exp(-rate * t)


# ----------------------------------------------------------------------
# Stylometric feature extraction (complete version)
# ----------------------------------------------------------------------


def stylometric_feature_extraction(texts: List[str]) -> np.ndarray:
    """
    Simple count‑based stylometric vector.
    Returns an (n_texts, n_categories) array where each entry is the count of
    category‑specific function words present in the corresponding text.
    """
    FUNCTION_CATS = {
        "pronoun": {
            "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself",
            "he", "him", "his", "she", "her", "hers", "they", "them", "their",
            "theirs", "we", "us", "our", "ours"
        },
        "article": {"a", "an", "the"},
        "preposition": {
            "about", "above", "after", "against", "around", "as", "at", "before",
            "behind", "below", "between", "by", "during", "for", "from", "in",
            "into", "of", "off", "on", "onto", "over", "through", "to", "under",
            "with", "without"
        },
        "auxiliary": {
            "am", "are", "be", "been", "being", "can", "could", "did", "do",
            "does", "had", "has", "have", "is", "may", "might"
        }
    }
    n_cats = len(FUNCTION_CATS)
    vectors = np.zeros((len(texts), n_cats), dtype=float)

    for idx, text in enumerate(texts):
        lowered = text.lower()
        for cat_idx, (cat, word_set) in enumerate(FUNCTION_CATS.items()):
            count = sum(1 for w in word_set if w in lowered)
            vectors[idx, cat_idx] = count

    return vectors


# ----------------------------------------------------------------------
# Evidence pruning combined with pheromone decay
# ----------------------------------------------------------------------


def prune_and_decay_evidence(evidence_list: List[MathEvidence],
                             pheromones: Mapping[str, PheromoneEntry],
                             t: float,
                             lam: float = 1.0,
                             alpha: float = 0.2,
                             seed: int | str | None = None) -> List[MathEvidence]:
    """
    1. Apply exponential decay to each pheromone (rate = ln(2)/half_life).
    2. Compute a pruning probability for each evidence item:
         p_prune = lam * exp(-alpha * t) * (1 - normalized_pheromone_signal)
       Evidence with higher remaining pheromone signal are less likely to be pruned.
    3. Randomly keep evidence according to (1 - p_prune).
    Returns the retained evidence list.
    """
    rng = random.Random(seed)

    # Step 1: decay pheromones in‑place
    for ph in pheromones.values():
        ph.apply_decay()

    # Gather signal values for normalization
    signals = np.array([ph.signal_value for ph in pheromones.values()], dtype=float)
    max_signal = signals.max() if signals.size > 0 else 1.0
    # Avoid division by zero
    norm_signals = signals / max_signal if max_signal > 0 else signals

    # Map evidence id -> normalized signal (default 0 if missing)
    signal_by_eid = {eid: norm for eid, norm in zip(pheromones.keys(), norm_signals)}

    retained: List[MathEvidence] = []
    for ev in evidence_list:
        base_prune = lam * exp_decay(t, alpha)  # λ·exp(-α·t)
        signal_factor = 1.0 - signal_by_eid.get(ev.id, 0.0)  # high signal ⇒ low prune prob
        prune_prob = min(1.0, base_prune * signal_factor)
        if rng.random() > prune_prob:
            retained.append(ev)
    return retained


# ----------------------------------------------------------------------
# Bayesian update that incorporates pheromone‑modulated likelihood ratios
# ----------------------------------------------------------------------


def update_hypothesis_with_pheromone(hypothesis: MathHypothesis,
                                    evidence: MathEvidence,
                                    pheromone: PheromoneEntry,
                                    base_likelihood_ratio: float) -> MathHypothesis:
    """
    Compute a likelihood ratio modulated by the current pheromone signal:
        LR_eff = base_likelihood_ratio * (1 + pheromone.signal_value)
    Then perform the standard Bayesian odds‑ratio update.
    """
    if base_likelihood_ratio < 0:
        raise ValueError("base_likelihood_ratio must be non‑negative")

    # Modulation – ensure the factor stays positive
    mod_factor = 1.0 + max(0.0, pheromone.signal_value)
    likelihood_ratio = base_likelihood_ratio * mod_factor

    # Bayesian odds update (identical to Parent A)
    p = max(0.0, min(1.0, hypothesis.posterior))
    if p <= 0.0 or likelihood_ratio == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * likelihood_ratio
        posterior = new_odds / (1.0 + new_odds)

    posterior = max(0.0, min(1.0, posterior))
    new_evidence_ids = list(hypothesis.evidence_ids) + [evidence.id]
    return MathHypothesis(id=hypothesis.id,
                          prior=hypothesis.posterior,
                          posterior=posterior,
                          evidence_ids=new_evidence_ids)


# ----------------------------------------------------------------------
# High‑level hybrid pipeline (demonstrates three functions together)
# ----------------------------------------------------------------------


def hybrid_pipeline(hypotheses: List[MathHypothesis],
                    evidences: List[MathEvidence],
                    half_life_seconds: int = 60,
                    t: float = 0.0,
                    lam: float = 1.0,
                    alpha: float = 0.2,
                    seed: int | str | None = None) -> List[MathHypothesis]:
    """
    1. Extract stylometric features from evidence texts and derive a base likelihood
       ratio per evidence (simple heuristic: sum of feature counts + 1).
    2. Attach a PheromoneEntry to each evidence item.
    3. Prune evidence using `prune_and_decay_evidence`.
    4. Update each hypothesis with the retained evidence via
       `update_hypothesis_with_pheromone`.
    Returns the list of updated hypotheses.
    """
    # 1. Feature extraction → base LR
    texts = [ev.text for ev in evidences]
    feature_matrix = stylometric_feature_extraction(texts)  # (n_evidence, n_cats)
    base_lrs = feature_matrix.sum(axis=1) + 1.0  # ensure >0

    # 2. Create pheromones (one per evidence)
    pheromones: dict[str, PheromoneEntry] = {}
    for ev in evidences:
        pheromones[ev.id] = PheromoneEntry(
            surface_key=ev.id,
            signal_kind="evidence_signal",
            signal_value=random.random(),  # random initial strength
            half_life_seconds=half_life_seconds,
        )

    # 3. Prune & decay
    retained_evidence = prune_and_decay_evidence(
        evidence_list=evidences,
        pheromones=pheromones,
        t=t,
        lam=lam,
        alpha=alpha,
        seed=seed,
    )

    # Mapping from id to base LR for fast lookup
    lr_by_id = {ev.id: lr for ev, lr in zip(evidences, base_lrs)}

    # 4. Update hypotheses
    updated_hyps: List[MathHypothesis] = []
    for hyp in hypotheses:
        cur_hyp = hyp
        for ev in retained_evidence:
            ph = pheromones[ev.id]
            base_lr = lr_by_id[ev.id]
            cur_hyp = update_hypothesis_with_pheromone(cur_hyp, ev, ph, base_lr)
        updated_hyps.append(cur_hyp)

    return updated_hyps


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # Create a simple hypothesis
    h0 = MathHypothesis(id="H0", prior=0.5, posterior=0.5, evidence_ids=[])

    # Create synthetic evidence
    evidences = [
        MathEvidence(id="E1", text="I think the theorem is true because of the axioms."),
        MathEvidence(id="E2", text="The proof uses induction and relies on previous lemmas."),
        MathEvidence(id="E3", text="Counterexample shows the claim fails in edge cases."),
    ]

    # Run the hybrid pipeline
    updated = hybrid_pipeline(
        hypotheses=[h0],
        evidences=evidences,
        half_life_seconds=30,
        t=5.0,
        lam=0.9,
        alpha=0.1,
        seed=42,
    )

    # Print results
    for hyp in updated:
        print(f"Hypothesis {hyp.id}: posterior={hyp.posterior:.4f}, evidence_used={hyp.evidence_ids}")