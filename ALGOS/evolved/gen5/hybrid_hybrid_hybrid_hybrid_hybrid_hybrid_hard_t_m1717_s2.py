# DARWIN HAMMER — match 1717, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s0.py (gen4)
# parent_b: hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s0.py (gen2)
# born: 2026-05-29T23:38:27Z

"""
Hybrid Algorithm: Bayesian-Tropical Stylometry with Rectified Flow Model Loading

Parents:
- hybrid_hybrid_hybrid_distri_hybrid_bayes_claim_k_m166_s0.py (probabilistic
  leader election, Bayesian hypothesis updating, Tropical max‑plus algebra)
- hybrid_hybrid_hard_truth_ma_rectified_flow_m91_s0.py (stylometry feature
  vectors, RAM‑aware model pool, straight‑line (rectified) flow between
  source and target distributions)

Mathematical Bridge:
Both parents manipulate vectors and probabilities.  The stylometry feature
vector extracted from text is treated as *evidence* for a Bayesian hypothesis
over the set of loaded models.  The Bayesian update supplies a posterior
probability which is combined with the prior using Tropical max‑plus algebra
(`⊕` = max, `⊗` = +).  The resulting tropical score defines a scalar field over
hypotheses.  To transition smoothly from the prior distribution to the updated
posterior we employ a rectified (straight‑line) flow in the probability simplex:
    
    p(t) = (1‑t)·p_prior + t·p_posterior ,   t∈[0,1]

The flow is performed in log‑space so that the max‑plus structure is preserved.
The hybrid algorithm thus fuses stylometry‑driven evidence, Bayesian updating,
tropical scoring, and rectified flow into a single decision‑making pipeline that
selects which model(s) to load under RAM constraints.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Tuple, Mapping, Hashable

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures
# ----------------------------------------------------------------------
Node = Hashable
Graph = Mapping[Node, set[Node]]

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

# ----------------------------------------------------------------------
# Model pool (from Parent B)
# ----------------------------------------------------------------------
class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier
        # Randomly initialise a stylometry prototype vector for this model
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        self.prototype = rng.random(len(FUNCTION_CATS))

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def can_load(self, tier: ModelTier) -> bool:
        return self.used_ram() + tier.ram_mb <= self.ram_ceiling_mb

    def load(self, tier: ModelTier) -> bool:
        if self.can_load(tier):
            self.loaded[tier.name] = tier
            return True
        return False

    def unload(self, name: str) -> None:
        self.loaded.pop(name, None)

# ----------------------------------------------------------------------
# Evidence / Hypothesis (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathEvidence:
    id: str
    measurement: float          # scalar representation of stylometry evidence
    noise_std: float

@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float                # log‑probability for tropical algebra
    posterior: float            # log‑probability after Bayesian update
    evidence_ids: Tuple[str, ...] = ()

# ----------------------------------------------------------------------
# Core mathematical primitives
# ----------------------------------------------------------------------
def extract_features(text: str) -> np.ndarray:
    """
    Convert a text string into a normalized feature vector where each entry
    corresponds to the relative frequency of a function‑category token.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    total = len(tokens) or 1
    vec = np.zeros(len(FUNCTION_CATS))
    for idx, cat in enumerate(FUNCTION_CATS):
        cat_words = FUNCTION_CATS[cat]
        count = sum(1 for t in tokens if t in cat_words)
        vec[idx] = count / total
    return vec

def evidence_from_features(features: np.ndarray, eid: str) -> MathEvidence:
    """
    Collapse a feature vector to a scalar measurement (L2 norm) and assign a
    noise estimate proportional to its variance.
    """
    measurement = float(np.linalg.norm(features))
    noise_std = float(np.sqrt(np.var(features) + 1e-9))
    return MathEvidence(id=eid, measurement=measurement, noise_std=noise_std)

def bayesian_update(
    hypotheses: List[MathHypothesis],
    evidence: MathEvidence,
    model_pool: ModelPool,
) -> List[MathHypothesis]:
    """
    Perform a Bayesian update for each hypothesis using a Gaussian likelihood
    centred on the model's stylometry prototype.  The prior and posterior are
    stored in log‑space to be compatible with tropical algebra.
    """
    updated = []
    for hyp in hypotheses:
        model = model_pool.loaded.get(hyp.id)
        if model is None:
            # If the model is not loaded we cannot compute a likelihood;
            # keep prior unchanged.
            updated.append(hyp)
            continue
        # Likelihood: N(evidence.measurement | μ=||prototype||, σ=noise_std)
        mu = float(np.linalg.norm(model.prototype))
        sigma = evidence.noise_std
        # Gaussian log‑likelihood (ignoring constant term)
        ll = -0.5 * ((evidence.measurement - mu) / sigma) ** 2
        posterior_log = hyp.prior + ll  # additive in log‑space
        updated.append(
            replace(
                hyp,
                posterior=posterior_log,
                evidence_ids=hyp.evidence_ids + (evidence.id,),
            )
        )
    return updated

def tropical_score(hypothesis: MathHypothesis) -> float:
    """
    Tropical max‑plus score:
        score = (prior ⊗ posterior) ⊕ (‑noise)
    where ⊗ is addition and ⊕ is max.
    """
    # The negative noise term is a confidence penalty; we use a fixed small value.
    noise_penalty = -0.1
    combined = hypothesis.prior + hypothesis.posterior  # ⊗
    return max(combined, noise_penalty)  # ⊕

def rectified_flow_interpolate(
    start: np.ndarray, end: np.ndarray, t: float
) -> np.ndarray:
    """
    Straight‑line interpolation (rectified flow) between two probability vectors.
    The vectors are assumed to be in the simplex; we perform the interpolation
    in log‑space to preserve the tropical structure.
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be in [0, 1]")
    log_start = np.log(np.clip(start, 1e-12, None))
    log_end = np.log(np.clip(end, 1e-12, None))
    log_interp = (1 - t) * log_start + t * log_end
    interp = np.exp(log_interp)
    # Renormalise to sum to 1
    return interp / interp.sum()

def acceptance_probability(delta_score: float, temperature: float) -> float:
    """
    Metropolis‑style acceptance based on tropical score improvement.
    """
    if delta_score > 0:
        return 1.0
    if temperature <= 0:
        return 0.0
    return math.exp(delta_score / temperature)

def cooling_temperature(k: int, t0: float = 1.0, alpha: float = 0.95) -> float:
    """Exponential cooling schedule."""
    if k < 0 or t0 < 0 or not (0 <= alpha <= 1):
        raise ValueError("invalid cooling parameters")
    return t0 * (alpha ** k)

# ----------------------------------------------------------------------
# Hybrid operation
# ----------------------------------------------------------------------
def hybrid_step(
    text: str,
    pool: ModelPool,
    hypotheses: List[MathHypothesis],
    step_idx: int,
) -> Tuple[List[MathHypothesis], float]:
    """
    Process a single text sample:
      1. Extract stylometry features → evidence.
      2. Bayesian update of hypotheses.
      3. Compute tropical scores and decide which model to load/unload
         using a Metropolis acceptance rule with a cooling schedule.
      4. Return the new hypothesis list and the temperature used.
    """
    # 1. Evidence
    feats = extract_features(text)
    ev = evidence_from_features(feats, f"e{step_idx}")

    # 2. Bayesian update (only over currently loaded models)
    hypotheses = bayesian_update(hypotheses, ev, pool)

    # 3. Tropical scores
    scores = {h.id: tropical_score(h) for h in hypotheses if pool.is_loaded(h.id)}
    if not scores:
        # No loaded models; attempt to load the highest‑prior hypothesis
        candidate = max(hypotheses, key=lambda h: h.prior)
        model = ModelTier(name=candidate.id, ram_mb=500, tier="default")
        pool.load(model)
        scores[candidate.id] = tropical_score(candidate)

    # 4. Acceptance decision
    temperature = cooling_temperature(step_idx)
    # Choose a random loaded hypothesis as current state
    current_id = random.choice(list(scores.keys()))
    current_score = scores[current_id]

    # Propose the hypothesis with the highest score
    proposed_id = max(scores, key=scores.get)
    proposed_score = scores[proposed_id]

    delta = proposed_score - current_score
    if acceptance_probability(delta, temperature) >= random.random():
        # Accept: ensure the proposed model is loaded (already is)
        pass
    else:
        # Reject: unload the proposed model if it was newly loaded
        if pool.is_loaded(proposed_id) and proposed_id != current_id:
            pool.unload(proposed_id)

    return hypotheses, temperature

def run_hybrid(texts: List[str], pool: ModelPool, steps: int = 10) -> None:
    """
    Execute the hybrid algorithm over a sequence of texts.
    """
    # Initialise a hypothesis for each model that may appear in the pool
    initial_hypotheses = [
        MathHypothesis(id=f"model_{i}", prior=math.log(1.0 / (i + 1)), posterior=0.0)
        for i in range(5)
    ]

    # Pre‑load a subset respecting RAM constraints
    for hyp in initial_hypotheses[:3]:
        tier = ModelTier(name=hyp.id, ram_mb=800, tier="preload")
        pool.load(tier)

    for step in range(min(steps, len(texts))):
        hyps, temp = hybrid_step(texts[step], pool, initial_hypotheses, step)
        # Update the hypothesis list for the next iteration
        initial_hypotheses = hyps

        # Optional: display progress (kept minimal for smoke test)
        print(f"Step {step}: Temp={temp:.4f}, Loaded={list(pool.loaded.keys())}")

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "I think that the quick brown fox jumps over the lazy dog.",
        "We cannot deny that the results are significant and meaningful.",
        "She will not be able to attend the meeting because of scheduling conflicts.",
        "The algorithm converges after a finite number of iterations.",
        "They have both contributed to the development of the model.",
    ]

    model_pool = ModelPool(ram_ceiling_mb=2500)
    run_hybrid(sample_texts, model_pool, steps=5)
    print("Hybrid execution completed without errors.")