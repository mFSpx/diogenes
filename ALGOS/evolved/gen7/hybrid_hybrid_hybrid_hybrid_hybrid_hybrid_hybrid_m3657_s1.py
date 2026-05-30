# DARWIN HAMMER — match 3657, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2507_s0.py (gen6)
# born: 2026-05-29T23:51:05Z

"""Hybrid Algorithm: Stylometry‑Tropical Bayesian Fusion

Parents:
- hybrid_hybrid_hard_truth_ma_kan_m27_s4.py (stylometry features, Bayesian tree cost)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2507_s0.py (tropical max‑plus network, morphology & VRAM budgeting)

Mathematical Bridge:
The categorical stylometry vector **s** ∈ ℝⁿ (n = number of function categories) is fed into a
tropical max‑plus network **T**.  The network computes a piece‑wise linear map

    ℓ_i = max(0, w_i·s + b_i) ,   i = 1…m

producing a likelihood‑like vector **ℓ**.  A Bayesian update combines a prior
distribution **π** over the same n categories with the normalized likelihood
to obtain a posterior **p**:

    p = normalize(π ⊙ ℓ)          (⊙ element‑wise product)

The posterior is then interpreted as a probabilistic cost that is merged with
resource constraints (VRAM of a model tier) and a morphology‑derived health
score.  The final hybrid score guides endpoint selection under stylometry‑aware
VRAM budgeting.

"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from collections import Counter

# ----------------------------------------------------------------------
# Parent‑A components (stylometry)
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

def words(text: str) -> List[str]:
    """Tokenise text into lower‑case alphabetic words."""
    return [w for w in (text or "").lower().split() if w.isalpha()]

def stylometry_vector(text: str) -> np.ndarray:
    """
    Return a normalized frequency vector over FUNCTION_CATS.
    Order of categories follows sorted(FUNCTION_CATS.keys()).
    """
    ws = words(text)
    total = max(1, len(ws))
    cat_counts = []
    for cat in sorted(FUNCTION_CATS.keys()):
        vocab = FUNCTION_CATS[cat]
        cnt = sum(1 for w in ws if w in vocab)
        cat_counts.append(cnt / total)
    return np.array(cat_counts, dtype=float)

# ----------------------------------------------------------------------
# Parent‑B components (tropical network, morphology, VRAM)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

@dataclass(frozen=True)
class ModelTier:
    """Lightweight descriptor for a model."""
    name: str
    ram_mb: int
    tier: str

class TropicalNetwork:
    """
    Max‑plus (tropical) feed‑forward layer.
    For each output dimension i:
        out_i = max(0, w_i·x + b_i)
    """
    def __init__(self, weights: np.ndarray, biases: np.ndarray):
        if weights.shape[0] != biases.shape[0]:
            raise ValueError("weights rows must match biases length")
        self.weights = weights.astype(float)
        self.biases = biases.astype(float)

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """Compute tropical activation."""
        out = np.maximum(0.0, self.weights @ input_vector + self.biases)
        return out

# ----------------------------------------------------------------------
# Hybrid core functions (bridge)
# ----------------------------------------------------------------------
def bayesian_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    Perform element‑wise Bayesian update and normalise.
    prior and likelihood must be 1‑D arrays of same length.
    """
    if prior.shape != likelihood.shape:
        raise ValueError("prior and likelihood must have identical shapes")
    unnorm = prior * likelihood
    total = unnorm.sum()
    if total == 0:
        # avoid division by zero – fall back to uniform distribution
        return np.full_like(prior, 1.0 / prior.size)
    return unnorm / total

def morphology_health(morph: Morphology) -> float:
    """
    Simple health metric: volume divided by (mass+1).
    Higher is better; bounded below by 0.
    """
    volume = morph.length * morph.width * morph.height
    return volume / (morph.mass + 1.0)

def hybrid_score(
    text: str,
    endpoint: EngineEndpoint,
    model: ModelTier,
    network: TropicalNetwork,
    prior: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute a unified score for a given text, endpoint and model.
    Returns a dictionary with:
        - posterior_cost   : Bayesian‑derived cost (lower is better)
        - vram_penalty     : RAM usage weighted by posterior
        - health_factor    : morphology health (higher better)
        - final_score      : weighted combination (lower better)
    """
    # 1. Stylometry vector (n categories)
    s_vec = stylometry_vector(text)

    # 2. Tropical network yields a likelihood-like vector (m dimensions)
    #    Align dimensions: if m != n, project by truncation or padding.
    ell_raw = network.evaluate(s_vec)
    if ell_raw.size != s_vec.size:
        # Simple alignment: truncate larger, pad smaller with zeros
        min_len = min(ell_raw.size, s_vec.size)
        ell = ell_raw[:min_len]
        prior_vec = s_vec[:min_len] if prior is None else prior[:min_len]
    else:
        ell = ell_raw
        prior_vec = s_vec if prior is None else prior

    # 3. Bayesian update (posterior over the aligned categories)
    posterior = bayesian_update(prior_vec, ell)

    # 4. VRAM penalty: expected RAM consumption under posterior weighting
    vram_penalty = float(posterior.sum() * model.ram_mb)

    # 5. Morphology health factor (scaled to [0, 1] via sigmoid)
    raw_health = morphology_health(endpoint.morphology)
    health_factor = 1.0 / (1.0 + math.exp(-raw_health + 5))  # shift to keep values moderate

    # 6. Combine into final score (lower is more desirable)
    #    We weight VRAM heavily, then penalise high posterior cost,
    #    and reward health.
    final_score = (vram_penalty * 0.6) + (posterior.mean() * 30.0) - (health_factor * 10.0)

    return {
        "posterior_cost": float(posterior.mean()),
        "vram_penalty": vram_penalty,
        "health_factor": health_factor,
        "final_score": final_score,
    }

def select_best_endpoint(
    text: str,
    endpoints: List[EngineEndpoint],
    model: ModelTier,
    network: TropicalNetwork,
    prior: np.ndarray | None = None,
) -> EngineEndpoint:
    """
    Evaluate all endpoints with `hybrid_score` and return the one with the
    lowest final_score.
    """
    best = None
    best_score = math.inf
    for ep in endpoints:
        score_dict = hybrid_score(text, ep, model, network, prior)
        if score_dict["final_score"] < best_score:
            best_score = score_dict["final_score"]
            best = ep
    if best is None:
        raise RuntimeError("No endpoint could be selected")
    return best

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text
    sample_text = (
        "The quick brown fox jumps over the lazy dog while the dog "
        "does not notice the fox. It is a simple sentence with common words."
    )

    # Create a tiny tropical network (input dim = number of categories)
    n_cats = len(FUNCTION_CATS)
    rng = np.random.default_rng(42)
    weights = rng.normal(size=(n_cats, n_cats))
    biases = rng.normal(size=n_cats)
    net = TropicalNetwork(weights, biases)

    # Uniform prior over categories
    uniform_prior = np.full(n_cats, 1.0 / n_cats)

    # Define a model tier
    model = ModelTier(name="tiny-gpt", ram_mb=256, tier="small")

    # Define two dummy endpoints with different morphologies
    ep1 = EngineEndpoint(
        engine_id="e1",
        channel="alpha",
        residency="edge",
        runtime="fast",
        resource_class="standard",
        always_on=True,
        endpoint="http://edge-1.local",
        capabilities=["nlp", "generation"],
        morphology=Morphology(length=10.0, width=5.0, height=2.0, mass=3.0),
    )
    ep2 = EngineEndpoint(
        engine_id="e2",
        channel="beta",
        residency="cloud",
        runtime="slow",
        resource_class="high_mem",
        always_on=False,
        endpoint="http://cloud-2.remote",
        capabilities=["nlp", "analysis"],
        morphology=Morphology(length=8.0, width=6.0, height=3.0, mass=10.0),
    )

    # Evaluate scores individually
    for ep in (ep1, ep2):
        res = hybrid_score(sample_text, ep, model, net, uniform_prior)
        print(f"Endpoint {ep.engine_id} scores: {res}")

    # Choose best endpoint
    best_ep = select_best_endpoint(sample_text, [ep1, ep2], model, net, uniform_prior)
    print(f"\nSelected best endpoint: {best_ep.engine_id} ({best_ep.endpoint})")