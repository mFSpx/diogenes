# DARWIN HAMMER — match 4657, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1467_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2597_s3.py (gen4)
# born: 2026-05-29T23:57:24Z

"""Hybrid Engine Endpoint Allocation with Fractional Memory and Morphology‑Weighted Pheromones

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_privac_m1467_s0 (EngineEndpoint, Morphology, VRAM scheduling)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2597_s3 (Caputo fractional‑memory allocation + pheromone dynamics)

Mathematical Bridge:
The bridge is the *effective time constant* τ_eff that drives the Caputo fractional‑memory term in the
allocation equation.  τ_eff is derived from the morphological volume of an EngineEndpoint and its
VRAM‑budget, linking the morphology‑based “circuit‑breaker” logic of Parent A with the fractional‑memory
allocation of Parent B.  Pheromone signals are released proportionally to the allocated VRAM weighted by
the same morphological factor, feeding back into the allocation for the next step.  Thus the unified
system solves

    a_i(t) = b_i + τ_eff,i·D^α a_i(t) + λ·ϕ_g(t)·w_i

where
- a_i(t)   – allocation for endpoint i at day t,
- b_i      – baseline allocation derived from mass,
- τ_eff,i  – (VRAM / max_VRAM)·(volume_i / max_volume)·τ₀,
- D^α      – Caputo fractional derivative approximated by Grünwald‑Letnikov series,
- ϕ_g(t)   – pheromone level of the group g to which i belongs,
- λ       – coupling constant,
- w_i      – morphology weight = volume_i / mass_i.

The code below implements this hybrid dynamics."""
import math
import random
import sys
import pathlib
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data structures from Parent A
# ---------------------------------------------------------------------------

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

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

@dataclass(frozen=True)
class EngineEndpoint:
    engine_id: str
    channel: str
    residency: str
    runtime: str
    resource_class: str                # e.g. one of GROUPS
    always_on: bool
    endpoint: str
    capabilities: List[str]
    morphology: Morphology
    outbound_state: str = "draft_only"

# Simple model tier to hold VRAM budget (in MB)
class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

# ---------------------------------------------------------------------------
# Gamma function via Lanczos approximation (Parent B)
# ---------------------------------------------------------------------------

LANCZOS_C = np.array([
    0.99999999999980993,
    676.5203681218851,
    -1259.13,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
])

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def lanczos_gamma(z: complex) -> complex:
    """Lanczos approximation of the Gamma function."""
    if z.real < 0.5:
        return math.pi / (math.sin(math.pi * z) * lanczos_gamma(1 - z))
    z = z - 1
    x = LANCZOS_C[0]
    for i in range(1, len(LANCZOS_C)):
        x += LANCZOS_C[i] / (z + i)
    t = z + len(LANCZOS_C) - 0.5
    return math.sqrt(2 * math.pi) * t ** (z + 0.5) * math.exp(-t) * x

def gamma(x: float) -> float:
    """Real‑valued Gamma using the Lanczos approximation."""
    return float(lanczos_gamma(complex(x)).real)

# ---------------------------------------------------------------------------
# Fractional‑memory (Caputo) utilities (Parent B)
# ---------------------------------------------------------------------------

def caputo_coeffs(alpha: float, n_terms: int) -> np.ndarray:
    """
    Compute Grünwald‑Letnikov coefficients for the Caputo derivative of order alpha.
    coeff_k = (-1)^k * C(alpha, k) where C is the generalized binomial coefficient.
    """
    coeffs = np.empty(n_terms, dtype=float)
    for k in range(n_terms):
        # C(alpha, k) = Gamma(alpha+1) / (Gamma(k+1) * Gamma(alpha - k + 1))
        binom = gamma(alpha + 1) / (gamma(k + 1) * gamma(alpha - k + 1))
        coeffs[k] = ((-1) ** k) * binom
    return coeffs

# ---------------------------------------------------------------------------
# Pheromone system (Parent B)
# ---------------------------------------------------------------------------

GROUPS = ("codex", "groq", "cohere", "local_models")

class PheromoneBank:
    """Maintain pheromone levels per group."""
    def __init__(self, decay: float = 0.05):
        self.levels: Dict[str, float] = {g: 0.0 for g in GROUPS}
        self.decay = decay

    def decay_all(self):
        for g in self.levels:
            self.levels[g] *= (1.0 - self.decay)

    def release(self, group: str, amount: float):
        self.levels[group] += amount

    def get(self, group: str) -> float:
        return self.levels.get(group, 0.0)

# ---------------------------------------------------------------------------
# Hybrid core (fusion of both parents)
# ---------------------------------------------------------------------------

@dataclass
class HybridParams:
    alpha: float                # fractional order (0 < α < 1)
    tau_base: float             # base effective time constant
    lambda_pheromone: float     # coupling strength of pheromone term
    max_history: int = 30       # how many past days to keep for memory

def init_hybrid_system(num_endpoints: int = 8) -> Tuple[List[EngineEndpoint], PheromoneBank, HybridParams]:
    """Create a synthetic set of EngineEndpoints, a pheromone bank and default parameters."""
    random.seed(42)
    endpoints: List[EngineEndpoint] = []
    for i in range(num_endpoints):
        morph = Morphology(
            length=random.uniform(0.5, 2.5),
            width=random.uniform(0.5, 2.5),
            height=random.uniform(0.5, 2.5),
            mass=random.uniform(10, 100)
        )
        ep = EngineEndpoint(
            engine_id=f"eng-{i}",
            channel="api",
            residency="cloud",
            runtime="python3.11",
            resource_class=random.choice(GROUPS),
            always_on=random.choice([True, False]),
            endpoint=f"https://service{i}.example.com",
            capabilities=["inference", "embedding"],
            morphology=morph
        )
        endpoints.append(ep)

    pheromones = PheromoneBank(decay=0.07)
    params = HybridParams(alpha=0.6, tau_base=0.8, lambda_pheromone=0.4, max_history=30)
    return endpoints, pheromones, params

def compute_tau_eff(endpoint: EngineEndpoint, max_volume: float, max_ram: int, params: HybridParams) -> float:
    """
    Effective time constant τ_eff,i = τ₀ * (volume_i / max_volume) * (VRAM_i / max_VRAM)
    VRAM_i is derived from a dummy ModelTier based on mass (larger mass → larger VRAM).
    """
    # Derive a synthetic VRAM budget: 10 MB per unit mass, capped between 128 and 2048 MB
    vram_i = int(min(max(128, endpoint.morphology.mass * 10), 2048))
    vol_factor = endpoint.morphology.volume / max_volume if max_volume > 0 else 0.0
    ram_factor = vram_i / max_ram if max_ram > 0 else 0.0
    return params.tau_base * vol_factor * ram_factor

def hybrid_allocate_day(
    endpoints: List[EngineEndpoint],
    pheromones: PheromoneBank,
    past_alloc: Dict[str, List[float]],
    params: HybridParams
) -> Dict[str, float]:
    """
    Perform allocation for a single day.
    Returns a dict mapping engine_id -> allocated VRAM (MB).
    """
    # Determine global scaling factors
    volumes = np.array([ep.morphology.volume for ep in endpoints])
    max_volume = float(volumes.max()) if volumes.size else 1.0
    # synthetic max RAM based on the same rule used in compute_tau_eff
    max_ram = max(int(min(max(128, ep.morphology.mass * 10), 2048)) for ep in endpoints)

    # Pre‑compute fractional coefficients (same for all endpoints)
    coeffs = caputo_coeffs(params.alpha, params.max_history)

    allocations: Dict[str, float] = {}
    for ep in endpoints:
        # Baseline proportional to mass
        baseline = ep.morphology.mass * 0.1  # MB per unit mass

        # Effective time constant from morphology & VRAM budget
        tau_eff = compute_tau_eff(ep, max_volume, max_ram, params)

        # Fractional memory term using past allocations for this endpoint
        hist = past_alloc.get(ep.engine_id, [])
        n = min(len(hist), params.max_history)
        if n > 0:
            frac_term = sum(coeffs[k] * hist[-k-1] for k in range(n))
        else:
            frac_term = 0.0

        # Pheromone contribution (group‑specific)
        phi = pheromones.get(ep.resource_class)
        morph_weight = ep.morphology.volume / (ep.morphology.mass + 1e-6)

        allocation = baseline + tau_eff * frac_term + params.lambda_pheromone * phi * morph_weight

        # Clip to non‑negative and to the synthetic VRAM ceiling
        vram_cap = int(min(max(128, ep.morphology.mass * 10), 2048))
        allocation = max(0.0, min(allocation, vram_cap))

        allocations[ep.engine_id] = allocation

    # ---- Update pheromones based on today's allocation ----
    pheromones.decay_all()
    for ep in endpoints:
        # Release pheromone proportional to allocated VRAM weighted by morphology volume
        release_amount = allocations[ep.engine_id] * (ep.morphology.volume / (max_volume + 1e-6))
        pheromones.release(ep.resource_class, release_amount * 0.01)  # small scaling factor

    # ---- Store allocations for future memory ----
    for ep_id, val in allocations.items():
        past_alloc.setdefault(ep_id, []).append(val)
        # keep history bounded
        if len(past_alloc[ep_id]) > params.max_history:
            past_alloc[ep_id].pop(0)

    return allocations

def summarize_hybrid(
    history: Dict[str, List[float]],
    start_date: date,
    days: int
) -> Tuple[float, float]:
    """
    Compute total allocated VRAM over the observed window and the
    average daily allocation. Returns (total, average_per_day).
    """
    total = sum(sum(vals) for vals in history.values())
    avg = total / days if days > 0 else 0.0
    return _pct(total), _pct(avg)

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Initialise system
    endpoints, pheromones, params = init_hybrid_system(num_endpoints=10)

    # Historical allocations per endpoint
    past_alloc: Dict[str, List[float]] = {}

    # Simulate 15 days
    start = date.today()
    for offset in range(15):
        today = start + timedelta(days=offset)
        daily = hybrid_allocate_day(endpoints, pheromones, past_alloc, params)
        print(f"{today.isoformat()} allocations (MB):")
        for eid, val in daily.items():
            print(f"  {eid}: {_pct(val)}")
        print("-" * 40)

    total, avg = summarize_hybrid(past_alloc, start, 15)
    print(f"\n=== Summary over 15 days ===")
    print(f"Total allocated VRAM: {total} MB")
    print(f"Average per day: {avg} MB")
    sys.exit(0)