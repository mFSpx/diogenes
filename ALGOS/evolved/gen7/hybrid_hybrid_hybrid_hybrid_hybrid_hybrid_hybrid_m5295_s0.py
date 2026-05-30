# DARWIN HAMMER — match 5295, survivor 0
# gen: 7
# parent_a: hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s3.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m2327_s2.py (gen5)
# born: 2026-05-30T00:01:07Z

"""Hybrid Bayesian‑Privacy / NLMS Adaptive Filter with Regex‑driven Textual Compatibility

Parents:
- hybrid_hybrid_hybrid_bayes__hybrid_hybrid_hybrid_m1827_s3.py (Bayesian prior → per‑sample learning factor λᵢ)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m2327_s2.py (Regex feature extraction → compatibility scalar α)

Mathematical Bridge:
1. From Parent A we obtain a spatial prior vector **p** (∑pᵢ=1) and a health‑score hᵢ.
   The per‑sample composite learning factor is λᵢ = μ·pᵢ·hᵢ.

2. From Parent B we extract a raw feature count vector **f** from free‑text,
   then compute a signed feature score σ = f·p_weights – f·n_weights.
   A compatibility scalar α = vᵀ·P·m is built from the first two NLMS weights
   (v = w[:2]), a fixed compatibility matrix **P**, and a model vector **m**.

3. The NLMS desired signal becomes d = α·σ.  The NLMS update
   w ← w + λᵢ·(d−y)·x / (ε+‖x‖²)
   thus fuses the Bayesian‑derived privacy modulation (λᵢ) with the
   text‑driven compatibility target (d).  The resulting single adaptive loop
   respects both statistical privacy risk and semantic evidence scoring.

The module implements:
- spatial_aware_privacy_prior – builds **p** from a list of Entity objects.
- extract_features – regex‑driven feature extraction returning f, p_weights, n_weights.
- hybrid_nlms_step – a single NLMS adaptation step that incorporates λᵢ, α, and σ.
"""

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Shared data structures (Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""


def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def _signature(e: Entity) -> str:
    """Canonical signature used for quasi‑identifier matching."""
    return (e.address_signature or e.category).strip().lower()


def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple risk estimator (higher when few records share a quasi‑identifier)."""
    if total_records == 0:
        return 0.0
    return 1.0 - (unique_quasi_identifiers / total_records)


def spatial_aware_privacy_prior(entities: List[Entity]) -> np.ndarray:
    """
    Build a normalized privacy‑risk prior vector p ∈ ℝⁿ.
    For each entity we compute a risk score based on its categorical rarity.
    The resulting vector is normalised to sum to 1.
    """
    if not entities:
        raise ValueError("Entity list must not be empty")
    # Count occurrences of each signature
    sig_counts: Dict[str, int] = {}
    for e in entities:
        sig = _signature(e)
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    total = len(entities)
    risks = []
    for e in entities:
        sig = _signature(e)
        # Rarer signatures receive higher risk
        risk = reconstruction_risk_score(1, sig_counts[sig])
        risks.append(risk)
    risk_arr = np.array(risks, dtype=float)
    if risk_arr.sum() == 0:
        # fallback to uniform prior
        prior = np.full_like(risk_arr, 1.0 / len(risk_arr))
    else:
        prior = risk_arr / risk_arr.sum()
    return prior


def health_score(entity: Entity, prior_i: float, size_factor: float = 1.0) -> float:
    """
    Combine a morphological size factor with the Bayesian prior.
    The health score hᵢ is a simple product; other formulations are possible.
    """
    return prior_i * size_factor


# ----------------------------------------------------------------------
# Regex feature extraction (Parent B)
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
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)


def extract_features(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (f, p_weights, n_weights):
    - f : raw count vector for the five regex categories.
    - p_weights : positive influence weights (evidence, planning, support).
    - n_weights : negative influence weights (delay, boundary).
    """
    counts = np.array([
        len(EVIDENCE_RE.findall(text)),
        len(PLANNING_RE.findall(text)),
        len(DELAY_RE.findall(text)),
        len(SUPPORT_RE.findall(text)),
        len(BOUNDARY_RE.findall(text)),
    ], dtype=float)

    # Positive categories: evidence, planning, support (indices 0,1,3)
    p_weights = np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
    # Negative categories: delay, boundary (indices 2,4)
    n_weights = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=float)

    return counts, p_weights, n_weights


# ----------------------------------------------------------------------
# Hybrid NLMS core (fusion of both parents)
# ----------------------------------------------------------------------
class HybridNLMS:
    """
    Adaptive filter that merges:
    • Bayesian‑privacy prior → per‑sample learning factor λᵢ = μ·pᵢ·hᵢ
    • Regex‑driven textual compatibility → target d = α·σ
    where:
        σ = f·p_weights – f·n_weights
        α = vᵀ·P·m   (v = w[:2])
    The filter weight vector w is of configurable length (default 10).
    """

    def __init__(
        self,
        prior: np.ndarray,
        mu: float = 0.1,
        epsilon: float = 1e-6,
        weight_len: int = 10,
        compat_matrix: np.ndarray = None,
        model_vec: np.ndarray = None,
    ):
        if prior.ndim != 1:
            raise ValueError("Prior must be a 1‑D array")
        self.prior = prior.astype(float)                # pᵢ
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.w = np.zeros(weight_len, dtype=float)      # initial weights
        self.weight_len = weight_len

        # Compatibility matrix P (size 2×2) and model vector m (size 2)
        self.P = (
            compat_matrix
            if compat_matrix is not None
            else np.eye(2, dtype=float)
        )
        if self.P.shape != (2, 2):
            raise ValueError("Compatibility matrix P must be 2×2")
        self.m = (
            model_vec
            if model_vec is not None
            else np.ones(2, dtype=float)
        )
        if self.m.shape != (2,):
            raise ValueError("Model vector m must be length 2")

    def _lambda_i(self, idx: int, entity: Entity, size_factor: float = 1.0) -> float:
        """Composite learning factor λᵢ for sample i."""
        h_i = health_score(entity, self.prior[idx], size_factor)
        return self.mu * self.prior[idx] * h_i

    def _compatibility(self) -> float:
        """Compute α = vᵀ·P·m using the first two weights."""
        v = self.w[:2]
        return float(v @ self.P @ self.m)

    def _nlms_update(
        self,
        x: np.ndarray,
        d: float,
        lam: float,
    ) -> None:
        """
        Perform a single NLMS weight update.
        x : input vector (same length as w)
        d : desired scalar output
        lam : composite learning factor λᵢ
        """
        if x.shape != self.w.shape:
            raise ValueError("Input vector x must match weight dimension")
        y = float(self.w @ x)                     # current output
        error = d - y
        norm_factor = self.epsilon + np.dot(x, x)
        self.w += (lam * error / norm_factor) * x

    def step(
        self,
        idx: int,
        entity: Entity,
        text: str,
        x_input: np.ndarray,
        size_factor: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Execute one hybrid adaptation step.
        Returns (error, updated_weight_norm).
        """
        # 1️⃣ Composite learning factor λᵢ
        lam_i = self._lambda_i(idx, entity, size_factor)

        # 2️⃣ Feature extraction → σ
        f_vec, p_w, n_w = extract_features(text)
        sigma = float(f_vec @ p_w - f_vec @ n_w)

        # 3️⃣ Compatibility α
        alpha = self._compatibility()

        # 4️⃣ Desired signal d = α·σ
        d_target = alpha * sigma

        # 5️⃣ NLMS update
        self._nlms_update(x_input, d_target, lam_i)

        # 6️⃣ Compute current error for reporting
        y = float(self.w @ x_input)
        error = d_target - y
        return error, float(np.linalg.norm(self.w))


# ----------------------------------------------------------------------
# Demonstration functions (at least three)
# ----------------------------------------------------------------------
def build_dummy_entities(n: int) -> List[Entity]:
    """Create n synthetic Entity objects with varied categories."""
    categories = ["health", "finance", "social", "education", "transport"]
    entities = []
    for i in range(n):
        cat = random.choice(categories)
        ent = Entity(
            id=f"ent_{i}",
            lat=random.uniform(-90, 90),
            lon=random.uniform(-180, 180),
            category=cat,
            address_signature=cat,
        )
        entities.append(ent)
    return entities


def prepare_input_vector(weight_len: int, seed: int = 0) -> np.ndarray:
    """Generate a deterministic input vector x (e.g., random but reproducible)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=weight_len)


def run_hybrid_demo():
    """Run a short demonstration of the HybridNLMS on synthetic data."""
    # --- 1. Build prior from entities ---
    entities = build_dummy_entities(8)
    prior = spatial_aware_privacy_prior(entities)

    # --- 2. Initialise filter ---
    filter_ = HybridNLMS(prior=prior, mu=0.05, weight_len=10)

    # --- 3. Simulate a stream of samples ---
    texts = [
        "The evidence was verified and the plan is ready.",
        "We need to wait before the next step, please hold.",
        "Contact the doctor and get a receipt of the audit.",
        "No contact, privacy is essential, do not share.",
    ]

    x_input = prepare_input_vector(weight_len=10)

    for i, (entity, txt) in enumerate(zip(entities, texts)):
        err, norm = filter_.step(
            idx=i,
            entity=entity,
            text=txt,
            x_input=x_input,
            size_factor=1.0 + 0.1 * i,   # arbitrary size scaling
        )
        print(
            f"Step {i+1:02d} | Error={err: .4f} | ||w||={norm: .4f} | α={filter_._compatibility(): .4f}"
        )


if __name__ == "__main__":
    run_hybrid_demo()