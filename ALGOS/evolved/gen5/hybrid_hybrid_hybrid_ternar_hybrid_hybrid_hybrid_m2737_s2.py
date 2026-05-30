# DARWIN HAMMER — match 2737, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py (gen4)
# born: 2026-05-29T23:45:28Z

"""Hybrid Model combining TTT‑Linear self‑supervision with NLMS adaptive filtering and
Decision‑Hygiene regex weighting.

Parents:
* hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (TTT‑Linear + ternary router,
  SSIM‑based pruning)
* hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py (NLMS with circuit‑breaker,
  morphology‑driven priority, decision‑hygiene regex scores, liquid‑time diffusion)

Mathematical bridge:
The weight matrix **W** of the TTT‑Linear model is reused as the adaptive filter of the
NLMS algorithm.  The NLMS error **e = d – W·x** is exactly the reconstruction error used by
the TTT loss.  The step‑size **μ** of NLMS is modulated by the ternary‑router pruning
probability **p_prune(SSIM)**, i.e.

    μ_t = μ_0 · (1 – p_prune)

where **SSIM** is approximated by a simple structural similarity based on mean and variance.
The circuit‑breaker state further scales the diffusion time‑constant **τ** used in the
liquid‑time diffusion equation

    dx/dt = –(1/τ + f)·x + f·A

with **f = σ( W·[x; I; s] + b )** coming from a decision‑hygiene layer whose regex pattern
weights are derived from rows of **W**.  Morphology‑driven priority influences the
diffusion timestep by a factor **s ∈ [0,1]**.

The resulting system jointly updates **W** (TTT/NLMS), adapts **μ** (pruning), and evolves
a latent state **x** (liquid diffusion) while scoring textual evidence via decision‑hygiene.
"""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Regex feature sets (decision‑hygiene)
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Simple SSIM approximation (mean‑variance based)
def approx_ssim(x: np.ndarray, y: np.ndarray) -> float:
    """Return a value in [0,1] approximating SSIM between two vectors."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return float(num / den) if den != 0 else 0.0

def ternary_prune_probability(ssim: float, threshold: float = 0.8) -> float:
    """Map SSIM to a pruning probability. Higher SSIM ⇒ lower pruning."""
    return max(0.0, min(1.0, 1.0 - (ssim - threshold) / (1.0 - threshold)))

# ----------------------------------------------------------------------
# Circuit‑breaker implementation (parent B)
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        return not self.open

# ----------------------------------------------------------------------
# Morphology placeholder (priority factor)
class Morphology:
    def __init__(self, priority: float = 0.5):
        # priority ∈ [0,1]; higher → more aggressive diffusion timestep scaling
        self.priority = max(0.0, min(1.0, priority))

# ----------------------------------------------------------------------
# Core hybrid model
class HybridModel:
    def __init__(
        self,
        d_in: int,
        d_out: int = None,
        scale: float = 0.01,
        seed: int = 0,
        mu_initial: float = 0.1,
    ):
        """Initialize weight matrix W (TTT‑Linear) and NLMS step size μ."""
        rng = np.random.default_rng(seed)
        if d_out is None:
            d_out = d_in
        self.W = rng.standard_normal((d_out, d_in)) * scale
        self.mu = mu_initial  # NLMS step size
        self.epsilon = 1e-8    # stability constant for NLMS normalization
        self.circuit_breaker = EndpointCircuitBreaker()
        self.morphology = Morphology()
        # bias term for decision‑hygiene layer
        self.b = np.zeros(d_out)

    # ------------------------------------------------------------------
    # TTT loss (reconstruction)
    def ttt_loss(self, x: np.ndarray, target: np.ndarray = None) -> float:
        pred = self.W @ x
        t = x if target is None else target
        resid = pred - t
        return float(resid @ resid)

    # NLMS weight update using the same error as TTT loss
    def nlms_update(self, x: np.ndarray, d: np.ndarray) -> None:
        """Perform a single NLMS adaptation step."""
        e = d - self.W @ x                               # error vector
        norm = float(x @ x) + self.epsilon               # squared norm of input
        step = (self.mu / norm) * np.outer(e, x)         # ΔW
        self.W += step

    # Decision‑hygiene regex scoring using rows of W as pattern weights
    def regex_score(self, text: str) -> float:
        """Score text by summing weighted regex matches."""
        evidence_matches = len(EVIDENCE_RE.findall(text))
        planning_matches = len(PLANNING_RE.findall(text))
        # Use first two rows of W (if available) as weights; otherwise fallback to 1.0
        w_evidence = self.W[0, 0] if self.W.shape[0] > 0 else 1.0
        w_planning = self.W[1, 0] if self.W.shape[0] > 1 else 1.0
        return w_evidence * evidence_matches + w_planning * planning_matches

    # Liquid‑time diffusion step (parent B)
    def diffusion_step(
        self,
        x: np.ndarray,
        I: np.ndarray,
        s: float,
        A: np.ndarray,
        tau: float,
    ) -> np.ndarray:
        """One Euler integration step of dx/dt = –(1/τ+f)·x + f·A."""
        # Concatenate state, input and priority flag
        concat = np.concatenate([x, I, np.array([s])])
        # Linear projection + bias, passed through sigmoid
        f = 1.0 / (1.0 + np.exp(- (self.W @ concat + self.b).mean()))
        dt = 0.01 * (1.0 - s) * (1.0 if self.circuit_breaker.allow() else 0.0)
        dx = -(1.0 / tau + f) * x + f * A
        return x + dt * dx

    # ------------------------------------------------------------------
    # Full hybrid step integrating all components
    def hybrid_step(
        self,
        x: np.ndarray,
        target: np.ndarray,
        text: str,
        I: np.ndarray,
        A: np.ndarray,
        tau: float = 1.0,
    ) -> dict:
        """
        Perform:
        1. Forward pass (W·x), compute SSIM against target.
        2. Derive pruning probability → adapt NLMS step size μ.
        3. NLMS weight update.
        4. Regex (decision‑hygiene) scoring.
        5. Liquid‑time diffusion of a latent state.
        Returns a dict with intermediate diagnostics.
        """
        # 1. Forward and SSIM
        y = self.W @ x
        ssim_val = approx_ssim(y, target)

        # 2. Pruning probability modulates μ
        p_prune = ternary_prune_probability(ssim_val)
        self.mu = max(1e-4, self.mu * (1.0 - p_prune))

        # 3. NLMS update using target as desired signal
        self.nlms_update(x, target)

        # 4. Decision‑hygiene regex score
        regex_sc = self.regex_score(text)

        # 5. Diffusion step; priority s taken from morphology
        s = self.morphology.priority
        x_next = self.diffusion_step(x, I, s, A, tau)

        # Circuit‑breaker feedback (simple heuristic)
        if ssim_val < 0.5:
            self.circuit_breaker.record_failure()
        else:
            self.circuit_breaker.record_success()

        return {
            "output": y,
            "ssim": ssim_val,
            "prune_prob": p_prune,
            "mu": self.mu,
            "regex_score": regex_sc,
            "state_next": x_next,
            "circuit_open": not self.circuit_breaker.allow(),
        }

# ----------------------------------------------------------------------
# Helper functions exposing the hybrid operations (at least three)
def init_hybrid_model(d_in: int, seed: int = 0) -> HybridModel:
    """Factory for a HybridModel with reproducible initialization."""
    return HybridModel(d_in=d_in, seed=seed)

def run_hybrid_simulation(steps: int = 5) -> None:
    """Run a short simulation demonstrating the hybrid dynamics."""
    dim = 8
    model = init_hybrid_model(d_in=dim, seed=42)

    # Random initial state vectors
    x = np.random.randn(dim)
    I = np.random.randn(dim)          # external input
    A = np.random.randn(dim)          # attractor vector for diffusion
    target = np.random.randn(dim)     # desired reconstruction

    sample_text = (
        "The evidence was verified and the plan was documented in the checklist."
    )

    for t in range(steps):
        diagnostics = model.hybrid_step(
            x=x,
            target=target,
            text=sample_text,
            I=I,
            A=A,
            tau=1.0,
        )
        # advance state for next iteration
        x = diagnostics["state_next"]
        # (optional) print a concise summary
        print(
            f"Step {t+1}: SSIM={diagnostics['ssim']:.3f}, μ={diagnostics['mu']:.5f}, "
            f"RegexScore={diagnostics['regex_score']:.2f}, CircuitOpen={diagnostics['circuit_open']}"
        )

def evaluate_pruning_effect(x: np.ndarray, target: np.ndarray) -> float:
    """Compute pruning probability for a single input‑target pair."""
    model = init_hybrid_model(d_in=x.shape[0], seed=1)
    y = model.W @ x
    ssim_val = approx_ssim(y, target)
    return ternary_prune_probability(ssim_val)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke test: run a brief simulation and a single pruning evaluation
    run_hybrid_simulation(steps=3)

    # Test pruning function
    dim = 4
    x_test = np.array([0.2, -0.1, 0.5, 0.0])
    target_test = np.array([0.1, -0.05, 0.45, 0.02])
    p = evaluate_pruning_effect(x_test, target_test)
    print(f"Pruning probability for test pair: {p:.3f}")