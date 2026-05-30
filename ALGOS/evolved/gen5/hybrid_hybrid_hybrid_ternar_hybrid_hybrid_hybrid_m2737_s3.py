# DARWIN HAMMER — match 2737, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py (gen4)
# born: 2026-05-29T23:45:28Z

"""Hybrid Ternary‑Router / TTT‑Linear / NLMS Decision‑Hygiene Model
================================================================

Parents
-------
* **Parent A** – `hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py`  
  Provides a TTT‑Linear model (`W`) with a self‑supervised loss  
  `L_TTT = ||W x − x||²` and a ternary router whose pruning probability
  is meant to be modulated by the model’s performance (SSIM).

* **Parent B** – `hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py`  
  Supplies a decision‑hygiene system that computes  

      f = σ( W·[x; I; s] + b )                (1)

  and evolves the internal state `x` with a liquid‑time‑constant ODE  

      dx/dt = -(1/τ + f)·x + f·A                (2)

  together with an NLMS‑based adaptation of regex‑pattern weights and a
  circuit‑breaker that influences the diffusion time constant `τ`.

Mathematical Bridge
-------------------
Both parents rely on a *single* weight matrix `W`.  In the hybrid we let
the **TTT‑Linear matrix** be the exact `W` used in (1).  The TTT loss
provides a scalar performance measure `L_TTT`; we map it to a pruning
probability `p_prune = σ(‑κ·L_TTT)` (κ > 0).  This probability drives the
ternary router, i.e. the choice of which update components (full ODE,
NLMS, or both) are applied at a given step.

Conversely, the NLMS update of the regex‑feature weights supplies an
error signal `e_NLMS` that is fed back into the TTT gradient descent as
an additional regularisation term, thereby coupling the two learning
mechanisms.

The circuit‑breaker state modulates the diffusion time constant `τ` in
(2): when the breaker is open, `τ` is multiplied by a safety factor
`τ_factor > 1`, slowing the dynamics.

The resulting unified system therefore evolves according to


L_TTT   = ||W x − x||²
p_prune = σ(‑κ·L_TTT)

route ∈ {0,1,2}   # 0 = full update, 1 = NLMS only, 2 = ODE only
route = ternary_router(p_prune)

if route != 2:
    # NLMS adaptation of regex weights w_r
    e_NLMS = target_regex – σ(w_r·r)
    w_r ← w_r + μ·e_NLMS·r / (ε + ‖r‖²)

if route != 1:
    # Decision‑hygiene update
    u   = concat(x, r, s)
    f   = σ(W·u + b)
    τ   = τ_base * (τ_factor if not circuit_breaker.allow() else 1.0)
    x   ← x + dt·[ -(1/τ + f)·x + f·A ]

# TTT gradient step (always performed)
W ← W – η·∇_W L_TTT  – η·λ·e_NLMS·∂(σ(w_r·r))/∂W   (λ couples NLMS error)
b ← b – η·∇_b L_TTT


The code below implements this hybrid dynamics with three public
functions:

* `init_hybrid(...)` – creates the model state.
* `hybrid_step(state, text)` – runs one iteration (router, NLMS, ODE,
  TTT update, circuit‑breaker handling).
* `evaluate_ssim(a, b)` – a lightweight SSIM‑like similarity used for
  the original routing metric.

Running the module as a script executes a short smoke test that
exercises all components without external dependencies."""

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Regex feature set (parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """
    Very small SSIM‑style similarity used for routing.
    Returns a value in [0, 1]; 1 means identical.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    numerator = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(numerator / denominator)


def extract_regex_features(text: str) -> np.ndarray:
    """
    Returns a 2‑dimensional feature vector:
    [evidence_match_count, planning_match_count] normalized by length.
    """
    length = max(len(text), 1)
    ev = len(EVIDENCE_RE.findall(text)) / length
    pl = len(PLANNING_RE.findall(text)) / length
    return np.array([ev, pl], dtype=np.float64)


# ----------------------------------------------------------------------
# Core classes
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Circuit‑breaker from parent B."""

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
        """True if the breaker is closed (operations allowed)."""
        return not self.open


# ----------------------------------------------------------------------
# Hybrid model initialization
# ----------------------------------------------------------------------
def init_hybrid(
    d_state: int = 3,
    d_regex: int = 2,
    scale: float = 0.01,
    seed: int = 0,
    lr_ttt: float = 1e-3,
    mu_nlms: float = 0.1,
    tau_base: float = 0.5,
    tau_factor: float = 3.0,
) -> dict:
    """
    Create the hybrid model state.

    Returns a dictionary with the following keys:
        W          – weight matrix for the decision‑hygiene (shape (d_out, d_in))
        b          – bias vector (shape (d_out,))
        x          – internal continuous state (shape (d_state,))
        s          – morphology priority scalar in [0,1]
        w_regex    – NLMS weights for regex features (shape (d_regex,))
        circuit    – EndpointCircuitBreaker instance
        lr_ttt     – learning rate for TTT gradient step
        mu_nlms    – NLMS step size
        tau_base   – base diffusion time constant
        tau_factor – factor applied when circuit breaker is open
        k_prune    – scaling factor κ for pruning probability
        rng        – NumPy random generator
    """
    rng = np.random.default_rng(seed)
    d_in = d_state + d_regex + 1  # x ; regex features ; s
    d_out = d_state  # keep same dimensionality for simplicity

    W = rng.standard_normal((d_out, d_in)) * scale
    b = np.zeros(d_out, dtype=np.float64)
    x = np.zeros(d_state, dtype=np.float64)
    s = 0.5  # neutral priority
    w_regex = np.zeros(d_regex, dtype=np.float64)

    state = {
        "W": W,
        "b": b,
        "x": x,
        "s": s,
        "w_regex": w_regex,
        "circuit": EndpointCircuitBreaker(),
        "lr_ttt": lr_ttt,
        "mu_nlms": mu_nlms,
        "tau_base": tau_base,
        "tau_factor": tau_factor,
        "k_prune": 5.0,  # κ in the pruning probability formula
        "rng": rng,
    }
    return state


# ----------------------------------------------------------------------
# TTT‑Linear utilities (parent A)
# ----------------------------------------------------------------------
def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    """Reconstruction loss ||W x − target||², target defaults to x."""
    if target is None:
        target = x
    pred = W @ x
    residual = pred - target
    return float(residual @ residual)


def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    """Gradient of the reconstruction loss w.r.t. W."""
    if target is None:
        target = x
    pred = W @ x
    residual = pred - target
    # dL/dW = 2 * residual * xᵀ
    return 2.0 * np.outer(residual, x)


# ----------------------------------------------------------------------
# NLMS update for regex feature weights (parent B)
# ----------------------------------------------------------------------
def nlms_update(
    w: np.ndarray,
    r: np.ndarray,
    target: float,
    mu: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalized LMS adaptation.

    w_new = w + μ * e * r / (ε + ||r||²)
    where e = target - σ(w·r)
    """
    pred = float(sigmoid(np.dot(w, r)))
    e = target - pred
    norm = eps + np.dot(r, r)
    return w + mu * e * r / norm


# ----------------------------------------------------------------------
# Ternary router (parent A)
# ----------------------------------------------------------------------
def ternary_router(state: dict, loss_val: float) -> int:
    """
    Decide which update components to execute.

    Pruning probability:
        p = σ(‑κ·loss_val)

    Returns:
        0 – execute **full** update (ODE + NLMS)
        1 – execute **NLMS only**
        2 – execute **ODE only**
    """
    κ = state["k_prune"]
    p = float(sigmoid(-κ * loss_val))
    r = state["rng"].random()
    if r < p:
        return 0
    # Split remaining probability equally between the two reduced modes
    return 1 if r < (p + (1 - p) / 2) else 2


# ----------------------------------------------------------------------
# Hybrid step – integrates all pieces
# ----------------------------------------------------------------------
def hybrid_step(state: dict, text: str, dt: float = 0.1) -> dict:
    """
    Perform a single hybrid iteration.

    1. Extract regex features `r`.
    2. Compute TTT loss on current internal state `x`.
    3. Choose a route via `ternary_router`.
    4. Apply NLMS adaptation if permitted.
    5. Apply decision‑hygiene ODE update if permitted.
    6. Perform TTT gradient descent (always).
    7. Update circuit‑breaker based on loss magnitude.
    8. Return the updated state (in‑place modification).
    """
    # ------------------------------------------------------------------
    # 1. Feature extraction
    # ------------------------------------------------------------------
    r = extract_regex_features(text)  # shape (2,)
    # Target for regex NLMS: 1 if any evidence pattern appears, else 0
    target_regex = 1.0 if EVIDENCE_RE.search(text) else 0.0

    # ------------------------------------------------------------------
    # 2. TTT loss on current x
    # ------------------------------------------------------------------
    W = state["W"]
    x = state["x"]
    loss_val = t