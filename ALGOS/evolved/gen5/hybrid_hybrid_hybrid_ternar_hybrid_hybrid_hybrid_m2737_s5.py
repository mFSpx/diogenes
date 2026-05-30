# DARWIN HAMMER — match 2737, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py (gen4)
# born: 2026-05-29T23:45:28Z

import numpy as np
import re
import math
import random
import sys
from pathlib import Path

# Regex feature set
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)

# Helper functions
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

class EndpointCircuitBreaker:
    """Circuit‑breaker."""

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
    return 2 * residual[:, None] @ x[None, :]

def hybrid_step(state: dict, text: str) -> None:
    """
    Run one iteration (router, NLMS, ODE, TTT update, circuit‑breaker handling).
    """
    # Extract regex features
    regex_features = extract_regex_features(text)

    # TTT loss and gradient
    ttt_loss_val = ttt_loss(state["W"], state["x"])
    ttt_grad_val = ttt_grad(state["W"], state["x"])

    # Pruning probability
    p_prune = sigmoid(-state["k_prune"] * ttt_loss_val)

    # Route selection
    route = np.random.choice([0, 1, 2], p=[p_prune, 1 - p_prune, 0])

    # NLMS adaptation of regex weights
    if route != 2:
        target_regex = np.array([1.0, 1.0])  # Target regex features
        error_nlms = target_regex - sigmoid(state["w_regex"] @ regex_features)
        state["w_regex"] += state["mu_nlms"] * error_nlms * regex_features

    # Decision-hygiene update
    if route != 1:
        u = np.concatenate((state["x"], regex_features, [state["s"]]))
        f = sigmoid(state["W"] @ u + state["b"])
        tau = state["tau_base"] * (state["tau_factor"] if not state["circuit"].allow() else 1.0)
        state["x"] += 0.01 * (-(1 / tau + f) * state["x"] + f * np.ones_like(state["x"]))

    # TTT gradient step
    state["W"] -= state["lr_ttt"] * ttt_grad_val

def evaluate_ssim(a: np.ndarray, b: np.ndarray) -> float:
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

# Smoke test
if __name__ == "__main__":
    state = init_hybrid()
    text = "This is a test text with some evidence and planning."
    for _ in range(10):
        hybrid_step(state, text)
        print(state["x"])