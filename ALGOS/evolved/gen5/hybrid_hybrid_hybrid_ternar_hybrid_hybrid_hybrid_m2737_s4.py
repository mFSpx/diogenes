# DARWIN HAMMER — match 2737, survivor 4
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

# ----------------------------------------------------------------------
# Regex feature set
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
    SSIM‑style similarity used for routing.
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
    rng = np.random.default_rng(seed)
    d_in = d_state + d_regex + 1  
    d_out = d_state  

    W = rng.standard_normal((d_out, d_in)) * scale
    b = np.zeros(d_out, dtype=np.float64)
    x = np.zeros(d_state, dtype=np.float64)
    s = 0.5  
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
        "k_prune": 5.0,  
        "rng": rng,
    }
    return state

# ----------------------------------------------------------------------
# TTT‑Linear utilities
# ----------------------------------------------------------------------
def ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> float:
    if target is None:
        target = x
    pred = W @ x
    residual = pred - target
    return float(residual @ residual)

def ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray = None) -> np.ndarray:
    if target is None:
        target = x
    pred = W @ x
    residual = pred - target
    return 2 * np.outer(residual, x)

# ----------------------------------------------------------------------
# Hybrid model step
# ----------------------------------------------------------------------
def hybrid_step(state: dict, text: str) -> dict:
    W, b, x, s, w_regex, circuit, lr_ttt, mu_nlms, tau_base, tau_factor, k_prune = (
        state["W"],
        state["b"],
        state["x"],
        state["s"],
        state["w_regex"],
        state["circuit"],
        state["lr_ttt"],
        state["mu_nlms"],
        state["tau_base"],
        state["tau_factor"],
        state["k_prune"],
    )

    r = extract_regex_features(text)
    u = np.concatenate((x, r, [s]))

    L_TTT = ttt_loss(W, x)
    p_prune = sigmoid(-k_prune * L_TTT)

    route = np.random.choice([0, 1, 2], p=[1 - p_prune, p_prune * 0.5, p_prune * 0.5])

    if route != 2:
        e_NLMS = r - sigmoid(w_regex @ r)
        w_regex += mu_nlms * e_NLMS * r / (1e-8 + np.linalg.norm(r) ** 2)

    if route != 1:
        f = sigmoid(W @ u + b)
        tau = tau_base * (tau_factor if not circuit.allow() else 1.0)
        dxdt = -(1 / tau + f) * x + f * np.ones_like(x)
        x += 0.01 * dxdt  # Assuming a fixed dt for simplicity

    # TTT gradient step
    grad_W = ttt_grad(W, x)
    grad_b = 2 * (W @ x - x)
    W -= lr_ttt * grad_W - lr_ttt * 0.1 * e_NLMS * np.outer(r, np.ones_like(x))
    b -= lr_ttt * grad_b

    state["W"] = W
    state["b"] = b
    state["x"] = x
    state["w_regex"] = w_regex
    return state

# ----------------------------------------------------------------------
# Evaluate SSIM
# ----------------------------------------------------------------------
def evaluate_ssim(a: np.ndarray, b: np.ndarray) -> float:
    return ssim_like(a, b)

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    state = init_hybrid()
    for _ in range(10):
        state = hybrid_step(state, "This is a test sentence.")
        print(state["x"])