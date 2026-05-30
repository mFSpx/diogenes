# DARWIN HAMMER — match 2737, survivor 6
# gen: 5
# parent_a: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1025_s0.py (gen4)
# born: 2026-05-29T23:45:28Z

import numpy as np
import re
import math
import random
from typing import Dict, Any

# ----------------------------------------------------------------------
# Regex feature extraction (parent B)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid, numerically stable."""
    # Clip to avoid overflow
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """Very small SSIM‑style similarity used for routing."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    sigma_a = a.var()
    sigma_b = b.var()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a + sigma_b + C2)
    return float(num / den)


def extract_regex_features(text: str) -> np.ndarray:
    """Return a 2‑dimensional feature vector normalized by text length."""
    length = max(len(text), 1)
    ev = len(EVIDENCE_RE.findall(text)) / length
    pl = len(PLANNING_RE.findall(text)) / length
    return np.array([ev, pl], dtype=np.float64)


# ----------------------------------------------------------------------
# Circuit breaker (parent B)
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple failure‑count circuit breaker."""

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
# Hybrid model
# ----------------------------------------------------------------------
def init_hybrid(
    d_state: int = 3,
    d_regex: int = 2,
    scale: float = 0.01,
    seed: int = 0,
    lr_ttt: float = 1e-3,
    lr_regex: float = 5e-3,
    tau_base: float = 0.5,
    tau_factor: float = 3.0,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Dict[str, Any]:
    """
    Initialise the hybrid model.

    Parameters
    ----------
    d_state : int
        Dimensionality of the continuous internal state `x`.
    d_regex : int
        Dimensionality of the regex feature vector (fixed to 2 here).
    scale : float
        Std‑dev of the initial weight matrix.
    seed : int
        Random seed for reproducibility.
    lr_ttt : float
        Learning rate for the TTT‑Linear part (shared weights `W`).
    lr_regex : float
        Learning rate for the regex predictor weights `w_regex`.
    tau_base : float
        Base diffusion time constant in the ODE.
    tau_factor : float
        Multiplicative factor applied when the circuit breaker is open.
    alpha, beta : float
        Relative weighting of the reconstruction loss and the regex loss
        in the joint optimisation objective.

    Returns
    -------
    dict
        Model state dictionary.
    """
    rng = np.random.default_rng(seed)

    d_in = d_state + d_regex + 1          # x ; regex features ; s
    d_out = d_state                       # keep same dimensionality for simplicity

    W = rng.standard_normal((d_out, d_in)) * scale
    b = np.zeros(d_out, dtype=np.float64)

    state = {
        "W": W,
        "b": b,
        "x": np.zeros(d_state, dtype=np.float64),
        "s": 0.5,                         # neutral morphology priority
        "w_regex": np.zeros(d_regex, dtype=np.float64),
        "circuit": EndpointCircuitBreaker(),
        "lr_ttt": lr_ttt,
        "lr_regex": lr_regex,
        "tau_base": tau_base,
        "tau_factor": tau_factor,
        "k_prune": 5.0,                  # κ in the pruning probability formula
        "alpha": alpha,
        "beta": beta,
        "rng": rng,
    }
    return state


def _ttt_loss(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> float:
    """Reconstruction loss ||W x - target||²."""
    pred = W @ x
    resid = pred - target
    return float(resid @ resid)


def _ttt_grad(W: np.ndarray, x: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gradient of reconstruction loss w.r.t. W."""
    resid = (W @ x) - target
    # d/dW (resid^T resid) = 2 * resid[:, None] * x[None, :]
    return 2.0 * np.outer(resid, x)


def _regex_loss(w: np.ndarray, r: np.ndarray, target: float) -> float:
    """Binary cross‑entropy‑like loss for regex prediction."""
    pred = sigmoid(w @ r)
    # Use squared error for simplicity (still differentiable)
    return float((pred - target) ** 2)


def _regex_grad(w: np.ndarray, r: np.ndarray, target: float) -> np.ndarray:
    """Gradient of regex loss w.r.t. w."""
    pred = sigmoid(w @ r)
    dloss_dpred = 2.0 * (pred - target)
    dpred_dz = pred * (1.0 - pred)
    return dloss_dpred * dpred_dz * r


def _ternary_router(p_prune: float, rng: np.random.Generator) -> int:
    """
    Decide which update components to apply.

    Returns
    -------
    int
        0 → full update (ODE + regex), 1 → regex only, 2 → ODE only.
    """
    # Sample a uniform number; three equally‑spaced intervals give a simple ternary decision.
    u = rng.random()
    if u < p_prune / 2.0:
        return 2                     # ODE only (prune regex update)
    elif u < p_prune:
        return 1                     # Regex only (prune ODE update)
    else:
        return 0                     # Full update


def hybrid_step(state: Dict[str, Any], text: str, target_regex: float = 1.0) -> None:
    """
    Perform a single hybrid iteration.

    Parameters
    ----------
    state : dict
        Model state returned by ``init_hybrid``.
    text : str
        Input text used for regex feature extraction.
    target_regex : float, optional
        Desired regex‑prediction target (default 1.0 → “evidence present”).
    """
    # ------------------------------------------------------------------
    # 1️⃣ Extract features and build the joint input vector
    # ------------------------------------------------------------------
    r = extract_regex_features(text)               # shape (d_regex,)
    u = np.concatenate([state["x"], r, np.array([state["s"]])])  # (d_in,)

    # ------------------------------------------------------------------
    # 2️⃣ Compute routing probability from the TTT loss
    # ------------------------------------------------------------------
    loss_ttt = _ttt_loss(state["W"], state["x"])
    p_prune = sigmoid(-state["k_prune"] * loss_ttt)   # maps large loss → small prune prob

    # ------------------------------------------------------------------
    # 3️⃣ Decide which components to run
    # ------------------------------------------------------------------
    route = _ternary_router(p_prune, state["rng"])

    # ------------------------------------------------------------------
    # 4️⃣ Regex (NLMS‑style) adaptation – now a proper gradient step
    # ------------------------------------------------------------------
    if route != 2:   # regex update allowed
        # Compute regex prediction loss and its gradient
        loss_regex = _regex_loss(state["w_regex"], r, target_regex)
        grad_w = _regex_grad(state["w_regex"], r, target_regex)

        # Simple SGD update (could be replaced by NLMS if desired)
        state["w_regex"] -= state["lr_regex"] * grad_w

        # Record success/failure for the circuit breaker based on loss magnitude
        if loss_regex < 0.01:
            state["circuit"].record_success()
        else:
            state["circuit"].record_failure()
    else:
        loss_regex = 0.0  # not used later but kept for completeness

    # ------------------------------------------------------------------
    # 5️⃣ Decision‑hygiene ODE update (RK2 for better stability)
    # ------------------------------------------------------------------
    if route != 1:   # ODE update allowed
        f = sigmoid(state["W"] @ u + state["b"])          # (d_state,)
        tau = state["tau_base"] * (state["tau_factor"] if not state["circuit"].allow() else 1.0)

        # RK2 (mid‑point) integration of dx/dt = -(1/τ + f)·x + f·A
        # For simplicity we set A = 0 (steady‑state target) – can be exposed later.
        A = np.zeros_like(state["x"])

        def _dx(x_cur: np.ndarray) -> np.ndarray:
            return -(1.0 / tau + f) * x_cur + f * A

        k1 = _dx(state["x"])
        x_mid = state["x"] + 0.5 * k1
        k2 = _dx(x_mid)
        state["x"] += k2   # dt = 1 for unit‑step simulation

    # ------------------------------------------------------------------
    # 6️⃣ Joint optimisation of W and b using the combined loss
    # ------------------------------------------------------------------
    # Joint loss = α * reconstruction + β * regex prediction (even if regex update was skipped)
    joint_loss = state["alpha"] * loss_ttt + state["beta"] * loss_regex

    # Gradients w.r.t. W and b
    grad_W = state["alpha"] * _ttt_grad(state["W"], state["x"])
    # Add coupling term: influence of regex error on W via the shared input u
    if route != 2:   # regex error is available
        grad_W += state["beta"] * np.outer(
            _regex_grad(state["w_regex"], r, target_regex),   # shape (d_regex,)
            np.concatenate([np.zeros_like(state["x"]), r, np.array([state["s"]])])
        )
    grad_b = state["alpha"] * 2.0 * (sigmoid(state["W"] @ u + state["b"]) - state["x"])

    # Simple SGD step
    state["W"] -= state["lr_ttt"] * grad_W
    state["b"] -= state["lr_ttt"] * grad_b

    # ------------------------------------------------------------------
    # 7️⃣ Optional: prune (zero‑out) a random subset of rows of W according to p_prune
    # ------------------------------------------------------------------
    if route == 2:   # ODE only – interpret as “prune regex influence”
        mask = state["rng"].random(state["W"].shape[0]) > p_prune
        state["W"] = state["W"] * mask[:, None]   # broadcast mask over columns

    # ------------------------------------------------------------------
    # 8️⃣ Update morphology priority `s` (simple exponential moving average of success)
    # ------------------------------------------------------------------
    success = state["circuit"].allow()
    state["s"] = 0.9 * state["s"] + 0.1 * float(success)


def evaluate_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Thin wrapper around the internal SSIM‑like function."""
    return ssim_like(a, b)


# ----------------------------------------------------------------------
# Simple smoke test (executed when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    state = init_hybrid(seed=42)
    texts = [
        "The report includes verified evidence and a detailed plan.",
        "Just a casual chat without any citations.",
        "Proof of concept was recorded, but the schedule is missing.",
    ]
    for i, txt in enumerate(texts, 1):
        hybrid_step(state, txt, target_regex=1.0)
        loss = _ttt_loss(state["W"], state["x"])
        print(f"Step {i:02d} | TTT loss: {loss:.6f} | s: {state['s']:.3f} | breaker open: {not state['circuit'].allow()}")
    print("Final W norm:", np.linalg.norm(state["W"]))