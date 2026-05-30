# DARWIN HAMMER — match 4371, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_minhash_hybri_m2577_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m2737_s5.py (gen5)
# born: 2026-05-29T23:55:13Z

"""Hybrid Bandit‑MinHash‑NLMS‑Regex Fusion

This module fuses the core mathematics of the two parent algorithms:

* **Parent A** – Hybrid Bandit‑Sketch‑Workshare‑MinHash‑NLMS:
  - Action reward is modulated by a *variational free‑energy* term that
    evaluates similarity between the chosen action and a weekday‑dependent
    weight vector.
  - MinHash signatures adjust the learning‑rate of an NLMS weight update.

* **Parent B** – Hybrid Ternary‑Route‑Regex:
  - Provides a 2‑dimensional regex‑based feature extractor.
  - Supplies an SSIM‑style similarity (`ssim_like`) used for routing decisions.
  - Implements a circuit‑breaker for robustness.

**Mathematical Bridge**

The bridge is built on the observation that both parents compute a *similarity*
between two vectors:

* Parent A uses a variational free‑energy function `VFE(a, w)`.
* Parent B offers an SSIM‑style similarity `ssim_like(a, b)`.

We unify them by defining the free‑energy as the negative logarithm of the
SSIM‑style similarity, i.e.:


VFE(action_vec, weekday_vec) = -log( ssim_like(action_vec, weekday_vec) + ε )


The regex feature vector from Parent B is injected as the *context* for the
bandit’s action‑selection policy.  MinHash signatures derived from the same
context modulate the NLMS learning‑rate, completing the hybrid loop.

The resulting system:
1. Extract regex features → context vector `c`.
2. Compute a weekday‑dependent weight vector `w(d)`.
3. Select an action via a softmax bandit using `c` as logits.
4. Evaluate reward with reconstruction‑risk and the unified VFE.
5. Update NLMS weights with a learning‑rate adapted by the MinHash signature.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from datetime import date

# ----------------------------------------------------------------------
# Parent‑B: Regex feature extraction & SSIM‑style similarity
# ----------------------------------------------------------------------
EVIDENCE_RE = __import__("re").compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    __import__("re").I,
)
PLANNING_RE = __import__("re").compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    __import__("re").I,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Element‑wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-z))


def ssim_like(a: np.ndarray, b: np.ndarray) -> float:
    """
    Tiny SSIM‑style similarity used for routing / free‑energy.
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
    [evidence_match_norm, planning_match_norm] where each component is
    normalized by the text length.
    """
    length = max(len(text), 1)
    ev = len(EVIDENCE_RE.findall(text)) / length
    pl = len(PLANNING_RE.findall(text)) / length
    return np.array([ev, pl], dtype=np.float64)


# ----------------------------------------------------------------------
# Parent‑A: Bandit, NLMS, MinHash utilities
# ----------------------------------------------------------------------
def weekday_weight_vector(groups: tuple, today: date) -> np.ndarray:
    """
    Produce a deterministic weight vector of length ``len(groups)`` that
    varies with the weekday.  The vector is normalised to sum to 1.
    """
    rng = random.Random(today.toordinal())
    weights = np.array([rng.random() for _ in groups], dtype=np.float64)
    weights /= weights.sum()
    return weights


def reconstruction_risk_score(uid: str, total_records: int) -> float:
    """
    Placeholder risk estimator.  Returns a value in [0, 1].
    For demonstration we use an inverse‑frequency heuristic.
    """
    freq = max(total_records, 1)
    return min(1.0, 1.0 / math.sqrt(freq))


def variational_free_energy(action_vec: np.ndarray, weekday_vec: np.ndarray) -> float:
    """
    Unified VFE = -log( ssim_like(action_vec, weekday_vec) + ε )
    """
    eps = 1e-12
    sim = ssim_like(action_vec, weekday_vec)
    return -math.log(sim + eps)


def minhash_signature(data: np.ndarray, n_perm: int = 64, seed: int = 0) -> np.ndarray:
    """
    Very small MinHash implementation based on random permutations of indices.
    Returns a signature vector of length ``n_perm`` containing minima of hashed
    positions.
    """
    rng = np.random.default_rng(seed)
    perms = rng.integers(0, data.size, size=(n_perm, data.size))
    signatures = np.min(data[perms], axis=1)
    return signatures


def nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    lr: float,
    mu: float = 0.1,
) -> np.ndarray:
    """
    Normalised Least‑Mean‑Squares weight update.
    w  – current weight vector
    x  – input vector
    d  – desired scalar output
    lr – learning rate (may be adapted)
    mu – regularisation constant to avoid division by zero
    """
    y = np.dot(w, x)
    e = d - y
    norm = np.dot(x, x) + mu
    w_new = w + (lr * e / norm) * x
    return w_new


# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Simple circuit‑breaker from Parent B."""

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
        """True if operations are permitted."""
        return not self.open


def bandit_softmax_logits(context: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Convert a context vector into a probability distribution over actions
    using a temperature‑scaled softmax.
    """
    z = context / max(temperature, 1e-6)
    exp_z = np.exp(z - np.max(z))  # stability
    probs = exp_z / exp_z.sum()
    return probs


def select_action(probs: np.ndarray, rng: random.Random) -> int:
    """Sample an action index according to the given probability vector."""
    return int(rng.choices(range(len(probs)), weights=probs, k=1)[0])


# ----------------------------------------------------------------------
# Hybrid Agent
# ----------------------------------------------------------------------
class HybridAgent:
    """
    Integrates:
    * regex‑based contextual features (Parent B)
    * weekday‑dependent weight vectors (Parent A)
    * bandit action selection with VFE‑modulated reward
    * MinHash‑adapted NLMS weight updates
    """

    def __init__(
        self,
        groups: tuple = ("codex", "groq", "cohere", "local_models"),
        d_state: int = 4,
        seed: int = 0,
        base_lr: float = 1e-3,
    ):
        self.groups = groups
        self.d_state = d_state
        self.rng = random.Random(seed)
        # NLMS weight vector initialised small
        self.w = np.zeros(d_state, dtype=np.float64)
        self.base_lr = base_lr
        self.circuit = EndpointCircuitBreaker()
        # placeholder for total records (used in risk score)
        self.total_records = 1

    # ------------------------------------------------------------------
    # Core hybrid steps
    # ------------------------------------------------------------------
    def process_text(self, text: str) -> dict:
        """
        Executes one hybrid cycle:
        1. Extract regex features → context vector ``c``.
        2. Compute weekday weight vector ``w_day``.
        3. Select an action via softmax bandit.
        4. Evaluate reward using reconstruction risk and VFE.
        5. Update NLMS weights with MinHash‑adjusted learning rate.
        Returns a dictionary with intermediate values for inspection.
        """
        if not self.circuit.allow():
            raise RuntimeError("Circuit breaker open – aborting operation.")

        # 1. Context from regex
        c = extract_regex_features(text)  # shape (2,)
        # Pad / project to state dimension
        if c.size < self.d_state:
            c = np.pad(c, (0, self.d_state - c.size), constant_values=0.0)

        # 2. Weekday weight vector
        today = date.today()
        w_day = weekday_weight_vector(self.groups, today)  # shape (len(groups),)

        # Align dimensions for similarity: repeat or truncate
        if w_day.size < self.d_state:
            w_day = np.pad(w_day, (0, self.d_state - w_day.size), constant_values=0.0)
        elif w_day.size > self.d_state:
            w_day = w_day[: self.d_state]

        # 3. Bandit selection
        logits = bandit_softmax_logits(c, temperature=0.5)
        action = select_action(logits, self.rng)
        # Action vector as one‑hot
        action_vec = np.zeros(self.d_state, dtype=np.float64)
        action_vec[action % self.d_state] = 1.0

        # 4. Reward computation
        uid = f"user_{action}"
        risk = reconstruction_risk_score(uid, self.total_records)
        vfe = variational_free_energy(action_vec, w_day)
        reward = (1.0 - risk) * math.exp(-vfe)  # exp(-VFE) ≈ similarity

        # 5. Learning‑rate adaptation via MinHash
        mh_sig = minhash_signature(c, n_perm=32, seed=42)
        # Use mean of signature (in [0,1]) to scale learning rate
        lr_adj = self.base_lr * (0.5 + mh_sig.mean() / 2.0)

        # Desired output for NLMS: we treat reward as the target scalar
        self.w = nlms_update(self.w, c, reward, lr_adj)

        # Update bookkeeping
        self.total_records += 1
        self.circuit.record_success()

        return {
            "context": c,
            "weekday_weights": w_day,
            "logits": logits,
            "action": action,
            "action_vector": action_vec,
            "risk": risk,
            "vfe": vfe,
            "reward": reward,
            "lr_adj": lr_adj,
            "nlms_weights": self.w.copy(),
        }

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def predict(self, text: str) -> int:
        """
        Returns the most probable action (argmax) without performing learning.
        """
        c = extract_regex_features(text)
        if c.size < self.d_state:
            c = np.pad(c, (0, self.d_state - c.size), constant_values=0.0)
        probs = bandit_softmax_logits(c, temperature=0.5)
        return int(np.argmax(probs))

    def reset(self) -> None:
        """Re‑initialise internal state."""
        self.w = np.zeros(self.d_state, dtype=np.float64)
        self.total_records = 1
        self.circuit = EndpointCircuitBreaker()


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    agent = HybridAgent(seed=123)
    sample_texts = [
        "The plan includes a checklist and a timeline. Evidence is attached as a hash.",
        "We need to verify the source and provide a citation for the facts.",
        "Budget allocation and roadmap are ready; no further audit required.",
    ]

    for i, txt in enumerate(sample_texts, 1):
        result = agent.process_text(txt)
        print(f"--- Cycle {i} ---")
        print(f"Action: {result['action']}, Reward: {result['reward']:.6f}")
        print(f"Adjusted LR: {result['lr_adj']:.6e}")
        print(f"NLMS weights: {result['nlms_weights']}\n")
    # Demonstrate prediction without learning
    pred = agent.predict("Please provide proof and a detailed schedule.")
    print(f"Predicted action (argmax) after training: {pred}")