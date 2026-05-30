# DARWIN HAMMER — match 4219, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_model__m2098_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1535_s3.py (gen6)
# born: 2026-05-29T23:54:20Z

"""
Hybrid Algorithm: Bayesian‑Matrix‑Tropical Fusion

Parents:
- hybrid_hybrid_bayes_claim_k_hybrid_hybrid_model__m2098_s1.py (Bayesian hypothesis
  updates + recurrent weight‑matrix gradient descent)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1535_s3.py (Sparse‑WTA confidence
  scalar, TropicalNetwork evaluation, SSIM comparison)

Mathematical Bridge:
The posterior probability *p* from the Bayesian update is interpreted as a
confidence scalar *c* (0 ≤ c ≤ 1).  This scalar simultaneously:
1. Scales the learning rate of the recurrent weight‑matrix update.
2. Modulates the activation of a TropicalNetwork (ReLU‑like tropical algebra).
3. Weights the Structural‑Similarity (SSIM) measure between the network output
   and a reference signal.

Thus a single scalar links the probabilistic reasoning of Parent A with the
matrix‑algebraic and tropical components of Parent B, yielding a unified hybrid
step."""
import math
import random
import sys
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import List, Dict, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data structures (adapted from both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float
    posterior: float
    evidence_ids: Tuple[str, ...]


@dataclass(frozen=True)
class MathEvidence:
    id: str
    likelihood_ratio: float  # pre‑computed LR for this piece of evidence


@dataclass(frozen=True)
class TropicalNetwork:
    weights: np.ndarray  # shape (n_out, n_in)
    biases: np.ndarray   # shape (n_out,)

    def evaluate(self, input_vector: np.ndarray) -> np.ndarray:
        """Tropical (ReLU‑like) evaluation: max(0, W·x + b) per output unit."""
        linear = self.weights @ input_vector + self.biases
        return np.maximum(0.0, linear)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def bayesian_update(hyp: MathHypothesis, ev: MathEvidence) -> MathHypothesis:
    """
    Perform a Bayesian update of a hypothesis given new evidence.
    The likelihood_ratio must be ≥ 0.
    """
    lr = ev.likelihood_ratio
    if lr < 0:
        raise ValueError("likelihood_ratio must be non‑negative")
    p = max(0.0, min(1.0, hyp.posterior))
    if p <= 0.0 or lr == 0.0:
        posterior = 0.0
    elif p >= 1.0:
        posterior = 1.0
    else:
        odds = p / (1.0 - p)
        new_odds = odds * lr
        posterior = new_odds / (1.0 + new_odds)
    posterior = max(0.0, min(1.0, posterior))
    new_evidence = hyp.evidence_ids + (ev.id,)
    return MathHypothesis(id=hyp.id, prior=hyp.prior, posterior=posterior,
                          evidence_ids=new_evidence)


def sparse_wta_confidence(vector: np.ndarray, top_k: int) -> float:
    """
    Sparse expansion + Winner‑Take‑All (WTA) mask.
    Returns a confidence scalar in [0, 1] equal to the mean of the top‑k
    absolute values normalised by the maximum absolute value in the vector.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    abs_vals = np.abs(vector)
    if abs_vals.max() == 0:
        return 0.0
    # Identify top‑k indices
    top_indices = np.argpartition(-abs_vals, top_k - 1)[:top_k]
    top_vals = abs_vals[top_indices]
    confidence = top_vals.mean() / abs_vals.max()
    return float(confidence)


def weighted_ssim(x: np.ndarray, y: np.ndarray, confidence: float,
                 dynamic_range: float = 255.0,
                 k1: float = 0.01, k2: float = 0.03) -> float:
    """
    Compute SSIM between two signals and weight it by a confidence scalar.
    The SSIM formula follows the standard definition.
    """
    if x.shape != y.shape:
        raise ValueError("Input vectors must have the same shape")
    mu1 = x.mean()
    mu2 = y.mean()
    sigma1 = x.std()
    sigma2 = y.std()
    sigma12 = ((x - mu1) * (y - mu2)).mean()

    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2)
    ssim_val = numerator / denominator if denominator != 0 else 0.0
    # Weight by confidence (still bounded in [0,1])
    return max(0.0, min(1.0, ssim_val * confidence))


def gradient_descent_update(W: np.ndarray,
                            grad: np.ndarray,
                            base_lr: float,
                            confidence: float) -> np.ndarray:
    """
    Recurrent weight‑matrix update using gradient descent.
    The effective learning rate is scaled by the confidence scalar.
    """
    if W.shape != grad.shape:
        raise ValueError("Weight matrix and gradient must share shape")
    effective_lr = base_lr * confidence
    return W - effective_lr * grad


# ----------------------------------------------------------------------
# Hybrid step that ties everything together
# ----------------------------------------------------------------------
def hybrid_step(hypothesis: MathHypothesis,
                evidence: MathEvidence,
                W: np.ndarray,
                grad_W: np.ndarray,
                input_vec: np.ndarray,
                network: TropicalNetwork,
                top_k: int = 5,
                base_lr: float = 0.01) -> Tuple[MathHypothesis,
                                                np.ndarray,
                                                np.ndarray,
                                                float]:
    """
    Perform a single hybrid iteration:
    1. Bayesian update → new posterior.
    2. Map posterior to confidence scalar via sparse‑WTA on the gradient.
    3. Update weight matrix with confidence‑scaled learning rate.
    4. Evaluate TropicalNetwork on the input, scaled by confidence.
    5. Compute confidence‑weighted SSIM between network output and a reference
       (here we reuse the input vector as a simple proxy).
    Returns the updated hypothesis, weight matrix, network output, and SSIM.
    """
    # 1. Bayesian update
    updated_hyp = bayesian_update(hypothesis, evidence)

    # 2. Confidence from posterior (treated as probability) combined with sparse WTA
    # Use the gradient vector as the sparse signal.
    confidence_from_posterior = updated_hyp.posterior
    confidence_wta = sparse_wta_confidence(grad_W.ravel(), top_k)
    confidence = confidence_from_posterior * confidence_wta

    # 3. Weight matrix update
    W_new = gradient_descent_update(W, grad_W, base_lr, confidence)

    # 4. Tropical network evaluation (scaled output)
    raw_output = network.evaluate(input_vec)
    scaled_output = raw_output * confidence

    # 5. SSIM between scaled output and input (as reference)
    ssim_score = weighted_ssim(scaled_output, input_vec, confidence)

    return updated_hyp, W_new, scaled_output, ssim_score


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a simple hypothesis
    hypo = MathHypothesis(id="H0", prior=0.5, posterior=0.5, evidence_ids=())

    # Create evidence with a moderate likelihood ratio
    ev = MathEvidence(id="E1", likelihood_ratio=2.0)

    # Random weight matrix and its gradient
    np.random.seed(42)
    W = np.random.randn(4, 4)
    grad_W = np.random.randn(4, 4) * 0.1

    # Input vector for the TropicalNetwork
    input_vec = np.random.rand(4)

    # Simple TropicalNetwork (identity‑like weights)
    net = TropicalNetwork(weights=np.eye(4), biases=np.zeros(4))

    # Run one hybrid step
    updated_hyp, W_new, net_out, ssim_val = hybrid_step(
        hypothesis=hypo,
        evidence=ev,
        W=W,
        grad_W=grad_W,
        input_vec=input_vec,
        network=net,
        top_k=2,
        base_lr=0.05
    )

    # Print results (ensuring no exception)
    print("Updated hypothesis posterior:", updated_hyp.posterior)
    print("Confidence‑scaled weight matrix (first row):", W_new[0])
    print("Network output (scaled):", net_out)
    print("Weighted SSIM:", ssim_val)