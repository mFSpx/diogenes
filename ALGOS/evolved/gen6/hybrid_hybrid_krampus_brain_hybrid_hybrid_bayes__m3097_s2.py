# DARWIN HAMMER — match 3097, survivor 2
# gen: 6
# parent_a: hybrid_krampus_brainmap_hybrid_hybrid_endpoi_m1498_s1.py (gen5)
# parent_b: hybrid_hybrid_bayes_claim_k_hybrid_hybrid_model__m2098_s0.py (gen4)
# born: 2026-05-29T23:47:51Z

import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np


class MathEvidence:
    """Simple container for evidence used in Bayesian updates."""

    def __init__(self, id: str, value: float):
        """
        Parameters
        ----------
        id : str
            Unique identifier of the evidence item.
        value : float
            A numeric observation in the range [0, 1] that can be interpreted
            as the likelihood of the hypothesis being true given this evidence.
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError("evidence value must be in [0, 1]")
        self.id = id
        self.value = value


class MathHypothesis:
    """Representation of a hypothesis with Bayesian priors/posteriors."""

    def __init__(
        self,
        id: str,
        prior: float,
        posterior: float = None,
        evidence_ids: List[str] | Tuple[str, ...] = None,
    ):
        if not (0.0 <= prior <= 1.0):
            raise ValueError("prior must be in [0, 1]")
        self.id = id
        self.prior = prior
        # If no posterior supplied, start with the prior
        self.posterior = prior if posterior is None else posterior
        self.evidence_ids = list(evidence_ids) if evidence_ids else []

    def __repr__(self) -> str:
        return (
            f"MathHypothesis(id={self.id!r}, prior={self.prior:.4f}, "
            f"posterior={self.posterior:.4f}, evidence_ids={self.evidence_ids})"
        )


class EndpointCircuitBreaker:
    """
    Circuit‑breaker that tracks consecutive failures and can be queried
    for its open/closed status.  When open, it can optionally dampen the
    impact of new evidence on the hypothesis.
    """

    def __init__(self, failure_threshold: int = 3, damp_factor: float = 0.1):
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if not (0.0 <= damp_factor <= 1.0):
            raise ValueError("damp_factor must be in [0, 1]")
        self.failure_threshold = failure_threshold
        self.damp_factor = damp_factor
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_success(self) -> None:
        """Reset failure counter and close the breaker."""
        self.failures = 0
        self.open = False
        self.last_event_at = ""

    def record_failure(self) -> None:
        """Increment failure counter and open the breaker if needed."""
        self.failures += 1
        self.open = self.failures >= self.failure_threshold
        self.last_event_at = ""

    def allow(self) -> bool:
        """True if the circuit is closed (i.e. endpoint is usable)."""
        return not self.open

    def _effective_likelihood_ratio(self, lr: float) -> float:
        """
        When the breaker is open we damp the likelihood ratio so that
        evidence cannot dramatically swing the posterior.
        """
        if self.open:
            return 1.0 + self.damp_factor * (lr - 1.0)
        return lr

    def update_hypothesis(
        self,
        hypothesis: MathHypothesis,
        evidence: MathEvidence,
        likelihood_ratio: float,
    ) -> MathHypothesis:
        """
        Perform a Bayesian update using a likelihood ratio that is
        optionally damped by the circuit‑breaker state.

        Parameters
        ----------
        hypothesis : MathHypothesis
            The hypothesis to be updated.
        evidence : MathEvidence
            New evidence that informs the update.
        likelihood_ratio : float
            Ratio P(evidence|H) / P(evidence|¬H).  Must be non‑negative.

        Returns
        -------
        MathHypothesis
            A new hypothesis instance with updated posterior and
            appended evidence identifier.
        """
        if likelihood_ratio < 0:
            raise ValueError("likelihood_ratio must be non‑negative")

        # Apply damping if the breaker is open
        lr_eff = self._effective_likelihood_ratio(likelihood_ratio)

        # Guard against degenerate cases
        p = max(0.0, min(1.0, hypothesis.posterior))
        if p == 0.0 or lr_eff == 0.0:
            posterior = 0.0
        elif p == 1.0:
            posterior = 1.0
        else:
            odds = p / (1.0 - p)
            new_odds = odds * lr_eff
            posterior = new_odds / (1.0 + new_odds)

        posterior = max(0.0, min(1.0, posterior))

        # Record the evidence
        new_evidence_ids = hypothesis.evidence_ids + [evidence.id]

        return MathHypothesis(
            id=hypothesis.id,
            prior=hypothesis.posterior,
            posterior=posterior,
            evidence_ids=new_evidence_ids,
        )

    def prune_probability(self, t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
        """
        Probability of pruning a routing decision at time ``t``.
        If the breaker is open we prune deterministically.
        """
        if self.open:
            return 1.0
        return 1.0 / (1.0 + np.exp(-(t - alpha) / lam))


def compute_likelihood_ratio(evidence: MathEvidence, hypothesis: MathHypothesis) -> float:
    """
    Derive a likelihood ratio from the evidence value.
    For a binary hypothesis we treat ``evidence.value`` as
    P(evidence | H) and (1‑value) as P(evidence | ¬H).

    Returns
    -------
    float
        Non‑negative likelihood ratio.
    """
    p_e_given_h = evidence.value
    p_e_given_not_h = 1.0 - evidence.value
    # Avoid division by zero
    if p_e_given_not_h == 0.0:
        return float("inf")
    return p_e_given_h / p_e_given_not_h


def update_hypothesis_and_circuitbreaker(
    hypothesis: MathHypothesis,
    evidence: MathEvidence,
    circuitbreaker: EndpointCircuitBreaker,
    t: float,
    lam: float = 1.0,
    alpha: float = 0.2,
) -> Tuple[MathHypothesis, EndpointCircuitBreaker]:
    """
    Integrate Bayesian updating with circuit‑breaker logic.

    Parameters
    ----------
    hypothesis : MathHypothesis
        Current hypothesis.
    evidence : MathEvidence
        New evidence to incorporate.
    circuitbreaker : EndpointCircuitBreaker
        Existing breaker instance (preserves state across calls).
    t : float
        Current time step (used for pruning probability).

    Returns
    -------
    Tuple[MathHypothesis, EndpointCircuitBreaker]
        Updated hypothesis and the (potentially mutated) circuit‑breaker.
    """
    # Compute a principled likelihood ratio from the evidence
    lr = compute_likelihood_ratio(evidence, hypothesis)

    # Perform the Bayesian update, respecting the breaker state
    updated_hypothesis = circuitbreaker.update_hypothesis(hypothesis, evidence, lr)

    # Example of using the pruning probability (could be used by routing logic)
    _ = circuitbreaker.prune_probability(t, lam, alpha)  # placeholder for side‑effects

    return updated_hypothesis, circuitbreaker


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """
    Gaussian kernel used elsewhere in the system.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    return math.exp(-((theta - center) ** 2) / (2 * width ** 2))


# ----------------------------------------------------------------------
# Example usage (can be removed or replaced by unit tests)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise a hypothesis and a circuit‑breaker
    hyp = MathHypothesis(id="H1", prior=0.6)
    cb = EndpointCircuitBreaker(failure_threshold=2, damp_factor=0.2)

    # Simulated evidence stream
    evidences = [
        MathEvidence(id="E1", value=0.8),
        MathEvidence(id="E2", value=0.3),
        MathEvidence(id="E3", value=0.9),
    ]

    for step, ev in enumerate(evidences, start=1):
        # Randomly decide whether to record a failure for demonstration
        if random.random() < 0.3:
            cb.record_failure()
        else:
            cb.record_success()

        hyp, cb = update_hypothesis_and_circuitbreaker(
            hypothesis=hyp,
            evidence=ev,
            circuitbreaker=cb,
            t=step * 0.5,
        )
        print(f"Step {step}: {hyp}, circuit open={not cb.allow()}")