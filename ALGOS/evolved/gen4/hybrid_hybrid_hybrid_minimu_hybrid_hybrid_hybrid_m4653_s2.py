# DARWIN HAMMER — match 4653, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s5.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_ternar_m1253_s2.py (gen3)
# born: 2026-05-29T23:57:22Z

import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Iterable, Optional, List, Any
from pathlib import Path

EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    """
    Immutable representation of an epistemic certainty flag.
    """
    label: str
    confidence_bps: int               # basis points, 0..10 000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not (0 <= int(self.confidence_bps) <= 10_000):
            raise ValueError("confidence_bps must be in 0..10000")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class BanditAction:
    """
    Result of a bandit query.
    """
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "HybridEpistemicBandit"


@dataclass(frozen=True)
class BanditUpdate:
    """
    Information required to update the bandit after observing a reward.
    """
    context_id: str
    action_id: str
    reward: float
    propensity: float


class HybridEpistemicBanditRouter:
    """
    A bandit that fuses epistemic certainty flags, a structural‑similarity
    (SSIM) metric on context embeddings, and classic Thompson‑sampling style
    updates.  The integration is deeper than a simple multiplicative factor:
    * SSIM drives an *adaptive learning‑rate schedule*.
    * Certainty weights both the learning‑rate and the posterior variance.
    * Action propensities are derived from a soft‑max over Bayesian‑estimated
      values, encouraging exploration proportional to epistemic confidence.
    """

    def __init__(
        self,
        d_in: int,
        d_out: Optional[int] = None,
        seed: int = 0,
        base_eta: float = 0.01,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1.0,
        store_decay: float = 0.99,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_eta = float(base_eta)
        self.alpha = float(alpha)          # prior pseudo‑counts for rewards
        self.beta = float(beta)            # prior pseudo‑counts for failures
        self.dt = float(dt)
        self.store_decay = float(store_decay)
        self.d_in = int(d_in)
        self.d_out = int(d_out) if d_out is not None else d_in

        # Internal stores
        self._action_stats: Dict[str, Dict[str, float]] = {}   # action_id -> {"wins":…, "plays":…}
        self._context_prototypes: Dict[str, np.ndarray] = {}  # context_id -> embedding vector

    # --------------------------------------------------------------------- #
    # SSIM utilities (supports 1‑D or 2‑D arrays, multi‑channel optional)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _ssim(x: np.ndarray, y: np.ndarray,
              dynamic_range: float = 255.0,
              k1: float = 0.01,
              k2: float = 0.03) -> float:
        """
        Compute the Structural Similarity Index Measure (SSIM) between two
        arrays.  The implementation works for 1‑D vectors as well as 2‑D
        grayscale images.  For higher‑dimensional data the arrays are flattened.
        """
        if x.shape != y.shape:
            raise ValueError("Input arrays must have the same shape")
        if x.ndim > 2:
            x = x.ravel()
            y = y.ravel()
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x = np.sqrt(np.var(x))
        sigma_y = np.sqrt(np.var(y))
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        c1 = (k1 * dynamic_range) ** 2
        c2 = (k2 * dynamic_range) ** 2
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2)
        return float(numerator / denominator)

    # --------------------------------------------------------------------- #
    # Core integration helpers
    # --------------------------------------------------------------------- #
    def _adaptive_eta(self,
                      ssim_val: float,
                      certainty: CertaintyFlag) -> float:
        """
        Compute an adaptive learning rate that respects both structural
        similarity and epistemic confidence.
        """
        confidence = certainty.confidence_bps / 10_000.0
        # Exponential scaling gives more dynamic range for high similarity
        eta = self.base_eta * (ssim_val ** 2) * confidence
        # Clamp to a reasonable interval to avoid vanishing/ exploding updates
        return max(min(eta, 1.0), 1e-6)

    def _softmax(self, values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Numerically stable soft‑max."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        shifted = values - np.max(values)
        exps = np.exp(shifted / temperature)
        return exps / np.sum(exps)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def register_context(self,
                         context_id: str,
                         embedding: np.ndarray) -> None:
        """
        Store a prototype embedding for a context.  If the context already
        exists, the stored prototype is exponentially decayed towards the new
        embedding, providing a simple form of continual learning.
        """
        if embedding.shape != (self.d_in,):
            raise ValueError(f"embedding must be of shape ({self.d_in},)")
        if context_id in self._context_prototypes:
            old = self._context_prototypes[context_id]
            self._context_prototypes[context_id] = (
                self.store_decay * old + (1.0 - self.store_decay) * embedding
            )
        else:
            self._context_prototypes[context_id] = embedding.copy()

    def get_action(self,
                   context_id: str,
                   certainty: CertaintyFlag,
                   candidate_actions: Optional[Iterable[str]] = None) -> BanditAction:
        """
        Return a BanditAction for the supplied context.  If `candidate_actions`
        is omitted, a single synthetic action is generated.
        """
        # -----------------------------------------------------------------
        # 1. Retrieve or synthesize a context prototype
        # -----------------------------------------------------------------
        if context_id not in self._context_prototypes:
            # Initialise with a random prototype (uniform in [0, 255])
            self._context_prototypes[context_id] = self.rng.integers(
                0, 256, size=self.d_in, dtype=np.float32
            )
        prototype = self._context_prototypes[context_id]

        # -----------------------------------------------------------------
        # 2. Compute SSIM between prototype and a fresh random observation.
        #    In a real system this would be the similarity between the current
        #    observation and the stored prototype.
        # -----------------------------------------------------------------
        observation = self.rng.integers(0, 256, size=self.d_in, dtype=np.float32)
        ssim_val = self._ssim(prototype, observation)

        # -----------------------------------------------------------------
        # 3. Adaptive learning rate
        # -----------------------------------------------------------------
        eta = self._adaptive_eta(ssim_val, certainty)

        # -----------------------------------------------------------------
        # 4. Prepare action statistics (wins, plays) for soft‑max
        # -----------------------------------------------------------------
        if candidate_actions is None:
            candidate_actions = [f"action_{context_id}"]
        actions = list(candidate_actions)

        wins = np.array([self._action_stats.get(a, {}).get("wins", 0.0) for a in actions])
        plays = np.array([self._action_stats.get(a, {}).get("plays", 0.0) for a in actions])

        # Bayesian posterior mean of reward (Beta prior)
        posterior_mean = (wins + self.alpha) / (plays + self.alpha + self.beta)

        # Temperature inversely proportional to epistemic confidence:
        temperature = max(0.1, 1.0 - certainty.confidence_bps / 10_000.0)

        propensities = self._softmax(posterior_mean, temperature=temperature)

        # Sample an action according to propensities
        chosen_idx = self.rng.choice(len(actions), p=propensities)
        chosen_action = actions[chosen_idx]
        propensity = float(propensities[chosen_idx])
        expected_reward = float(posterior_mean[chosen_idx])

        # Confidence bound is derived from learning rate and variance estimate
        variance_est = (expected_reward * (1 - expected_reward)) / max(plays[chosen_idx], 1)
        confidence_bound = eta * np.sqrt(variance_est)

        return BanditAction(
            action_id=chosen_action,
            propensity=propensity,
            expected_reward=expected_reward,
            confidence_bound=confidence_bound,
        )

    def update(self,
               update: BanditUpdate,
               certainty: CertaintyFlag,
               observation: Optional[np.ndarray] = None) -> None:
        """
        Incorporate a reward observation.  The learning rate is adapted using
        SSIM between the stored prototype (or the provided observation) and the
        current context embedding.
        """
        # Ensure stats entry exists
        stats = self._action_stats.setdefault(update.action_id, {"wins": 0.0, "plays": 0.0})

        # Update counts
        stats["plays"] += 1
        stats["wins"] += float(update.reward)

        # Optionally refine the context prototype with the supplied observation
        if observation is not None:
            if observation.shape != (self.d_in,):
                raise ValueError(f"observation must be of shape ({self.d_in},)")
            # Compute similarity to current prototype
            prototype = self._context_prototypes.get(update.context_id)
            if prototype is not None:
                ssim_val = self._ssim(prototype, observation)
                eta = self._adaptive_eta(ssim_val, certainty)
                # Blend prototype towards observation proportionally to eta
                self._context_prototypes[update.context_id] = (
                    (1 - eta) * prototype + eta * observation
                )
            else:
                # First observation for this context
                self._context_prototypes[update.context_id] = observation.copy()


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Optional[Iterable[str]] = None,
) -> CertaintyFlag:
    """
    Helper to construct a CertaintyFlag with validation.
    """
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs) if evidence_refs else (),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: Optional[str] = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10_000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody",
        evidence_refs=refs,
    )


if __name__ == "__main__":
    # Demo usage
    router = HybridEpistemicBanditRouter(d_in=64, seed=42)
    ctx_id = "ctx_001"
    # Register a random prototype for the context
    router.register_context(ctx_id, router.rng.integers(0, 256, size=64).astype(np.float32))

    cert = filesystem_observation(sha256="deadbeef", path="/tmp/example.txt")
    action = router.get_action(context_id=ctx_id, certainty=cert, candidate_actions=["a", "b", "c"])
    print(asdict(action))

    # Simulate a reward and update
    upd = BanditUpdate(context_id=ctx_id, action_id=action.action_id, reward=1.0, propensity=action.propensity)
    router.update(upd, cert, observation=router.rng.integers(0, 256, size=64).astype(np.float32))