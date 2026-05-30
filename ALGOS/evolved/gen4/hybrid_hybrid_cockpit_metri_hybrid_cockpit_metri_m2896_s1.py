# DARWIN HAMMER — match 2896, survivor 1
# gen: 4
# parent_a: hybrid_cockpit_metrics_hybrid_hybrid_ternar_m229_s0.py (gen3)
# parent_b: hybrid_cockpit_metrics_hybrid_hybrid_ternar_m229_s3.py (gen3)
# born: 2026-05-29T23:46:35Z

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np

Vector = Sequence[float]


def utc_now() -> str:
    """Current UTC timestamp in ISO‑8601 format without microseconds."""
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a JSON manifest from *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims; clipped to [0,1]."""
    if total_claims_emitted <= 0:
        return 1.0
    return max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Proportion of displayed items that are verified as OK; clipped to [0,1]."""
    total = displayed_ok + unknown_displayed_as_ok
    if total <= 0:
        return 1.0
    return max(0.0, min(1.0, displayed_ok / total))


def social_interaction(
    x: Vector,
    g_best: Vector,
    *,
    k: int = 1,
    r: float | None = None,
    seed: int | str | None = None,
) -> np.ndarray:
    """
    Compute a stochastic interaction vector between the current position ``x`` and
    the global best ``g_best``.  The formula is a slight generalisation of the
    classic PSO velocity update:

        v_i = x_i + r * (g_i - k * x_i)

    Parameters
    ----------
    x, g_best:
        Same‑length sequences representing particle position and global best.
    k:
        Contraction factor (normally 1 or 2).  Values outside this range raise
        ``ValueError``.
    r:
        Random coefficient in ``[0, 1]``.  If ``None`` a reproducible random
        number is drawn using ``seed``.
    seed:
        Seed for the internal RNG; ignored if ``r`` is supplied.

    Returns
    -------
    np.ndarray
        Interaction vector of shape ``(len(x),)``.
    """
    if len(x) != len(g_best):
        raise ValueError("x and g_best must have the same dimension")
    if k not in (1, 2):
        raise ValueError("k must be 1 or 2")
    rng = random.Random(seed)
    r_val = r if r is not None else rng.random()
    if not (0.0 <= r_val <= 1.0):
        raise ValueError("r must be in the interval [0, 1]")
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(g_best, dtype=float)
    return x_arr + r_val * (g_arr - k * x_arr)


def evasion_delta(t: int, t_max: int, *, delta_max: float = 1.0, alpha: float = 0.1) -> float:
    """
    Decaying factor that reduces pruning aggressiveness as the algorithm
    approaches its time horizon.

    Parameters
    ----------
    t, t_max:
        Current iteration and maximum iteration count.  ``t_max`` must be > 0.
    delta_max:
        Upper bound of the decay factor (default 1.0).
    alpha:
        Shape parameter of the decay curve (default 0.1).

    Returns
    -------
    float
        Decay factor in ``[0, delta_max]``.
    """
    if t_max <= 0:
        raise ValueError("t_max must be a positive integer")
    if t < 0:
        raise ValueError("t must be non‑negative")
    progress = min(1.0, max(0.0, t / t_max))
    return delta_max * (1.0 - progress) ** alpha


@dataclass(frozen=True)
class FusionConfig:
    """
    Configuration for the hybrid pruning schedule.  The weights control the
    relative influence of each metric; they must sum to 1 for a convex blend.
    """

    w_honesty: float = 0.25
    w_slop: float = 0.25
    w_interaction: float = 0.30
    w_delta: float = 0.20
    k_interaction: int = 1
    seed: int | str | None = None

    def __post_init__(self) -> None:
        total = self.w_honesty + self.w_slop + self.w_interaction + self.w_delta
        if not np.isclose(total, 1.0):
            raise ValueError("Weights must sum to 1.0")
        if not (0.0 <= self.w_honesty <= 1.0):
            raise ValueError("w_honesty out of bounds")
        if not (0.0 <= self.w_slop <= 1.0):
            raise ValueError("w_slop out of bounds")
        if not (0.0 <= self.w_interaction <= 1.0):
            raise ValueError("w_interaction out of bounds")
        if not (0.0 <= self.w_delta <= 1.0):
            raise ValueError("w_delta out of bounds")
        if self.k_interaction not in (1, 2):
            raise ValueError("k_interaction must be 1 or 2")


def hybrid_pruning_schedule(
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    x: Vector,
    g_best: Vector,
    t: int,
    t_max: int,
    *,
    cfg: FusionConfig,
) -> Tuple[float, np.ndarray]:
    """
    Compute a pruning probability that fuses honesty, slop ratio,
    stochastic social interaction, and a time‑based decay.

    Returns
    -------
    schedule:
        Scalar pruning probability in ``[0, 1]``.
    interaction:
        The interaction vector used in the computation (cached to avoid
        recomputation by callers).
    """
    # Core metrics
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    slop = anti_slop_ratio(claims_with_evidence, total_claims_emitted)

    # Interaction vector – deterministic given the config seed
    interaction = social_interaction(
        x,
        g_best,
        k=cfg.k_interaction,
        seed=cfg.seed,
    )
    # Normalise interaction magnitude to [0,1] for blending
    interaction_norm = np.linalg.norm(interaction)
    max_norm = np.sqrt(len(x)) * np.max(np.abs(interaction)) if interaction.size else 1.0
    interaction_score = interaction_norm / max_norm if max_norm > 0 else 0.0

    # Time‑dependent decay
    delta = evasion_delta(t, t_max)

    # Weighted convex combination
    schedule = (
        cfg.w_honesty * honesty
        + cfg.w_slop * slop
        + cfg.w_interaction * interaction_score
        + cfg.w_delta * delta
    )
    # Clamp to [0,1] for safety
    schedule = max(0.0, min(1.0, schedule))
    return schedule, interaction


def fused_hybrid_algorithm(
    x: Vector,
    g_best: Vector,
    claims_with_evidence: int,
    total_claims_emitted: int,
    displayed_ok: int,
    unknown_displayed_as_ok: int,
    t: int,
    t_max: int,
    *,
    cfg: FusionConfig | None = None,
) -> Tuple[float, np.ndarray]:
    """
    Public entry point that returns the pruning schedule and the underlying
    interaction vector.  ``cfg`` may be omitted to use the default weighting.
    """
    cfg = cfg or FusionConfig()
    schedule, interaction = hybrid_pruning_schedule(
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        x,
        g_best,
        t,
        t_max,
        cfg=cfg,
    )
    return schedule, interaction


def smoke_test_hybrid() -> None:
    """Simple sanity check that runs the fused algorithm with representative data."""
    x = [1.0, 2.0, 3.0]
    g_best = [2.0, 3.0, 4.0]
    claims_with_evidence = 10
    total_claims_emitted = 20
    displayed_ok = 15
    unknown_displayed_as_ok = 5
    t = 10
    t_max = 100

    cfg = FusionConfig(
        w_honesty=0.30,
        w_slop=0.20,
        w_interaction=0.35,
        w_delta=0.15,
        k_interaction=1,
        seed=42,
    )

    schedule, interaction = fused_hybrid_algorithm(
        x,
        g_best,
        claims_with_evidence,
        total_claims_emitted,
        displayed_ok,
        unknown_displayed_as_ok,
        t,
        t_max,
        cfg=cfg,
    )
    print(f"Schedule: {schedule:.4f}")
    print(f"Interaction vector: {interaction}")


if __name__ == "__main__":
    smoke_test_hybrid()