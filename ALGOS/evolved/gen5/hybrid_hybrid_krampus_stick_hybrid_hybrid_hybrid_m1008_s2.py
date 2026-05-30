# DARWIN HAMMER — match 1008, survivor 2
# gen: 5
# parent_a: hybrid_krampus_stickers_hybrid_pheromone_inf_m107_s1.py (gen2)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_capybara_opti_m32_s0.py (gen4)
# born: 2026-05-29T23:32:21Z

import numpy as np
import random
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Optional

Vector = List[float]


class PheromoneEntry:
    __slots__ = (
        "uuid",
        "surface_key",
        "signal_kind",
        "signal_value",
        "half_life_seconds",
        "created_at",
        "last_decay",
    )

    def __init__(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
    ) -> None:
        self.uuid = f"{random.getrandbits(128):032x}"
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = float(signal_value)
        self.half_life_seconds = max(1, half_life_seconds)  # avoid division by zero
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    # --------------------------------------------------------------------- #
    # Decay utilities
    # --------------------------------------------------------------------- #
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Thread‑safe global store for pheromone entries."""

    _entries: Dict[str, PheromoneEntry] = {}
    _lock = threading.RLock()

    @classmethod
    def add(cls, entry: PheromoneEntry) -> None:
        with cls._lock:
            cls._entries[entry.uuid] = entry

    @classmethod
    def get_by_surface(cls, surface_key: str) -> List[PheromoneEntry]:
        with cls._lock:
            return [
                e for e in cls._entries.values() if e.surface_key == surface_key
            ]

    @classmethod
    def decay_surface(cls, surface_key: str) -> List[Dict[str, float]]:
        """Apply decay to all entries of a surface and return a log."""
        rows = []
        for entry in cls.get_by_surface(surface_key):
            before = entry.signal_value
            entry.apply_decay()
            rows.append(
                {"surface_key": surface_key, "uuid": entry.uuid, "before": before, "after": entry.signal_value}
            )
        return rows

    @classmethod
    def prune_near_zero(cls, threshold: float = 1e-8) -> None:
        """Remove entries whose signal value fell below *threshold* after decay."""
        with cls._lock:
            dead = [uid for uid, e in cls._entries.items() if e.signal_value < threshold]
            for uid in dead:
                del cls._entries[uid]


# --------------------------------------------------------------------- #
# Core fused functions
# --------------------------------------------------------------------- #


def _seed_random(seed: Optional[int | str]) -> None:
    """Deterministic seeding helper."""
    if seed is None:
        return
    if isinstance(seed, str):
        seed = abs(hash(seed)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def social_interaction_and_pheromone(
    text: str,
    g_best: Vector,
    k: int = 1,
    r1: Optional[float] = None,
    seed: Optional[int | str] = None,
) -> List[float]:
    """
    Computes a pheromone value from lexical statistics, then blends it
    with the global best vector *g_best* using a configurable scaling factor.
    The result is stored in the global :class:`PheromoneStore`.
    """
    _seed_random(seed)

    words = text.split()
    if not words:
        return []

    # ----------------------------------------------------------------- #
    # Lexical statistics → base pheromone
    # ----------------------------------------------------------------- #
    counts = Counter(words)
    freqs = np.array([counts[w] / len(words) for w in counts])
    base_pheromone = freqs.mean()

    # ----------------------------------------------------------------- #
    # Fusion with optimisation meta‑data (g_best)
    # ----------------------------------------------------------------- #
    g_arr = np.array(g_best, dtype=float)
    if r1 is None:
        r1 = random.random()
    # Weighted combination: keep the lexical signal but bias it toward the
    # direction indicated by g_best.
    fused = base_pheromone * (1 - r1) + r1 * (g_arr.mean() if g_arr.size else base_pheromone)

    # ----------------------------------------------------------------- #
    # Store the fused pheromone
    # ----------------------------------------------------------------- #
    entry = PheromoneEntry(
        surface_key="social_interaction",
        signal_kind="lexical_fused",
        signal_value=float(fused),
        half_life_seconds=3600,
    )
    PheromoneStore.add(entry)

    # Return a per‑word contribution scaled by the fused value.
    return (freqs * fused).tolist()


def stylometry_analysis_with_pheromone(text: str, surface_key: str = "stylometry") -> float:
    """
    Performs a lightweight stylometric score and enriches it with the
    most recent pheromone value for *surface_key*.
    """
    words = text.split()
    if not words:
        return 0.0

    counts = Counter(words)
    freqs = np.array([counts[w] / len(words) for w in counts])
    stylometric_score = freqs.mean()

    # Pull the freshest pheromone for the given surface (if any)
    entries = PheromoneStore.get_by_surface(surface_key)
    if entries:
        # Apply decay before reading
        for e in entries:
            e.apply_decay()
        latest = max(entries, key=lambda e: e.last_decay)
        pheromone_factor = latest.signal_value
    else:
        pheromone_factor = 1.0  # neutral factor when no pheromone exists

    return float(stylometric_score * pheromone_factor)


def predator_evasion_with_pheromone_and_signal_scores(
    data: bytes,
    text: str,
    status_code: Optional[int] = None,
    mime: str = "",
    keyword_hits: int = 0,
    structural_links: int = 0,
    surface_key: str = "predator_evasion",
) -> Tuple[float, List[float]]:
    """
    Combines a low‑level byte‑level signal with stylometric pheromones to
    produce an evasion score.
    """
    if not data:
        byte_signal = 0.0
    else:
        # ``data`` is an iterable of ints (0‑255); use them directly.
        byte_signal = np.mean([b for b in data])

    # Enrich the byte signal with stylometric pheromone
    pheromone_score = stylometry_analysis_with_pheromone(text, surface_key=surface_key)

    # Core evasion metric
    evasion = pheromone_score * (keyword_hits + structural_links) * (byte_signal / 255.0)

    # Return detailed breakdown for possible downstream use
    breakdown = [
        pheromone_score * keyword_hits,
        pheromone_score * structural_links,
        byte_signal,
    ]
    return float(evasion), [float(v) for v in breakdown]


def hybrid_fusion(
    text: str,
    data: bytes,
    g_best: Vector,
    k: int = 1,
    r1: Optional[float] = None,
    seed: Optional[int | str] = None,
    keyword_hits: int = 0,
    structural_links: int = 0,
) -> Dict[str, any]:
    """
    High‑level orchestrator that demonstrates a deeper mathematical
    integration of the three constituent mechanisms.
    """
    # 1️⃣ Social interaction → pheromone creation
    social_vals = social_interaction_and_pheromone(
        text, g_best, k=k, r1=r1, seed=seed
    )

    # 2️⃣ Stylometry enriched by the freshly created pheromone
    stylometric = stylometry_analysis_with_pheromone(text, surface_key="social_interaction")

    # 3️⃣ Predator evasion that consumes both byte‑level and stylometric signals
    evasion_score, evasion_breakdown = predator_evasion_with_pheromone_and_signal_scores(
        data,
        text,
        keyword_hits=keyword_hits,
        structural_links=structural_links,
        surface_key="social_interaction",
    )

    # 4️⃣ Optional housekeeping: decay all surfaces and prune
    for surface in {"social_interaction", "stylometry", "predator_evasion"}:
        PheromoneStore.decay_surface(surface)
    PheromoneStore.prune_near_zero()

    return {
        "social_values": social_vals,
        "stylometric_score": stylometric,
        "evasion_score": evasion_score,
        "evasion_breakdown": evasion_breakdown,
        "pheromone_snapshot": {
            s: [
                {
                    "uuid": e.uuid,
                    "value": e.signal_value,
                    "age_s": e.age_seconds(),
                }
                for e in PheromoneStore.get_by_surface(s)
            ]
            for s in {"social_interaction", "stylometry", "predator_evasion"}
        },
    }


if __name__ == "__main__":
    sample_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua."
    )
    sample_data = b"example payload for predator evasion"
    best_vector = [0.42, 0.73, 0.15]

    result = hybrid_fusion(
        text=sample_text,
        data=sample_data,
        g_best=best_vector,
        k=2,
        r1=0.6,
        seed=12345,
        keyword_hits=3,
        structural_links=2,
    )
    print(result)