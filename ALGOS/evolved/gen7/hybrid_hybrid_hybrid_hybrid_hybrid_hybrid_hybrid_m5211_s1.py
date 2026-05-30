# DARWIN HAMMER — match 5211, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_cockpi_m1465_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2453_s1.py (gen6)
# born: 2026-05-30T00:00:47Z

import os
import re
import json
import math
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Configuration & environment helpers
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path.cwd()
RUNTIME_DIR = ROOT / "04_RUNTIME" / "inference_os"

DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))
BASE_LEARNING_RATE = 0.05
CIRCUIT_BREAKER_THRESHOLD = 0.7
CIRCUIT_BREAKER_HYSTERESIS = 0.05  # prevents flapping


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def clamp01(x: float) -> float:
    """Clamp a float to the interval [0, 1]."""
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------
# Text processing utilities
# ----------------------------------------------------------------------
WORD_RE = re.compile(r"\b[a-zA-Z]+\b")


def words(text: Optional[str]) -> List[str]:
    """Return a list of lower‑cased alphabetic tokens."""
    if not text:
        return []
    return WORD_RE.findall(text.lower())


FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": {"a", "an", "the"},
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}


def stylometry_feature_extraction(text: str) -> Dict[str, float]:
    """Extract a lightweight stylometry vector from *text*."""
    tokens = words(text)
    n = len(tokens) or 1  # avoid division by zero
    features: Dict[str, float] = {
        "token_count": n,
        "unique_token_ratio": len(set(tokens)) / n,
    }
    for cat, vocab in FUNCTION_CATS.items():
        features[f"freq_{cat}"] = sum(1 for t in tokens if t in vocab) / n
    return features


# ----------------------------------------------------------------------
# Core hybrid logic
# ----------------------------------------------------------------------
class HybridEngine:
    """
    Maintains state required for the regret‑aware, VRAM‑aware, and
    Fisher‑score‑driven hybrid algorithm.
    """

    def __init__(self,
                 budget_mb: int = DEFAULT_BUDGET_MB,
                 reserve_mb: int = DEFAULT_RESERVE_MB,
                 base_lr: float = BASE_LEARNING_RATE):
        self.budget_mb = budget_mb
        self.reserve_mb = reserve_mb
        self.base_lr = base_lr

        # Historical data for regret and Fisher‑score computation
        self.loss_history: List[float] = []          # cumulative loss per step
        self.text_history: List[Dict[str, float]] = []  # stylometry vectors
        self.circuit_breaker_active: bool = False

    # ------------------------------------------------------------------
    # VRAM helpers
    # ------------------------------------------------------------------
    def vram_utilization(self, free_mb: int) -> float:
        """
        Return the proportion of *free_mb* relative to the usable budget.
        The usable budget excludes the reserved safety margin.
        """
        usable = max(self.budget_mb - self.reserve_mb, 1)
        return clamp01(free_mb / usable)

    # ------------------------------------------------------------------
    # Regret & learning‑rate handling
    # ------------------------------------------------------------------
    def update_regret(self, loss: float) -> float:
        """
        Update internal loss history and return the *instantaneous regret*.
        Regret is defined as the excess loss over the best observed loss so far.
        """
        self.loss_history.append(loss)
        best_so_far = min(self.loss_history)
        regret = loss - best_so_far
        return regret

    def adaptive_learning_rate(self, regret: float) -> float:
        """
        Scale the base learning rate according to regret.
        Larger regret → more aggressive updates, but capped to avoid explosion.
        """
        scale = 1.0 + math.tanh(regret)  # smooth growth, bounded in (0,2)
        return clamp01(self.base_lr * scale)

    # ------------------------------------------------------------------
    # Trust‑weighted velocity field
    # ------------------------------------------------------------------
    def trust_weighted_velocity(self,
                                free_vram_mb: int,
                                current_text: str,
                                previous_text: Optional[str] = None) -> float:
        """
        Compute a velocity term that blends:
        * VRAM availability (more free memory → higher trust)
        * Text similarity (high similarity → lower velocity, i.e., more cautious)
        * Adaptive learning rate derived from regret.
        """
        # 1️⃣ VRAM trust factor
        vram_trust = self.vram_utilization(free_vram_mb)

        # 2️⃣ Text similarity (Jaccard over token sets)
        if previous_text:
            cur_set = set(words(current_text))
            prev_set = set(words(previous_text))
            intersect = len(cur_set & prev_set)
            union = len(cur_set | prev_set) or 1
            similarity = intersect / union
        else:
            similarity = 0.0

        # 3️⃣ Regret‑driven learning rate
        # For demo purposes we treat a synthetic loss = 1 - vram_trust
        synthetic_loss = 1.0 - vram_trust
        regret = self.update_regret(synthetic_loss)
        lr = self.adaptive_learning_rate(regret)

        # Combine: higher trust & lower similarity boost velocity,
        # while learning‑rate scales magnitude.
        trust_component = vram_trust * (1.0 - similarity)
        return lr * trust_component

    # ------------------------------------------------------------------
    # Fisher‑score driven endpoint circuit breaker
    # ------------------------------------------------------------------
    def _fisher_score(self) -> float:
        """
        Approximate a Fisher‑score‑like statistic over the stored stylometry vectors.
        We treat the mean vector as class‑1 and the overall variance as class‑2.
        The score is the ratio of between‑class variance to within‑class variance.
        """
        if len(self.text_history) < 5:
            return 0.0  # insufficient data

        data = np.array([list(v.values()) for v in self.text_history])
        mean_vec = data.mean(axis=0)
        overall_var = data.var(axis=0, ddof=1).mean()

        # Simulate a second class by shuffling the rows
        shuffled = np.random.permutation(data)
        mean_shuffled = shuffled.mean(axis=0)
        between_var = ((mean_vec - mean_shuffled) ** 2).mean()

        # Avoid division by zero
        if overall_var == 0:
            return 0.0
        return between_var / overall_var

    def endpoint_circuit_breaker(self,
                                 free_vram_mb: int,
                                 current_text: str) -> bool:
        """
        Decide whether to trip the circuit breaker.
        The breaker activates when:
        * Fisher score exceeds a threshold **and**
        * VRAM utilisation falls below a safety margin.
        Hysteresis prevents rapid toggling.
        """
        # Record current stylometry for future Fisher computation
        self.text_history.append(stylometry_feature_extraction(current_text))
        if len(self.text_history) > 50:
            self.text_history.pop(0)  # keep a bounded window

        fisher = self._fisher_score()
        vram_trust = self.vram_utilization(free_vram_mb)

        # Breaker logic with hysteresis
        if not self.circuit_breaker_active:
            trigger = fisher > CIRCUIT_BREAKER_THRESHOLD and vram_trust < 0.3
        else:
            trigger = fisher > (CIRCUIT_BREAKER_THRESHOLD - CIRCUIT_BREAKER_HYSTERESIS) and vram_trust < 0.35

        self.circuit_breaker_active = trigger
        return trigger

    # ------------------------------------------------------------------
    # Public API – a single cohesive update step
    # ------------------------------------------------------------------
    def hybrid_update_step(self,
                           free_vram_mb: int,
                           current_text: str,
                           previous_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute one hybrid iteration and return a diagnostic dictionary.
        """
        velocity = self.trust_weighted_velocity(free_vram_mb, current_text, previous_text)
        breaker = self.endpoint_circuit_breaker(free_vram_mb, current_text)

        diagnostics = {
            "timestamp": now_z(),
            "free_vram_mb": free_vram_mb,
            "vram_utilization": self.vram_utilization(free_vram_mb),
            "trust_weighted_velocity": velocity,
            "circuit_breaker_active": breaker,
            "stylometry": self.text_history[-1],
            "loss_history": self.loss_history[-5:],  # recent slice for inspection
        }
        return diagnostics


# ----------------------------------------------------------------------
# Example driver (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    engine = HybridEngine()
    free_vram = 1024  # Example free memory in MB
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A different sentence, with some overlapping words.",
        "Yet another example to test the hybrid engine's behaviour.",
    ]

    prev = None
    for txt in texts:
        diag = engine.hybrid_update_step(free_vram, txt, prev)
        print(json.dumps(diag, indent=2))
        prev = txt
        # Simulate VRAM consumption change
        free_vram = max(0, free_vram - random.randint(50, 150))