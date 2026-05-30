# DARWIN HAMMER — match 202, survivor 3
# gen: 3
# parent_a: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s1.py (gen2)
# parent_b: hybrid_bandit_router_honeybee_store_m9_s5.py (gen1)
# born: 2026-05-29T23:27:39Z

import numpy as np
import random
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional


def now_z() -> str:
    """Current UTC timestamp in ISO‑8601 with trailing “Z”."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_context(text: Optional[str]) -> Dict[str, Any]:
    """Parse a JSON string into a dict, returning an empty dict on ``None``."""
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"context must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("context must be a JSON object")
    return value


# ----------------------------------------------------------------------
# Bandit core – a minimal Thompson‑sampling Bernoulli bandit
# ----------------------------------------------------------------------
@dataclass
class BanditAction:
    """Result of an action selection."""

    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str = "thompson_sampling"


@dataclass
class BanditUpdate:
    """Single observation used to update the policy."""

    context_id: str
    action_id: str
    reward: float
    propensity: float = 1.0


class ThompsonBandit:
    """A lightweight Thompson‑sampling bandit for continuous rewards."""

    def __init__(self, actions: List[str], prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self._alpha: Dict[str, float] = {a: prior_alpha for a in actions}
        self._beta: Dict[str, float] = {a: prior_beta for a in actions}
        self._actions = actions

    def sample(self) -> str:
        """Draw a sample from each Beta posterior and return the best action."""
        samples = {a: np.random.beta(self._alpha[a], self._beta[a]) for a in self._actions}
        return max(samples, key=samples.get)

    def update(self, upd: BanditUpdate) -> None:
        """Update the posterior for the given action."""
        # Clip reward to [0,1] because Beta expects a probability‑like observation.
        r = max(0.0, min(1.0, upd.reward))
        self._alpha[upd.action_id] += r
        self._beta[upd.action_id] += 1.0 - r

    def expected(self, action_id: str) -> float:
        """Mean of the Beta posterior for *action_id*."""
        a = self._alpha[action_id]
        b = self._beta[action_id]
        return a / (a + b)


# ----------------------------------------------------------------------
# Store (honeybee‑style) – keeps a bounded “dance” signal
# ----------------------------------------------------------------------
@dataclass
class StoreState:
    """Encapsulates the honeybee‑style store and its derived control signal."""

    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: List[float], outflow: List[float]) -> Tuple[float, float]:
        """Apply the store equation and recompute the dance signal."""
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self._last_delta = delta
        self.level = max(0.0, self.level + self.dt * delta)
        return self.level, delta

    @property
    def dance(self) -> float:
        """Bounded control signal derived from the most recent Δ."""
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))


# ----------------------------------------------------------------------
# Numeric conversion utilities – bridge strings → vectors
# ----------------------------------------------------------------------
def _string_to_vector(s: str, dim: int = 8) -> np.ndarray:
    """
    Convert a short string into a fixed‑size numeric vector.

    The conversion is deterministic: each character contributes its Unicode code
    point, wrapped around *dim* using a simple rolling hash.
    """
    vec = np.zeros(dim, dtype=float)
    for i, ch in enumerate(s):
        idx = i % dim
        vec[idx] += ord(ch)
    # Normalise to roughly the same scale as image pixel values (0‑255)
    if vec.max() > 0:
        vec = (vec / vec.max()) * 255.0
    return vec


# ----------------------------------------------------------------------
# Structural Similarity Index – robust version for 1‑D vectors
# ----------------------------------------------------------------------
def ssim(
    x: np.ndarray,
    y: np.ndarray,
    dynamic_range: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    eps: float = 1e-12,
) -> float:
    """
    Compute a 1‑D SSIM‑like similarity between two vectors.

    The classic SSIM is defined for images; this adaptation works on any
    1‑dimensional numeric arrays.  It returns a value in ``[-1, 1]`` where
    ``1`` means identical.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    sigma_x = np.sqrt(np.mean((x - mu_x) ** 2) + eps)
    sigma_y = np.sqrt(np.mean((y - mu_y) ** 2) + eps)

    cov_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    luminance = (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1)
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x ** 2 + sigma_y ** 2 + c2)
    structure = (cov_xy + c2 / 2) / (sigma_x * sigma_y + c2 / 2)

    return luminance * contrast * structure


# ----------------------------------------------------------------------
# Core hybrid operation
# ----------------------------------------------------------------------
def route_packet(packet: Dict[str, Any], store_state: StoreState) -> Dict[str, Any]:
    """
    Produce a routing dict enriched with the current ``dance`` signal.
    """
    text = str(
        packet.get("text_surface")
        or packet.get("raw_command")
        or packet.get("text")
        or ""
    )
    intent = str(
        packet.get("normalized_intent")
        or packet.get("intent")
        or "bytewax_rete_bandit"
    )
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref"),
        "ontology_terms": packet.get("ontology_terms") or [],
        "epistemic_flag": packet.get("epistemic_flag"),
        "payload": packet.get("payload") or {},
    }

    # The route dict mirrors the input but also carries the dance signal.
    route = {
        "text": text,
        "intent": intent,
        "context": context,
        "dance_signal": store_state.dance,
    }
    return route


def hybrid_operation(
    packet: Dict[str, Any],
    store_state: StoreState,
    bandit: ThompsonBandit,
) -> Dict[str, Any]:
    """
    End‑to‑end hybrid step:

    1. Produce a base route.
    2. Convert textual fields to numeric vectors.
    3. Compute a similarity score with a robust SSIM.
    4. Feed the similarity (scaled to [0,1]) to the bandit.
    5. Update the store using the bandit’s posterior mean as a reward.
    6. Return the enriched route.
    """
    route = route_packet(packet, store_state)

    # ------------------------------------------------------------------
    # 2‑3. Vectorise and compare
    # ------------------------------------------------------------------
    in_vec = np.concatenate(
        [
            _string_to_vector(str(packet.get("text_surface", ""))),
            _string_to_vector(str(packet.get("intent", ""))),
        ]
    )
    out_vec = np.concatenate(
        [_string_to_vector(route["text"]), _string_to_vector(route["intent"])]
    )
    similarity_raw = ssim(in_vec, out_vec)

    # Normalise similarity to [0,1] for the bandit (SSIM can be slightly <0)
    similarity = (similarity_raw + 1.0) / 2.0
    similarity = max(0.0, min(1.0, similarity))

    # ------------------------------------------------------------------
    # 4‑5. Bandit update and store reward
    # ------------------------------------------------------------------
    chosen_action = bandit.sample()
    upd = BanditUpdate(
        context_id="hybrid_context",
        action_id=chosen_action,
        reward=similarity,
    )
    bandit.update(upd)

    # Use the posterior mean as a smoother reward signal for the store.
    reward_for_store = bandit.expected(chosen_action)
    store_state.update([reward_for_store], [])

    # Attach diagnostics
    route.update(
        {
            "similarity_raw": similarity_raw,
            "similarity_norm": similarity,
            "bandit_action": chosen_action,
            "bandit_expected_reward": reward_for_store,
        }
    )
    return route


# ----------------------------------------------------------------------
# Example usage (executed only when run as a script)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    example_packet = {
        "text_surface": "Hello",
        "intent": "greeting",
        "source": "demo",
    }

    # Initialise components
    store = StoreState()
    bandit = ThompsonBandit(actions=["hybrid_action"])

    # Run a single hybrid step
    result = hybrid_operation(example_packet, store, bandit)

    # Pretty‑print the result
    print(json.dumps(result, indent=2, ensure_ascii=False))