# DARWIN HAMMER — match 2021, survivor 0
# gen: 5
# parent_a: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py (gen2)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s0.py (gen4)
# born: 2026-05-29T23:40:34Z

"""Hybrid Fisher‑SSIM‑Bandit Algorithm
Parent A: hybrid_fisher_localization_hybrid_ternary_route_m40_s0.py
Parent B: hybrid_hybrid_hybrid_bandit_hybrid_hybrid_hybrid_m185_s0.py

Mathematical Bridge
-------------------
The Fisher‑information score (A) quantifies the sharpness of a Gaussian‑shaped
“text surface”.  The Structural Similarity Index (SSIM, A) measures similarity
between the packet text and a reference sample.  The Schoolfield temperature‑
dependent activity curve (B) provides a scalar factor 𝜙(T) that modulates rates
as a function of ambient temperature.

We fuse the two by defining a *temperature‑modulated hybrid score*  

    H = 𝜙(T) · F(θ; μ, σ) · SSIM(x, y)

where  
* 𝜙(T) = developmental_rate(K) from the Schoolfield model,  
* F(θ; μ, σ) = fisher_score from the Gaussian beam,  
* SSIM(x, y) = structural similarity between the packet text (x) and a
  reference text (y).

The hybrid score H is then used as the instantaneous reward for a
contextual bandit update, yielding a temperature‑aware learning rule that
adapts routing decisions based on both information geometry and perceptual
similarity.  The following module implements this unified system."""

import math
import random
import sys
import pathlib
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Core primitives from Parent A
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity I(θ)."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) for 1‑D grayscale signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.cov(x, y, ddof=0)[0, 1]
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    numerator = (2 * mx * my + c1) * (2 * cov + c2)
    denominator = (mx * mx + my * my + c1) * (vx + vy + c2)
    return numerator / denominator

def _text_to_array(text: str) -> np.ndarray:
    """Convert a Unicode string to a uint8 numpy array (grayscale proxy)."""
    # Simple mapping: take the lower‑byte of each code point.
    arr = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    return arr.astype(np.float64)  # SSIM works with float values

# ----------------------------------------------------------------------
# Core primitives from Parent B (Schoolfield temperature model & bandit)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0          # rate at 25 °C (298.15 K)
    delta_h_activation: float = 12_000.0   # J mol⁻¹
    t_low: float = 283.15        # K
    t_high: float = 307.15       # K
    delta_h_low: float = -45_000.0  # J mol⁻¹
    delta_h_high: float = 65_000.0  # J mol⁻¹
    r_cal: float = 1.987         # gas constant cal mol⁻¹ K⁻¹

_POLICY: dict[str, list[float]] = {}
_STORE: dict[str, float] = {}          # virtual VRAM store per key

def reset_policy() -> None:
    """Clear all learned statistics."""
    _POLICY.clear()
    _STORE.clear()

def update_policy(updates: list[BanditUpdate]) -> None:
    """Aggregate rewards for each action."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)      # cumulative reward
        stats[1] += 1.0                   # count

def _reward(action_id: str) -> float:
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    return total / n if n else 0.0

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """
    Schoolfield temperature‑dependent activity curve.
    Returns a dimensionless scaling factor (rate relative to rho_25).
    """
    if temp_k <= 0:
        raise ValueError("temperature must be Kelvin‑positive")
    # Arrhenius term for activation
    arrhenius = math.exp(
        -(params.delta_h_activation / params.r_cal) *
        (1.0 / temp_k - 1.0 / 298.15)
    )
    # Low‑temperature inhibition term
    low_term = math.exp(
        (params.delta_h_low / params.r_cal) *
        (1.0 / params.t_low - 1.0 / temp_k)
    )
    # High‑temperature inhibition term
    high_term = math.exp(
        (params.delta_h_high / params.r_cal) *
        (1.0 / temp_k - 1.0 / params.t_high)
    )
    denominator = 1.0 + low_term + high_term
    return params.rho_25 * arrhenius / denominator

def temperature_modulated_learning_rate(temp_c: float) -> float:
    """Convenient wrapper returning a learning‑rate‑like scalar."""
    return developmental_rate(c_to_k(temp_c))

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def hybrid_score(packet: dict,
                 reference_text: str,
                 center: float,
                 width: float,
                 temp_celsius: float) -> float:
    """
    Compute the temperature‑modulated hybrid score:
        H = φ(T) · FisherScore(θ) · SSIM(text, reference)

    - θ is extracted from the packet as a float (fallback to 0.0).
    - The packet text is turned into a numeric array for SSIM.
    """
    # 1️⃣ Fisher component
    theta = float(packet.get("theta", 0.0))
    f = fisher_score(theta, center, width)

    # 2️⃣ SSIM component
    pkt_text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    x_arr = _text_to_array(pkt_text)
    y_arr = _text_to_array(reference_text)
    # Pad the shorter array to equal length (required by SSIM implementation)
    if x_arr.size < y_arr.size:
        x_arr = np.pad(x_arr, (0, y_arr.size - x_arr.size), mode='constant')
    elif y_arr.size < x_arr.size:
        y_arr = np.pad(y_arr, (0, x_arr.size - y_arr.size), mode='constant')
    s = ssim(x_arr, y_arr)

    # 3️⃣ Temperature modulation
    phi = temperature_modulated_learning_rate(temp_celsius)

    return phi * f * s

def route_decision(packet: dict,
                   reference_text: str,
                   center: float,
                   width: float,
                   temp_celsius: float,
                   actions: list[BanditAction]) -> BanditAction:
    """
    Choose a BanditAction based on the hybrid score.
    The action with the highest expected reward plus the hybrid score
    (treated as an instantaneous reward boost) is selected.
    """
    h = hybrid_score(packet, reference_text, center, width, temp_celsius)

    # Compute a simple utility: expected_reward + h * propensity
    utilities = []
    for a in actions:
        util = a.expected_reward + h * a.propensity
        utilities.append((util, a))

    # Select the action with maximal utility
    _, best = max(utilities, key=lambda pair: pair[0])
    return best

def update_bandit_from_packet(packet: dict,
                              reference_text: str,
                              center: float,
                              width: float,
                              temp_celsius: float,
                              context_id: str,
                              actions: list[BanditAction]) -> None:
    """
    Perform a full hybrid cycle:
        1. Compute hybrid score → instantaneous reward.
        2. Choose an action via route_decision.
        3. Record a BanditUpdate and feed it to the policy store.
    """
    chosen = route_decision(packet, reference_text, center, width,
                            temp_celsius, actions)

    reward = hybrid_score(packet, reference_text, center, width, temp_celsius)

    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen.action_id,
        reward=reward,
        propensity=chosen.propensity,
    )
    update_policy([update])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reset any lingering state
    reset_policy()

    # Dummy packet
    pkt = {
        "theta": 0.3,
        "text_surface": "Hello world! This is a test packet.",
        "source": "sensor_A"
    }

    # Reference text for SSIM
    ref = "Hello world! This is a reference sample."

    # Define a small action set
    actions = [
        BanditAction(
            action_id="route_A",
            propensity=0.8,
            expected_reward=_reward("route_A"),
            confidence_bound=0.1,
            algorithm="hybrid_fisher_ssim_bandit"
        ),
        BanditAction(
            action_id="route_B",
            propensity=0.5,
            expected_reward=_reward("route_B"),
            confidence_bound=0.2,
            algorithm="hybrid_fisher_ssim_bandit"
        ),
    ]

    # Run a single hybrid update at 22 °C
    update_bandit_from_packet(
        packet=pkt,
        reference_text=ref,
        center=0.0,
        width=1.0,
        temp_celsius=22.0,
        context_id="ctx_001",
        actions=actions,
    )

    # Print learned policy statistics
    print("Policy after one update:")
    for aid, stats in _POLICY.items():
        total, count = stats
        print(f"  Action {aid}: total_reward={total:.4f}, count={int(count)}")
    # Demonstrate that the hybrid score can be recomputed
    h = hybrid_score(pkt, ref, center=0.0, width=1.0, temp_celsius=22.0)
    print(f"Hybrid score (debug): {h:.6f}")