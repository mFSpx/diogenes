# DARWIN HAMMER — match 4452, survivor 4
# gen: 6
# parent_a: hybrid_bandit_router_poikilotherm_schoolf_m20_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_shannon_entro_m1636_s3.py (gen5)
# born: 2026-05-29T23:55:55Z

"""Hybrid Bandit‑RSA‑Poikilotherm Algorithm
================================================
This module fuses the temperature‑dependent developmental rate from the
Schoolfield poikilotherm model (Parent A) with the contextual multi‑armed
bandit machinery and the RSA‑based Shannon‑entropy analysis from the
morphology‑entropy hybrid (Parent B).

Mathematical bridge
-------------------
* **Temperature → reward scaling** – The developmental rate  
  \\[
  \\rho(T)=\\frac{\\rho_{25}\\,\\frac{T}{T_{25}}\\,
  \\exp\\Bigl(\\frac{\\Delta H_{a}}{R}\\bigl(\\frac{1}{T_{25}}-\\frac{1}{T}\\bigr)\\Bigr)}
  {1+\\exp\\Bigl(\\frac{\\Delta H_{L}}{R}\\bigl(\\frac{1}{T_{L}}-\\frac{1}{T}\\bigr)\\Bigr)
   +\\exp\\Bigl(\\frac{\\Delta H_{H}}{R}\\bigl(\\frac{1}{T_{H}}-\\frac{1}{T}\\bigr)\\Bigr)}
  \\]
  is used as a multiplicative factor that rescales the *raw* bandit
  reward.

* **RSA‑encrypted action identifiers → entropy‑driven confidence** – Each
  action identifier is transformed into an integer, RSA‑encrypted, and
  normalised by the modulus \\(n\\). The Shannon entropy of the set of
  normalised encrypted values quantifies the information spread of the
  current action space. This entropy term is added to the classic
  Upper‑Confidence‑Bound (UCB) confidence term, allowing the exploration
 ‑exploitation trade‑off to adapt to the cryptographic “diversity” of the
  actions.

The hybrid reward for an action \\(a\\) at temperature \\(T\\) is therefore
\\[
\\hat r_a = \\rho(T)\\,\\frac{\\mathrm{RSA}(\\text{id}_a)}{n}
\\]
and the selection rule becomes
\\[
a^* = \\arg\\max_a \\bigl( \\hat r_a + \\beta
\\sqrt{\\frac{2\\ln N}{N_a}}\\,(1+H) \\bigr),
\\]
where \\(N\\) is the total number of pulls, \\(N_a\\) the count for action
\\(a\\), \\(\\beta\\) a tunable scalar, and \\(H\\) the Shannon entropy of
the normalised encrypted identifiers.

The code below implements these equations while preserving the original
data‑class structures of both parents.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Schoolfield developmental rate and bandit scaffolding
# ----------------------------------------------------------------------
R_CAL = 1.987  # cal·mol⁻¹·K⁻¹
K25 = 298.15   # reference temperature (K)

@dataclass(frozen=True)
class SchoolfieldParams:
    rho_25: float = 1.0
    delta_h_activation: float = 12_000.0
    t_low: float = 283.15
    t_high: float = 307.15
    delta_h_low: float = -45_000.0
    delta_h_high: float = 65_000.0
    r_cal: float = R_CAL

def c_to_k(celsius: float) -> float:
    """Convert Celsius to Kelvin."""
    return celsius + 273.15

def developmental_rate(temp_k: float,
                       params: SchoolfieldParams = SchoolfieldParams()) -> float:
    """Schoolfield temperature‑dependent developmental rate."""
    if temp_k <= 0 or params.rho_25 < 0:
        raise ValueError("temperature must be Kelvin‑positive and rho_25 non‑negative")
    numerator = (params.rho_25 *
                 (temp_k / K25) *
                 math.exp((params.delta_h_activation / params.r_cal) *
                          ((1.0 / K25) - (1.0 / temp_k))))
    low = math.exp((params.delta_h_low / params.r_cal) *
                   ((1.0 / params.t_low) - (1.0 / temp_k)))
    high = math.exp((params.delta_h_high / params.r_cal) *
                    ((1.0 / params.t_high) - (1.0 / temp_k)))
    return numerator / (1.0 + low + high)

# ----------------------------------------------------------------------
# Parent B – RSA encryption, Shannon entropy, and morphology stub
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

def _egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    g, y, x = _egcd(b % a, a)
    return g, x - (b // a) * y, y

def _modinv(a: int, m: int) -> int:
    """Modular inverse of a modulo m."""
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("inverse does not exist")
    return x % m

def generate_rsa_keypair(prime_bits: int = 16) -> Tuple[int, int, int]:
    """Generate a tiny RSA keypair (e, d, n) for demonstration."""
    def _rand_prime(bits: int) -> int:
        while True:
            p = random.getrandbits(bits) | 1
            if all(p % q for q in range(3, int(math.isqrt(p)) + 1, 2)):
                return p
    p = _rand_prime(prime_bits)
    q = _rand_prime(prime_bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    while math.gcd(e, phi) != 1:
        e += 2
    d = _modinv(e, phi)
    return e, d, n

def rsa_encrypt_int(message: int, e: int, n: int) -> int:
    """RSA encryption of a non‑negative integer."""
    if not (0 <= message < n):
        raise ValueError("message out of range for modulus")
    return pow(message, e, n)

def shannon_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy (base‑e) of a probability vector."""
    probs = probabilities[probabilities > 0]
    return -float(np.sum(probs * np.log(probs)))

# ----------------------------------------------------------------------
# Hybrid structures and global policy store
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float = 0.0          # will be filled by selector
    expected_reward: float = 0.0     # scaled by temperature & RSA
    confidence_bound: float = 0.0    # includes entropy term
    algorithm: str = "HybridBanditRSA"

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

# Global policy: maps action_id -> [cumulative_reward, count]
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Clear the internal policy statistics."""
    _POLICY.clear()

def update_policy(updates: List[BanditUpdate]) -> None:
    """Incorporate observed rewards into the policy."""
    for u in updates:
        stats = _POLICY.setdefault(u.action_id, [0.0, 0.0])
        stats[0] += float(u.reward)
        stats[1] += 1.0

# ----------------------------------------------------------------------
# Hybrid core functions (fulfil requirement ≥3)
# ----------------------------------------------------------------------
def hybrid_encrypted_scaling(action_id: str, e: int, n: int) -> float:
    """
    Convert an action identifier into a normalized RSA‑encrypted scalar.
    The identifier string is first mapped to an integer via a simple
    character code sum, then encrypted and divided by the modulus.
    """
    message = sum(ord(ch) for ch in action_id) % n
    encrypted = rsa_encrypt_int(message, e, n)
    return encrypted / n  # value in (0,1)

def hybrid_select_action(context_id: str,
                         actions: List[BanditAction],
                         temp_c: float,
                         rsa_key: Tuple[int, int, int],
                         beta: float = 1.0) -> BanditAction:
    """
    Select an action using a temperature‑scaled reward and an
    entropy‑augmented UCB confidence term.

    Parameters
    ----------
    context_id: identifier of the current context (unused in this simple demo)
    actions: list of candidate BanditAction objects (action_id must be set)
    temp_c: ambient temperature in Celsius
    rsa_key: (e, d, n) RSA keypair – only (e, n) are needed here
    beta: exploration scaling factor

    Returns
    -------
    BanditAction with updated propensity, expected_reward, and confidence_bound.
    """
    e, _, n = rsa_key
    temp_k = c_to_k(temp_c)
    rate = developmental_rate(temp_k)

    # Compute scaled rewards for each action
    scaled_rewards = []
    for a in actions:
        enc_scalar = hybrid_encrypted_scaling(a.action_id, e, n)
        scaled = rate * enc_scalar
        scaled_rewards.append(scaled)

    # Entropy term from the distribution of encrypted scalars
    prob_vec = np.array(scaled_rewards) / (np.sum(scaled_rewards) + 1e-12)
    entropy = shannon_entropy(prob_vec)

    total_pulls = sum(_POLICY.get(a.action_id, [0.0, 0.0])[1] for a in actions) + 1e-12
    # Build enriched actions
    enriched = []
    for a, r in zip(actions, scaled_rewards):
        count = _POLICY.get(a.action_id, [0.0, 0.0])[1]
        avg_reward = (_POLICY.get(a.action_id, [0.0, 0.0])[0] / count) if count > 0 else 0.0
        ucb = r + beta * math.sqrt((2 * math.log(total_pulls)) / (count + 1)) * (1.0 + entropy)
        enriched.append(BanditAction(
            action_id=a.action_id,
            propensity=r,
            expected_reward=avg_reward,
            confidence_bound=ucb,
            algorithm=a.algorithm
        ))

    # Choose action with maximal confidence bound
    chosen = max(enriched, key=lambda x: x.confidence_bound)
    return chosen

def hybrid_update_policy(context_id: str,
                         chosen_action: BanditAction,
                         observed_reward: float,
                         propensity: float) -> None:
    """
    Record an observed reward for the chosen action. The update follows the
    parent‑A policy bookkeeping while also preserving the propensity
    (temperature‑scaled RSA scalar) for potential diagnostics.
    """
    update = BanditUpdate(
        context_id=context_id,
        action_id=chosen_action.action_id,
        reward=observed_reward,
        propensity=propensity
    )
    update_policy([update])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise RSA keypair
    rsa_key = generate_rsa_keypair(prime_bits=12)  # small for speed

    # Define a small action space
    action_ids = ["alpha", "beta", "gamma", "delta"]
    actions = [BanditAction(action_id=a) for a in action_ids]

    # Simulated loop
    reset_policy()
    for step in range(10):
        temp_c = random.uniform(5.0, 35.0)  # ambient temperature
        chosen = hybrid_select_action(
            context_id="ctx1",
            actions=actions,
            temp_c=temp_c,
            rsa_key=rsa_key,
            beta=0.5
        )
        # Simulated environment reward (here just a noisy temperature‑dependent signal)
        true_reward = developmental_rate(c_to_k(temp_c)) * random.random()
        hybrid_update_policy(
            context_id="ctx1",
            chosen_action=chosen,
            observed_reward=true_reward,
            propensity=chosen.propensity
        )
        print(f"Step {step:02d}: temp={temp_c:.1f}°C, chosen={chosen.action_id}, "
              f"propensity={chosen.propensity:.4f}, reward={true_reward:.4f}")

    # Final policy dump
    print("\nFinal policy statistics:")
    for aid, (cum, cnt) in _POLICY.items():
        print(f"  {aid}: total_reward={cum:.4f}, pulls={int(cnt)}")