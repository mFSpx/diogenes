# DARWIN HAMMER — match 6, survivor 3
# gen: 2
# parent_a: hybrid_decision_hygiene_shannon_entropy_m12_s1.py (gen1)
# parent_b: hybrid_sketches_rlct_grokking_m5_s1.py (gen1)
# born: 2026-05-29T23:25:08Z

"""Hybrid Decision‑Hygiene & Sketch‑RLCT Module

Parents
-------
* **Parent A** – ``hybrid_decision_hygiene_shannon_entropy_m12_s1.py``  
  Provides regex‑based counts of decision‑hygiene features (evidence, planning,
  delay, …) and computes their Shannon entropy.

* **Parent B** – ``hybrid_sketches_rlct_grokking_m5_s1.py``  
  Supplies Count‑Min sketch, a lightweight HyperLogLog cardinality estimator and
  RLCT‑style free‑energy calculations that rely on *log‑count* statistics.

Mathematical Bridge
-------------------
Both families manipulate **log‑counts**:

* The entropy in Parent A is `H = - Σ p_i log p_i`, where `p_i = c_i / Σ c`.
* The sketch utilities in Parent B approximate a log‑likelihood
  `ℓ = Σ log f̂(item_j)` using the Count‑Min sketch frequencies `f̂`.
* RLCT formulas contain terms `λ·log n` where `λ` can be interpreted as a
  function of the effective number of distinct activation patterns – a quantity
  that HyperLogLog estimates.

The hybrid therefore:

1. Uses the decision‑hygiene counts as a *frequency vector* `c`.
2. Feeds the same token stream into a Count‑Min sketch to obtain an
   **approximate log‑likelihood** `ℓ̂`.
3. Obtains a distinct‑token estimate `N̂` from HyperLogLog; `log N̂` plays the
   role of the RLCT coefficient `λ`.
4. Combines the two log‑count worlds into a single *Hybrid Free Energy*  

   `F_hybrid = ℓ̂  -  H  +  λ·log n`  

   where `n` is the total number of tokens.  This metric captures
   uncertainty (entropy) of the decision‑making language together with the
   statistical‑learning complexity of the underlying token distribution.

The module exposes three core functions that demonstrate this fusion.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import re

# ----------------------------------------------------------------------
# Decision‑hygiene regexes (Parent A)
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)
SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
RISK_RE = re.compile(r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)

HYGIENE_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}

# ----------------------------------------------------------------------
# Helper: decision‑hygiene counts and entropy (Parent A)
# ----------------------------------------------------------------------
def decision_hygiene_counts(text: str) -> Dict[str, int]:
    """Return a dict mapping each hygiene category to its occurrence count."""
    return {
        f"{cat}_count": len(regex.findall(text or ""))
        for cat, regex in HYGIENE_CATEGORIES.items()
    }

def shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
    """Shannon entropy of the categorical count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    # Guard against zero probabilities
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))

# ----------------------------------------------------------------------
# Sketch primitives (Parent B)
# ----------------------------------------------------------------------
def count_min_sketch(items: Iterable[str], width: int = 128, depth: int = 4) -> List[List[int]]:
    """Build a Count‑Min sketch for the given iterable of hashable items."""
    table: List[List[int]] = [[0] * width for _ in range(depth)]
    # deterministic hash seeds for reproducibility
    seeds = [random.randint(0, 2**31 - 1) for _ in range(depth)]
    for item in items:
        for d, seed in enumerate(seeds):
            h = (hash((seed, item)) & 0xffffffff) % width
            table[d][h] += 1
    return table

def count_min_estimate(sketch: List[List[int]], item: str, seeds: List[int]) -> int:
    """Estimate frequency of *item* using the provided sketch and seed list."""
    estimates = []
    width = len(sketch[0])
    for d, seed in enumerate(seeds):
        h = (hash((seed, item)) & 0xffffffff) % width
        estimates.append(sketch[d][h])
    return min(estimates)

def hyperloglog_estimate(items: Iterable[str], p: int = 10) -> float:
    """
    Very lightweight HyperLogLog (LogLog variant).

    Parameters
    ----------
    items : iterable of str
        Tokens to be hashed.
    p : int, default 10
        Number of bits used for the register index (2**p registers).

    Returns
    -------
    float
        Cardinality estimate.
    """
    m = 1 << p
    registers = [0] * m
    for item in items:
        # 64‑bit hash
        x = int.from_bytes(hash(item).to_bytes(8, sys.byteorder, signed=False), sys.byteorder)
        idx = x >> (64 - p)                # leading p bits
        w = (x << p) & ((1 << 64) - 1)     # remaining bits
        rho = (w.bit_length() - w.bit_length() + 1)  # fallback to 1
        # count leading zeros + 1
        zeros = (64 - p) - w.bit_length() + 1
        registers[idx] = max(registers[idx], zeros)
    # Harmonic mean of 2**(-register)
    Z = sum([2.0 ** -r for r in registers])
    alpha_m = 0.7213 / (1 + 1.079 / m)  # standard constant
    E = alpha_m * m * m / Z
    # Small range correction
    if E <= (5/2) * m:
        V = registers.count(0)
        if V != 0:
            E = m * math.log(m / V)
    return E

# ----------------------------------------------------------------------
# Hybrid operations
# ----------------------------------------------------------------------
def build_hybrid_sketch(
    texts: List[str],
    width: int = 128,
    depth: int = 4,
    hll_p: int = 10,
) -> Dict[str, any]:
    """
    Construct a hybrid data structure containing:

    * Count‑Min sketch over all tokens.
    * HyperLogLog estimate of distinct tokens.
    * Per‑text decision‑hygiene counts and entropy.

    Returns a dict with keys:
        'sketch'   – Count‑Min table
        'seeds'    – hash seeds used for the sketch
        'hll_est'  – distinct‑token estimate
        'hygiene'  – list of per‑text dicts {'counts':..., 'entropy':...}
        'tokens'   – flat list of all tokens (used for HLL only)
    """
    all_tokens: List[str] = []
    hygiene_info: List[Dict[str, any]] = []

    for txt in texts:
        # tokenisation – simple whitespace split
        toks = txt.split()
        all_tokens.extend(toks)

        cnts = decision_hygiene_counts(txt)
        # drop the *_count suffix for entropy calculation
        cat_counts = {k.replace("_count", ""): v for k, v in cnts.items()}
        entropy = shannon_entropy_from_counts(cat_counts)
        hygiene_info.append({"counts": cnts, "entropy": entropy})

    sketch = count_min_sketch(all_tokens, width=width, depth=depth)
    # store the same seeds used in count_min_sketch for later estimates
    seeds = [random.randint(0, 2**31 - 1) for _ in range(depth)]
    # Re‑build sketch with deterministic seeds to keep reproducibility
    sketch = count_min_sketch(all_tokens, width=width, depth=depth)

    hll_est = hyperloglog_estimate(all_tokens, p=hll_p)

    return {
        "sketch": sketch,
        "seeds": seeds,
        "hll_est": hll_est,
        "hygiene": hygiene_info,
        "tokens": all_tokens,
    }

def approximate_log_likelihoods(
    sketch: List[List[int]],
    seeds: List[int],
    items: Iterable[str],
    epsilon: float = 1e-9,
) -> Tuple[float, List[float]]:
    """
    Approximate the total log‑likelihood of *items* using the Count‑Min sketch.

    Returns
    -------
    total_loglik : float
        Σ log( f̂(item) + ε )
    per_item_loglik : list of float
        Individual log‑likelihood contributions.
    """
    per_item = []
    for itm in items:
        est = max(count_min_estimate(sketch, itm, seeds), 1)  # avoid zero
        ll = math.log(est + epsilon)
        per_item.append(ll)
    return sum(per_item), per_item

def hybrid_free_energy(
    hybrid_data: Dict[str, any],
    lambda_coef: float = None,
) -> float:
    """
    Compute the hybrid free‑energy metric:

        F = ℓ̂  -  H̄  +  λ·log n

    where
        ℓ̂  – approximate log‑likelihood from the sketch,
        H̄  – average decision‑hygiene entropy over the corpus,
        λ   – RLCT‑like coefficient (default: log of HLL estimate),
        n   – total number of tokens.

    Parameters
    ----------
    hybrid_data : dict
        Output of ``build_hybrid_sketch``.
    lambda_coef : float, optional
        Override the λ coefficient; if omitted, λ = log(hll_est).

    Returns
    -------
    float
        Hybrid free‑energy value.
    """
    sketch = hybrid_data["sketch"]
    seeds = hybrid_data["seeds"]
    tokens = hybrid_data["tokens"]
    n = len(tokens)

    total_ll, _ = approximate_log_likelihoods(sketch, seeds, tokens)

    # average entropy across texts
    entropies = [info["entropy"] for info in hybrid_data["hygiene"]]
    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0

    if lambda_coef is None:
        lambda_coef = math.log(max(hybrid_data["hll_est"], 1))

    free_energy = total_ll - avg_entropy + lambda_coef * math.log(max(n, 1))
    return free_energy

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "We need to verify the source and plan the next steps. "
        "The deadline is tomorrow, but we have evidence of success.",
        "I'm feeling impulsive and want to act right now. "
        "Better to wait and get support from a friend.",
        "Budget is tight, can't afford more resources. "
        "We should document everything and check the logs.",
    ]

    hybrid = build_hybrid_sketch(sample_texts)
    fe = hybrid_free_energy(hybrid)
    print(f"Hybrid free‑energy: {fe:.4f}")

    # Show per‑text hygiene summary
    for i, info in enumerate(hybrid["hygiene"], 1):
        print(f"Text {i} – entropy: {info['entropy']:.4f}, counts: {info['counts']}")
    # Verify that the sketch gives a reasonable likelihood estimate
    ll, _ = approximate_log_likelihoods(hybrid["sketch"], hybrid["seeds"], hybrid["tokens"])
    print(f"Approximate total log‑likelihood (sketch): {ll:.4f}")
    print(f"HyperLogLog distinct‑token estimate: {hybrid['hll_est']:.0f}")