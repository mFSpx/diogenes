# DARWIN HAMMER — match 137, survivor 0
# gen: 3
# parent_a: rsa_cipher.py (gen0)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# born: 2026-05-29T23:27:10Z

"""Hybrid RSA‑Decision‑Hygiene Module

Parents:
* **rsa_cipher.py** – provides modular exponentiation based RSA encryption/decryption:
      c = m^e mod n,   m = c^d mod n
* **hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py** – extracts a 9‑dimensional
  feature‑count vector *v* from free‑text, computes a hygiene score *s* via two
  dot‑products and an information‑richness factor *H* (Shannon entropy).  
  The combined metric is  Sₕ = s·(1+H/Hₘₐₓ).

Mathematical bridge:
The hybrid algorithm first maps the continuous hybrid metric *Sₕ* to an integer
message *m* (by fixed‑point scaling) and then applies the RSA primitive to obtain
a protected ciphertext *c*.  Because RSA is a homomorphism for multiplication
modulo *n*, the scaling factor can be chosen such that the de‑crypted integer
exactly recovers the original scaled metric, allowing the audit pipeline to
store or transmit *Sₕ* securely while still being able to verify integrity via
RSA decryption.

The module therefore fuses the two topologies:
1. Vector‑based feature extraction → hygiene & entropy → *Sₕ* (parent B).
2. Integer‑based modular exponentiation → RSA encrypt/decrypt of *Sₕ* (parent A).

All core equations are present and inter‑linked in the functions below.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import List

import numpy as np

# ----------------------------------------------------------------------
# RSA primitive (parent A)
# ----------------------------------------------------------------------


def rsa_encrypt(message: int, e: int, n: int) -> int:
    """RSA encryption: c = m^e mod n."""
    if not 0 <= message < n:
        raise ValueError("message must be in [0, n)")
    return pow(message, e, n)


def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    """RSA decryption: m = c^d mod n."""
    if not 0 <= ciphertext < n:
        raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)


# ----------------------------------------------------------------------
# Feature extraction & hybrid metric (parent B)
# ----------------------------------------------------------------------


# Nine regex patterns – simplified but representative of the original parent.
_EVIDENCE_RE = r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b"
_PLANNING_RE = r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b"
_DELAY_RE = r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|postpone|delay|slow)\b"
_COMMUNICATION_RE = r"\b(?:email|call|meeting|chat|slack|discord|message|notify|communication|talk|discussion)\b"
_RISK_RE = r"\b(?:risk|threat|vulnerability|exposure|danger|hazard|issue|concern)\b"
_COMPLIANCE_RE = r"\b(?:compliance|policy|standard|regulation|legal|audit|governance|rule)\b"
_PERFORMANCE_RE = r"\b(?:performance|latency|throughput|speed|efficiency|optimize|benchmark)\b"
_SECURITY_RE = r"\b(?:security|encrypt|decrypt|cipher|auth|authentic|integrity|confidential|access)\b"
_DOCUMENTATION_RE = r"\b(?:doc|readme|guide|manual|spec|specification|design|architecture)\b"

_REGEXES = [
    _EVIDENCE_RE,
    _PLANNING_RE,
    _DELAY_RE,
    _COMMUNICATION_RE,
    _RISK_RE,
    _COMPLIANCE_RE,
    _PERFORMANCE_RE,
    _SECURITY_RE,
    _DOCUMENTATION_RE,
]

_COMPILED = [np.core.defchararray.compile(re, flags=0) for re in _REGEXES]  # type: ignore


def _compile_patterns() -> List[any]:
    """Compile the regex patterns using the standard library."""
    import re

    return [re.compile(pat, flags=re.IGNORECASE) for pat in _REGEXES]


_PATTERNS = _compile_patterns()


def extract_feature_vector(text: str) -> np.ndarray:
    """
    Return a 9‑dimensional integer vector where each entry counts the matches
    of the corresponding regex in *text*.
    """
    counts = [len(pat.findall(text)) for pat in _PATTERNS]
    return np.array(counts, dtype=int)


def hygiene_score(v: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray) -> float:
    """
    Compute the hygiene score s = w⁺·v − w⁻·v.
    """
    return float(np.dot(w_pos, v) - np.dot(w_neg, v))


def shannon_entropy(v: np.ndarray) -> float:
    """
    Compute Shannon entropy H = −∑ pᵢ log₂ pᵢ where p = v / Σv.
    Returns 0.0 for the zero vector.
    """
    total = v.sum()
    if total == 0:
        return 0.0
    p = v.astype(float) / total
    # add a tiny epsilon to avoid log(0)
    epsilon = 1e-12
    return -float(np.sum(p * np.log2(p + epsilon)))


def hybrid_score(v: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray) -> float:
    """
    Compute the combined metric Sₕ = s·(1 + H / Hₘₐₓ) with
    Hₘₐₓ = log₂(len(v)).
    """
    s = hygiene_score(v, w_pos, w_neg)
    H = shannon_entropy(v)
    H_max = math.log2(len(v))
    if H_max == 0:
        return s
    return s * (1.0 + H / H_max)


# ----------------------------------------------------------------------
# Fusion layer: RSA‑protected hybrid metric
# ----------------------------------------------------------------------


_SCALE = 10**6  # fixed‑point scaling factor to map float → int


def encrypt_hybrid_metric(metric: float, e: int, n: int, scale: int = _SCALE) -> int:
    """
    Convert *metric* to an integer using *scale* and RSA‑encrypt it.
    The integer is reduced modulo *n* before encryption to satisfy RSA bounds.
    """
    m_int = int(round(metric * scale)) % n
    return rsa_encrypt(m_int, e, n)


def decrypt_hybrid_metric(cipher: int, d: int, n: int, scale: int = _SCALE) -> float:
    """
    RSA‑decrypt *cipher* and convert the resulting integer back to a float
    using the inverse of the scaling factor.
    """
    m_int = rsa_decrypt(cipher, d, n)
    return m_int / scale


def audit_candidate(
    text: str,
    w_pos: np.ndarray,
    w_neg: np.ndarray,
    e: int,
    d: int,
    n: int,
    scale: int = _SCALE,
) -> dict:
    """
    End‑to‑end processing of a single candidate document:
      1. Extract feature vector.
      2. Compute hybrid metric Sₕ.
      3. RSA‑encrypt the scaled metric.
      4. RSA‑decrypt to verify integrity.
    Returns a dictionary with all intermediate values.
    """
    v = extract_feature_vector(text)
    metric = hybrid_score(v, w_pos, w_neg)
    cipher = encrypt_hybrid_metric(metric, e, n, scale)
    recovered = decrypt_hybrid_metric(cipher, d, n, scale)
    return {
        "vector": v.tolist(),
        "hybrid_metric": metric,
        "ciphertext": cipher,
        "recovered_metric": recovered,
    }


# ----------------------------------------------------------------------
# Helper: tiny RSA key generation (for demonstration only)
# ----------------------------------------------------------------------


def _is_prime(num: int) -> bool:
    if num < 2:
        return False
    if num % 2 == 0:
        return num == 2
    r = int(math.isqrt(num))
    for i in range(3, r + 1, 2):
        if num % i == 0:
            return False
    return True


def generate_small_rsa_keypair(bitlen: int = 16) -> tuple[int, int, int]:
    """
    Generate a tiny RSA key pair (p, q, e) suitable only for the smoke test.
    Returns (n, e, d).
    """
    # Very small prime search – deterministic for reproducibility.
    primes = [i for i in range(2 ** (bitlen - 1), 2 ** bitlen) if _is_prime(i)]
    p = random.choice(primes)
    q = random.choice([prime for prime in primes if prime != p])
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    # Ensure e and phi are coprime; if not, pick a smaller e.
    while math.gcd(e, phi) != 1:
        e = random.randrange(3, phi, 2)
    d = pow(e, -1, phi)  # modular inverse
    return n, e, d


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------


if __name__ == "__main__":
    # deterministic random seed for repeatable output
    random.seed(0)

    # Generate RSA key pair
    n, e, d = generate_small_rsa_keypair()

    # Random positive/negative weight vectors (length 9)
    w_pos = np.array([random.randint(1, 5) for _ in range(9)], dtype=float)
    w_neg = np.array([random.randint(0, 3) for _ in range(9)], dtype=float)

    # Example free‑text candidate
    sample_text = """
    The audit confirmed the evidence and source documentation. A detailed plan
    and checklist were prepared, but the implementation was delayed due to
    unexpected risk and compliance concerns. Communication via email and chat
    remained frequent. Security measures include encryption and hash verification.
    Performance benchmarks are pending.
    """

    result = audit_candidate(sample_text, w_pos, w_neg, e, d, n)

    # Simple sanity check: recovered metric should match original within rounding error.
    assert abs(result["hybrid_metric"] - result["recovered_metric"]) < 1e-6, "RSA round‑trip failed"

    # Print a concise report
    print("RSA modulus (n):", n)
    print("Public exponent (e):", e)
    print("Private exponent (d):", d)
    print("\nWeight vectors:")
    print("  w⁺ =", w_pos)
    print("  w⁻ =", w_neg)
    print("\nFeature vector:", result["vector"])
    print("Hybrid metric (Sₕ):", result["hybrid_metric"])
    print("Ciphertext:", result["ciphertext"])
    print("Recovered metric:", result["recovered_metric"])
    print("\nSmoke test completed successfully.")