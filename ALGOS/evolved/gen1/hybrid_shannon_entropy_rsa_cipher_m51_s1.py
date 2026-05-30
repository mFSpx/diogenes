# DARWIN HAMMER — match 51, survivor 1
# gen: 1
# parent_a: shannon_entropy.py (gen0)
# parent_b: rsa_cipher.py (gen0)
# born: 2026-05-29T23:23:50Z

from __future__ import annotations
import math
from collections import Counter
from collections.abc import Hashable, Iterable
import numpy as np
import random
import sys
import pathlib

"""
This module combines the Shannon entropy calculation from shannon_entropy.py with the RSA cipher from rsa_cipher.py.
The mathematical bridge between the two is the use of entropy as a measure of the uncertainty of the encrypted message.
In this hybrid algorithm, we use the Shannon entropy to analyze the randomness of the encrypted message and adjust the RSA cipher parameters accordingly.
"""

def hybrid_encrypt(message: int, e: int, n: int) -> int:
    encrypted_message = rsa_encrypt(message, e, n)
    entropy = calculate_entropy(encrypted_message, n)
    return encrypted_message, entropy

def calculate_entropy(message: int, n: int) -> float:
    binary_message = np.array([int(x) for x in bin(message)[2:].zfill(n.bit_length())])
    observations = binary_message.tolist()
    return shannon_entropy(observations, is_distribution=False)

def shannon_entropy(observations: Iterable[Hashable | float], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs: return 0.0
    if is_distribution:
        probs=[float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs)-1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c=Counter(xs); total=sum(c.values()); probs=[v/total for v in c.values()]
    return -sum(p*math.log2(p) for p in probs if p > 0)

def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n: raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n: raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)

def find_optimal_key(n: int) -> int:
    max_entropy = 0
    optimal_key = 0
    for e in range(2, n):
        if math.gcd(e, (n-1)) == 1:
            message = random.randint(0, n-1)
            encrypted_message, entropy = hybrid_encrypt(message, e, n)
            if entropy > max_entropy:
                max_entropy = entropy
                optimal_key = e
    return optimal_key

if __name__ == "__main__":
    n = 257
    e = 17
    d = 77
    message = 123
    encrypted_message, entropy = hybrid_encrypt(message, e, n)
    decrypted_message = rsa_decrypt(encrypted_message, d, n)
    assert decrypted_message == message
    optimal_key = find_optimal_key(n)
    print(f"Optimal key: {optimal_key}")