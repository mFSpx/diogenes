# DARWIN HAMMER — match 51, survivor 0
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
This module integrates the Shannon entropy calculation from the Shannon entropy algorithm 
and the RSA encryption/decryption from the RSA cipher algorithm. The mathematical bridge 
between these two structures is the concept of information security and uncertainty. 
The Shannon entropy is used to measure the uncertainty or randomness of a probability 
distribution, while the RSA algorithm is used to secure this information by encrypting 
it. In this hybrid algorithm, we use the Shannon entropy to measure the randomness of 
a message before and after encryption, demonstrating the effect of encryption on the 
information's uncertainty.
"""

def calculate_shannon_entropy(observations: Iterable[Hashable | float], is_distribution: bool = False) -> float:
    xs = list(observations)
    if not xs: return 0.0
    if is_distribution:
        probs = [float(x) for x in xs]
        if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("distribution must sum to 1")
    else:
        c = Counter(xs)
        total = sum(c.values())
        probs = [v / total for v in c.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n: raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n: raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)

def hybrid_operation(message: str, e: int, n: int, d: int) -> None:
    # Convert the message to a numerical representation
    numerical_message = [ord(char) for char in message]
    # Calculate the Shannon entropy of the message
    entropy_before_encryption = calculate_shannon_entropy(numerical_message)
    print(f"Shannon entropy before encryption: {entropy_before_encryption}")
    # Encrypt the message
    encrypted_message = [rsa_encrypt(num, e, n) for num in numerical_message]
    # Calculate the Shannon entropy of the encrypted message
    entropy_after_encryption = calculate_shannon_entropy(encrypted_message)
    print(f"Shannon entropy after encryption: {entropy_after_encryption}")
    # Decrypt the message
    decrypted_message = [rsa_decrypt(num, d, n) for num in encrypted_message]
    # Calculate the Shannon entropy of the decrypted message
    entropy_after_decryption = calculate_shannon_entropy(decrypted_message)
    print(f"Shannon entropy after decryption: {entropy_after_decryption}")
    # Convert the decrypted message back to a string
    decrypted_string = ''.join([chr(num) for num in decrypted_message])
    print(f"Decrypted message: {decrypted_string}")

def generate_random_message(length: int) -> str:
    return ''.join([random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)])

def test_hybrid_operation() -> None:
    message = generate_random_message(10)
    e = 17
    n = 323
    d = 275
    hybrid_operation(message, e, n, d)

if __name__ == "__main__":
    test_hybrid_operation()