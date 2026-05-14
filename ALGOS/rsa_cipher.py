#!/usr/bin/env python3
"""Tiny textbook RSA integer primitive for demos/tests only; not production crypto."""
from __future__ import annotations

def rsa_encrypt(message: int, e: int, n: int) -> int:
    if not 0 <= message < n: raise ValueError("message must be in [0, n)")
    return pow(message, e, n)

def rsa_decrypt(ciphertext: int, d: int, n: int) -> int:
    if not 0 <= ciphertext < n: raise ValueError("ciphertext must be in [0, n)")
    return pow(ciphertext, d, n)
