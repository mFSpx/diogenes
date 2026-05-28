import pytest

from pypeline.math.algos.rsa_cipher import rsa_decrypt, rsa_encrypt


def test_rsa_encrypts_known_plaintext_with_demo_key():
    assert rsa_encrypt(42, 17, 3233) == 2557


def test_rsa_decrypts_known_ciphertext_with_demo_key():
    assert rsa_decrypt(2557, 2753, 3233) == 42


def test_rsa_round_trips_small_message_with_demo_key():
    ciphertext = rsa_encrypt(123, 17, 3233)
    assert rsa_decrypt(ciphertext, 2753, 3233) == 123


def test_rsa_encrypts_zero_message():
    assert rsa_encrypt(0, 17, 3233) == 0


def test_rsa_encrypt_rejects_message_outside_modulus():
    with pytest.raises(ValueError, match=r"\[0, n\)"):
        rsa_encrypt(3233, 17, 3233)
