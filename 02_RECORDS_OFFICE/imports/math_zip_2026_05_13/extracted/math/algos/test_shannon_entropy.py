import pytest

from pypeline.math.algos.shannon_entropy import shannon_entropy


def test_shannon_entropy_counts_uniform_observations():
    assert shannon_entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)


def test_shannon_entropy_is_zero_for_certain_observations():
    assert shannon_entropy(["a", "a", "a"]) == pytest.approx(0.0)


def test_shannon_entropy_empty_observations_are_zero():
    assert shannon_entropy([]) == 0.0


def test_shannon_entropy_accepts_normalized_distribution():
    assert shannon_entropy([0.25, 0.25, 0.25, 0.25], is_distribution=True) == pytest.approx(2.0)


def test_shannon_entropy_rejects_bad_distribution_sum():
    with pytest.raises(ValueError, match="distribution must sum"):
        shannon_entropy([0.2, 0.2], is_distribution=True)
