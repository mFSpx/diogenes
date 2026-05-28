import pytest

from pypeline.math.algos.gini_coefficient import gini_coefficient


def test_gini_empty_distribution_is_zero():
    assert gini_coefficient([]) == 0.0


def test_gini_equal_values_are_perfectly_equal():
    assert gini_coefficient([5, 5, 5, 5]) == pytest.approx(0.0)


def test_gini_all_zero_values_are_zero():
    assert gini_coefficient([0, 0, 0]) == 0.0


def test_gini_concentrated_distribution_has_high_inequality():
    assert gini_coefficient([0, 0, 0, 10]) == pytest.approx(0.75)


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError, match="non-negative"):
        gini_coefficient([1, -1, 2])
