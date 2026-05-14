import pytest

from pypeline.math.algos.bayes_update import bayes_marginal, bayes_update


def test_bayes_marginal_combines_prior_and_complement():
    assert bayes_marginal(0.01, 0.95, 0.05) == pytest.approx(0.059)


def test_bayes_update_uses_prior_likelihood_and_marginal():
    assert bayes_update(0.01, 0.95, 0.059) == pytest.approx(0.1610169492)


def test_bayes_update_returns_prior_when_likelihood_matches_marginal():
    assert bayes_update(0.4, 0.2, 0.2) == pytest.approx(0.4)


def test_bayes_update_rejects_zero_marginal():
    with pytest.raises(ValueError, match="P\\(E\\) must be > 0"):
        bayes_update(0.4, 0.2, 0.0)
