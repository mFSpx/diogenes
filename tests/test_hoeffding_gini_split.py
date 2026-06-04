from __future__ import annotations

from collections import Counter

import pytest

from ALGOS import hoeffding_gini_split as hgs


class _OneShotIterable:
    """Iterable that can be consumed only once."""

    def __init__(self, values):
        self._it = iter(values)
        self._used = False

    def __iter__(self):
        if self._used:
            raise AssertionError("Iterable was consumed more than once")
        self._used = True
        return self

    def __next__(self):
        return next(self._it)


def test_gini_gain_supports_counter_inputs():
    parent = Counter({0: 2, 1: 2})
    left = Counter({0: 2})
    right = Counter({1: 2})

    gain = hgs.gini_gain(parent, left, right)
    assert gain == pytest.approx(0.5)


def test_gini_gain_consumes_iterables_once():
    parent_labels = _OneShotIterable([0, 0, 1, 1])
    left_labels = _OneShotIterable([0, 0])
    right_labels = _OneShotIterable([1, 1])

    gain = hgs.gini_gain(parent_labels, left_labels, right_labels)
    assert gain == pytest.approx(0.5)


def test_streaming_node_evaluate_splits_uses_counter_counts(monkeypatch):
    node = hgs.StreamingNode()
    for lbl, feat in ((0, {0: 0, 1: 1}), (1, {0: 1, 1: 0}), (0, {0: 0, 1: 1})):
        node.update(feat, lbl)

    def _bad_elements(self):
        raise AssertionError("Counter.elements was used in split evaluation")

    monkeypatch.setattr(Counter, "elements", _bad_elements)

    decision = node.evaluate_splits()
    assert hasattr(decision, "should_split")
    assert isinstance(decision.should_split, bool)
