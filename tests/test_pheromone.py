#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace

from ALGOS.runtime_caps import MAX_DB_ROWS
from ALGOS import pheromone


def test_decay_limit_is_clamped_to_caps():
    assert pheromone._clamp_limit(0) == 1
    assert pheromone._clamp_limit(MAX_DB_ROWS + 99) == MAX_DB_ROWS


def test_decay_runs_with_clamped_limit_when_execute_false(tmp_path, monkeypatch):
    # Ensure output lands in test sandbox.
    monkeypatch.setattr(pheromone, "OUT", tmp_path)

    args = SimpleNamespace(execute=False, surface_key="x", limit=MAX_DB_ROWS + 99)
    rc = pheromone.decay(args)
    assert rc == 0
