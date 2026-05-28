#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_treelite_matrix_has_26_routers_and_prunes_weak_branch():
    import adhd_slow_lane_divergence as slow

    branches = [
        {"frame": "evidence", "lane": "deepseek", "text": "Evidence: source, timeline, risk, first step."},
        {"frame": "vibes", "lane": "bonsai", "text": "Maybe it feels interesting."},
    ]
    scored = slow.score_branches(branches)
    survivors = slow.prune(scored, survivors=1)

    assert slow.ROUTER_COUNT == 26
    assert len(scored[0]["router_scores"]) == 26
    assert len(scored[1]["router_scores"]) == 26
    assert len(survivors) == 1
    assert survivors[0]["frame"] == "evidence"
