"""Tests for pypeline.math.temporal_motifs — RFC-0018 §3.7 WO-BM-016/017/018."""
from __future__ import annotations

import time
import pytest
from pypeline.math.temporal_motifs import (
    BurstSignal,
    TemporalMotif,
    sessionize_events,
    detect_bursts,
    mine_temporal_motifs,
)


def _ev(ts: float) -> dict:
    return {"ts": ts, "type": "action"}


class TestSessionizeEvents:
    def test_empty_returns_empty(self):
        assert sessionize_events([]) == []

    def test_single_event_one_session(self):
        sessions = sessionize_events([_ev(0.0)])
        assert len(sessions) == 1
        assert len(sessions[0]) == 1

    def test_close_events_same_session(self):
        events = [_ev(0.0), _ev(60.0), _ev(120.0)]
        sessions = sessionize_events(events, session_gap_seconds=1800.0)
        assert len(sessions) == 1
        assert len(sessions[0]) == 3

    def test_gap_splits_sessions(self):
        events = [_ev(0.0), _ev(7200.0)]  # 2-hour gap
        sessions = sessionize_events(events, session_gap_seconds=1800.0)
        assert len(sessions) == 2

    def test_unsorted_events_sorted_within_session(self):
        events = [_ev(120.0), _ev(0.0), _ev(60.0)]
        sessions = sessionize_events(events)
        ts_vals = [e["ts"] for e in sessions[0]]
        assert ts_vals == sorted(ts_vals)

    def test_iso_timestamps_parsed(self):
        events = [
            {"ts": "2026-01-01T00:00:00Z"},
            {"ts": "2026-01-01T00:10:00Z"},
        ]
        sessions = sessionize_events(events, session_gap_seconds=1800.0)
        assert len(sessions) == 1
        assert len(sessions[0]) == 2

    def test_custom_gap_respected(self):
        events = [_ev(0.0), _ev(100.0), _ev(300.0)]
        sessions = sessionize_events(events, session_gap_seconds=150.0)
        assert len(sessions) == 2


class TestDetectBursts:
    def _rapid_events(self, count: int, base: float = 0.0, step: float = 1.0) -> list[dict]:
        return [_ev(base + i * step) for i in range(count)]

    def test_empty_events_returns_empty(self):
        assert detect_bursts([], "entity_001") == []

    def test_single_event_returns_empty(self):
        assert detect_bursts([_ev(0.0)], "entity_001") == []

    def test_two_events_returns_empty(self):
        assert detect_bursts([_ev(0.0), _ev(1.0)], "entity_001") == []

    def test_burst_detected_returns_signals(self):
        events = self._rapid_events(20, step=1.0)
        signals = detect_bursts(events, "entity_001")
        assert isinstance(signals, list)
        assert len(signals) >= 1

    def test_signal_has_correct_entity_id(self):
        events = self._rapid_events(10, step=1.0)
        signals = detect_bursts(events, "target_entity")
        for s in signals:
            assert s.entity_id == "target_entity"

    def test_signal_fields_valid(self):
        events = self._rapid_events(15, step=0.5)
        signals = detect_bursts(events, "e1")
        for s in signals:
            assert isinstance(s, BurstSignal)
            assert s.event_count >= 3
            assert 0.0 <= s.inter_arrival_anomaly_score <= 1.0
            assert 0.0 <= s.hawkes_self_excitement_score <= 1.0
            assert isinstance(s.is_coordinated_burst, bool)

    def test_rapid_burst_has_high_anomaly_score(self):
        # Rapid events (step=1s) vs slow background (step=3600s)
        background = self._rapid_events(5, base=0.0, step=3600.0)
        burst = self._rapid_events(10, base=background[-1]["ts"] + 3600.0, step=1.0)
        signals = detect_bursts(background + burst, "e2")
        # The burst session should exist with non-zero anomaly
        assert any(s.inter_arrival_anomaly_score > 0 for s in signals)

    def test_signal_ids_unique(self):
        events = self._rapid_events(20, step=0.5)
        signals = detect_bursts(events, "e3")
        ids = [s.signal_id for s in signals]
        assert len(ids) == len(set(ids))


class TestMineTemporalMotifs:
    def test_empty_sequences_returns_empty(self):
        assert mine_temporal_motifs([]) == []

    def test_rare_pattern_below_min_frequency_excluded(self):
        seqs = [["a", "b", "c"]] * 2  # only frequency 2, min=5
        motifs = mine_temporal_motifs(seqs, n_gram_size=3, min_frequency=5)
        assert motifs == []

    def test_frequent_pattern_included(self):
        seqs = [["a", "b", "c"]] * 10
        motifs = mine_temporal_motifs(seqs, n_gram_size=3, min_frequency=5)
        assert len(motifs) == 1
        assert motifs[0].pattern == ("a", "b", "c")
        assert motifs[0].frequency == 10

    def test_most_common_has_lowest_anomaly_score(self):
        seqs = [["x", "y", "z"]] * 20 + [["a", "b", "c"]] * 6
        motifs = mine_temporal_motifs(seqs, n_gram_size=3, min_frequency=5)
        assert len(motifs) == 2
        # Most frequent pattern should have anomaly_score=0 (max freq)
        most_frequent = min(motifs, key=lambda m: m.anomaly_score)
        assert most_frequent.pattern == ("x", "y", "z")
        assert most_frequent.anomaly_score == pytest.approx(0.0)

    def test_motif_ids_unique(self):
        seqs = [["a", "b", "c"]] * 8 + [["d", "e", "f"]] * 6
        motifs = mine_temporal_motifs(seqs, n_gram_size=3, min_frequency=5)
        ids = [m.motif_id for m in motifs]
        assert len(ids) == len(set(ids))

    def test_false_positive_band_set(self):
        seqs = [["a", "b", "c", "d"]] * 10
        motifs = mine_temporal_motifs(seqs, n_gram_size=3, min_frequency=5)
        for m in motifs:
            assert m.false_positive_band == (0.05, 0.20)

    def test_bigrams(self):
        seqs = [["x", "y", "x", "y", "x", "y"]] * 6
        motifs = mine_temporal_motifs(seqs, n_gram_size=2, min_frequency=5)
        patterns = [m.pattern for m in motifs]
        assert ("x", "y") in patterns or ("y", "x") in patterns
