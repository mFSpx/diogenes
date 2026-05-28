#!/usr/bin/env python3
"""Generic deterministic signal aggregation CLI.

Purpose:
- Combine overlapping alarms/sensor outputs/model scores into one advisory stance.
- Stay domain-agnostic: Ahoy, ops, graph triage, monitoring, routing, etc.
- Use only Python stdlib so it can be dropped anywhere.

Input JSON shape (all fields optional except signals):
{
  "signals": {
    "friction": {"score": 0.85, "reliability": 0.9, "tier": 1,
                  "entity": "node-a", "hypotheses": {"CONSOLIDATE": 0.7, "RECOVER": 0.3}},
    "opportunity": {"score": 0.40, "tier": 2, "hypotheses": {"EXPLOIT": 1.0}}
  },
  "graph": {"entity_centrality": {"node-a": 0.8}},
  "config": {"strategy": "ensemble", "half_life_seconds": 30}
}

If a signal has no hypotheses, its own uppercased name becomes its hypothesis.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "lucidota.generic.signal_aggregate.v1"
THETA = "*"
CONTRADICTION = "CONTRADICTION"
DEFAULT_HOLD = "HOLD"


def clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except Exception:
        return lo
    if math.isnan(v) or math.isinf(v):
        return lo
    return max(lo, min(hi, v))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def decay(score: float, age_seconds: float, half_life_seconds: float) -> float:
    if age_seconds <= 0:
        return score
    return score * math.exp(-math.log(2.0) * age_seconds / max(0.001, half_life_seconds))


def top_score(scores: dict[str, float], default: str = DEFAULT_HOLD) -> tuple[str, float]:
    if not scores:
        return default, 0.0
    k = max(scores, key=lambda x: scores[x])
    return k, float(scores[k])


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    vals = {str(k): max(0.0, safe_float(v)) for k, v in (weights or {}).items()}
    total = sum(vals.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in vals.items()}


@dataclass(frozen=True)
class Signal:
    name: str
    score: float
    reliability: float = 1.0
    tier: int = 3
    entity: str | None = None
    timestamp: float | None = None
    age_seconds: float = 0.0
    source: str = "signal"
    label: str | None = None
    hypotheses: dict[str, float] | None = None
    centrality_weight: float = 0.35

    @classmethod
    def from_raw(cls, key: str, raw: Any, *, now: float) -> "Signal":
        if isinstance(raw, dict):
            ts = raw.get("timestamp") or raw.get("time")
            age = raw.get("age_seconds", raw.get("age", None))
            if age is None and ts is not None:
                age = max(0.0, now - safe_float(ts, now))
            return cls(
                name=str(raw.get("name") or key),
                score=clamp(raw.get("score", raw.get("probability", raw.get("value", 0.0)))),
                reliability=clamp(raw.get("reliability", raw.get("weight", 1.0))),
                tier=int(safe_float(raw.get("tier", raw.get("priority", 3)), 3)),
                entity=str(raw.get("entity")) if raw.get("entity") is not None else None,
                timestamp=safe_float(ts) if ts is not None else None,
                age_seconds=max(0.0, safe_float(age, 0.0)),
                source=str(raw.get("source", "signal")),
                label=str(raw.get("label")) if raw.get("label") is not None else None,
                hypotheses=normalize_weights(raw.get("hypotheses") or raw.get("stances") or raw.get("classes") or {}),
                centrality_weight=safe_float(raw.get("centrality_weight", 0.35), 0.35),
            )
        return cls(name=str(key), score=clamp(raw))

    def decayed_score(self, half_life_seconds: float) -> float:
        return decay(self.score, self.age_seconds, half_life_seconds)

    def weighted_score(self, centrality: dict[str, float], half_life_seconds: float) -> float:
        v = self.decayed_score(half_life_seconds) * self.reliability
        if self.entity is not None:
            v *= 1.0 + self.centrality_weight * clamp(centrality.get(self.entity, 0.0))
        return clamp(v)

    def hypothesis_weights(self) -> dict[str, float]:
        if self.hypotheses:
            return dict(self.hypotheses)
        name = (self.label or self.name).upper().replace(" ", "_").replace("-", "_")
        return {name: 1.0}

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "reliability": self.reliability,
            "tier": self.tier,
            "entity": self.entity,
            "age_seconds": self.age_seconds,
            "source": self.source,
            "label": self.label,
            "hypotheses": self.hypothesis_weights(),
        }


@dataclass(frozen=True)
class Config:
    strategy: str = "ensemble"
    half_life_seconds: float = 30.0
    suppression_threshold: float = 0.62
    contradiction_threshold: float = 0.55
    contradiction_margin: float = 0.18
    fuzzy_rules: tuple[dict[str, Any], ...] = ()
    ensemble_weights: dict[str, float] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any], args: argparse.Namespace) -> "Config":
        raw = dict(payload.get("config") or {})
        if args.strategy:
            raw["strategy"] = args.strategy
        if args.half_life_seconds is not None:
            raw["half_life_seconds"] = args.half_life_seconds
        ew = normalize_weights(raw.get("ensemble_weights") or {"fuzzy": 0.30, "dempster_shafer": 0.30, "priority_decay": 0.40})
        return cls(
            strategy=str(raw.get("strategy", "ensemble")),
            half_life_seconds=safe_float(raw.get("half_life_seconds", 30.0), 30.0),
            suppression_threshold=clamp(raw.get("suppression_threshold", 0.62)),
            contradiction_threshold=clamp(raw.get("contradiction_threshold", 0.55)),
            contradiction_margin=clamp(raw.get("contradiction_margin", 0.18)),
            fuzzy_rules=tuple(raw.get("fuzzy_rules") or raw.get("rules") or ()),
            ensemble_weights=ew,
        )


def parse_signals(payload: dict[str, Any], *, now: float) -> list[Signal]:
    raw = payload.get("signals", payload.get("alarms", payload.get("events", {})))
    if isinstance(raw, dict):
        return [Signal.from_raw(k, v, now=now) for k, v in raw.items()]
    if isinstance(raw, list):
        return [Signal.from_raw(str(i), v, now=now) for i, v in enumerate(raw)]
    raise ValueError("input must contain signals object or list")


def centrality_from_graph(graph: Any) -> dict[str, float]:
    if not graph:
        return {}
    if isinstance(graph, dict):
        direct = graph.get("entity_centrality") or graph.get("centrality")
        if isinstance(direct, dict):
            return {str(k): clamp(v) for k, v in direct.items()}
        edges = graph.get("edges")
        if isinstance(edges, list):
            deg: Counter[str] = Counter()
            for e in edges:
                if isinstance(e, dict):
                    a, b = e.get("source"), e.get("target")
                elif isinstance(e, (list, tuple)) and len(e) >= 2:
                    a, b = e[0], e[1]
                else:
                    continue
                if a is not None and b is not None:
                    deg[str(a)] += 1
                    deg[str(b)] += 1
            m = max(deg.values(), default=0)
            return {k: v / m for k, v in deg.items()} if m else {}
    return {}


class GenericSignalAggregator:
    def __init__(self, config: Config) -> None:
        self.config = config

    def aggregate(self, signals: list[Signal], graph: Any = None) -> dict[str, Any]:
        centrality = centrality_from_graph(graph)
        strategy = self.config.strategy
        if strategy == "fuzzy_logic":
            return self._fuzzy(signals, centrality)
        if strategy in {"dempster_shafer", "dempster-shafer", "ds"}:
            return self._dempster_shafer(signals, centrality)
        if strategy == "priority_decay":
            return self._priority_decay(signals, centrality)
        if strategy == "centrality_weighted":
            return self._centrality_weighted(signals, centrality)
        if strategy in {"max", "max_signal", "max_alarm"}:
            return self._max_signal(signals, centrality)
        return self._ensemble(signals, centrality)

    def weighted_signal_score(self, signal: Signal, centrality: dict[str, float]) -> float:
        return signal.weighted_score(centrality, self.config.half_life_seconds)

    def stance_scores(self, signals: list[Signal], centrality: dict[str, float], *, priority: bool = False) -> dict[str, float]:
        scores: defaultdict[str, float] = defaultdict(float)
        for s in signals:
            score = self.weighted_signal_score(s, centrality)
            if priority:
                score *= {0: 1.0, 1: 0.78, 2: 0.58}.get(s.tier, 0.42)
            for hyp, ratio in s.hypothesis_weights().items():
                scores[hyp] += score * ratio
        return dict(scores)

    def _receipt(
        self,
        strategy: str,
        stance: str,
        confidence: float,
        signals: list[Signal],
        centrality: dict[str, float],
        stance_scores: dict[str, float],
        *,
        conflict: float = 0.0,
        plausibility: dict[str, float] | None = None,
        suppressed: list[str] | None = None,
        components: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        queue = [
            {"priority": s.tier, "weighted_score": self.weighted_signal_score(s, centrality), "name": s.name}
            for s in signals
        ]
        queue.sort(key=lambda x: (x["priority"], -x["weighted_score"], x["name"]))
        return {
            "schema": SCHEMA,
            "verdict": "PASS",
            "strategy": strategy,
            "stance": stance,
            "confidence": clamp(confidence),
            "conflict": clamp(conflict),
            "stance_scores": {k: float(v) for k, v in sorted(stance_scores.items())},
            "plausibility": {k: float(v) for k, v in sorted((plausibility or {}).items())},
            "active_signals": [s.as_dict() for s in signals if s.score > 0.0],
            "suppressed_signals": sorted(set(suppressed or [])),
            "priority_queue": queue,
            "centrality": centrality,
            "components": components or {},
            "advisory_only": True,
        }

    def _max_signal(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        if not signals:
            return self._receipt("max_signal", DEFAULT_HOLD, 0.0, [], centrality, {DEFAULT_HOLD: 1.0})
        s = max(signals, key=lambda x: self.weighted_signal_score(x, centrality))
        scores = self.stance_scores([s], centrality)
        stance, confidence = top_score(scores)
        return self._receipt("max_signal", stance, confidence, [s], centrality, scores)

    def _centrality_weighted(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        scores = self.stance_scores(signals, centrality)
        stance, confidence = top_score(scores)
        return self._receipt("centrality_weighted", stance, confidence, signals, centrality, scores)

    def _fuzzy(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        # Generic fuzzy layer: weighted hypothesis truth plus optional named rules.
        base = self.stance_scores(signals, centrality)
        signal_truth = {s.name: self.weighted_signal_score(s, centrality) for s in signals}
        rule_scores: defaultdict[str, float] = defaultdict(float)
        for rule in self.config.fuzzy_rules:
            cond = rule.get("if", {}) if isinstance(rule, dict) else {}
            if not isinstance(cond, dict):
                continue
            truths = []
            for name, threshold in cond.items():
                val = signal_truth.get(str(name), 0.0)
                if isinstance(threshold, str):
                    need = {"low": 0.25, "medium": 0.50, "high": 0.75}.get(threshold.lower(), 0.50)
                else:
                    need = safe_float(threshold, 0.50)
                truths.append(clamp((val - need + 0.25) / 0.25))
            if truths:
                then = str(rule.get("then") or rule.get("stance") or DEFAULT_HOLD)
                rule_scores[then] += min(truths) * safe_float(rule.get("weight", 1.0), 1.0)
        combined = dict(base)
        for k, v in rule_scores.items():
            combined[k] = combined.get(k, 0.0) + v
        stance, confidence = top_score(combined)
        return self._receipt("fuzzy_logic", stance, confidence, signals, centrality, combined)

    def _signal_mass(self, signal: Signal, centrality: dict[str, float], all_hypotheses: tuple[str, ...]) -> dict[frozenset[str], float]:
        score = self.weighted_signal_score(signal, centrality)
        weights = signal.hypothesis_weights()
        committed = min(0.96, score * signal.reliability)
        masses = {frozenset([h]): committed * w for h, w in weights.items() if h in all_hypotheses}
        masses[frozenset(all_hypotheses)] = max(0.0, 1.0 - sum(masses.values()))
        return masses

    def _combine_mass(self, a: dict[frozenset[str], float], b: dict[frozenset[str], float]) -> tuple[dict[frozenset[str], float], float]:
        out: defaultdict[frozenset[str], float] = defaultdict(float)
        conflict = 0.0
        for ha, ma in a.items():
            for hb, mb in b.items():
                inter = ha & hb
                if inter:
                    out[inter] += ma * mb
                else:
                    conflict += ma * mb
        norm = max(1e-9, 1.0 - conflict)
        return {k: v / norm for k, v in out.items()}, conflict

    def _dempster_shafer(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        hyps = sorted({h for s in signals for h in s.hypothesis_weights()}) or [DEFAULT_HOLD]
        all_h = tuple(hyps)
        mass: dict[frozenset[str], float] = {frozenset(all_h): 1.0}
        conflicts = []
        for s in signals:
            mass, conflict = self._combine_mass(mass, self._signal_mass(s, centrality, all_h))
            conflicts.append(conflict)
        belief = {h: mass.get(frozenset([h]), 0.0) for h in all_h}
        plausibility = {h: sum(v for m, v in mass.items() if h in m) for h in all_h}
        stance, confidence = top_score(belief)
        vals = sorted(belief.values(), reverse=True)
        conflict_score = max(conflicts, default=0.0)
        if conflict_score >= self.config.contradiction_threshold and len(vals) > 1 and vals[0] - vals[1] <= self.config.contradiction_margin:
            stance, confidence = CONTRADICTION, conflict_score
        return self._receipt("dempster_shafer", stance, confidence, signals, centrality, belief, conflict=conflict_score, plausibility=plausibility)

    def _priority_decay(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        hot = [s for s in signals if self.weighted_signal_score(s, centrality) >= 0.05]
        if not hot:
            return self._receipt("priority_decay", DEFAULT_HOLD, 0.0, [], centrality, {DEFAULT_HOLD: 1.0})
        min_tier = min(s.tier for s in hot if self.weighted_signal_score(s, centrality) >= self.config.suppression_threshold) if any(self.weighted_signal_score(s, centrality) >= self.config.suppression_threshold for s in hot) else None
        kept, suppressed = [], []
        for s in hot:
            if min_tier is not None and s.tier > min_tier:
                suppressed.append(s.name)
            else:
                kept.append(s)
        scores = self.stance_scores(kept, centrality, priority=True)
        stance, confidence = top_score(scores)
        return self._receipt("priority_decay", stance, confidence, kept, centrality, scores, suppressed=suppressed)

    def _ensemble(self, signals: list[Signal], centrality: dict[str, float]) -> dict[str, Any]:
        fuzzy = self._fuzzy(signals, centrality)
        ds = self._dempster_shafer(signals, centrality)
        prio = self._priority_decay(signals, centrality)
        weights = self.config.ensemble_weights or {"fuzzy": 0.30, "dempster_shafer": 0.30, "priority_decay": 0.40}
        votes: defaultdict[str, float] = defaultdict(float)
        for key, rec in (("fuzzy", fuzzy), ("dempster_shafer", ds), ("priority_decay", prio)):
            stance = rec.get("stance", DEFAULT_HOLD)
            if stance == CONTRADICTION:
                continue
            votes[str(stance)] += weights.get(key, 0.0) * safe_float(rec.get("confidence"), 0.0)
        if ds.get("stance") == CONTRADICTION and safe_float(ds.get("conflict")) >= self.config.contradiction_threshold:
            stance, confidence = CONTRADICTION, safe_float(ds.get("conflict"))
        else:
            stance, confidence = top_score(votes)
        return self._receipt(
            "ensemble",
            stance,
            confidence,
            signals,
            centrality,
            dict(votes),
            conflict=safe_float(ds.get("conflict")),
            suppressed=prio.get("suppressed_signals", []),
            components={"fuzzy_logic": fuzzy, "dempster_shafer": ds, "priority_decay": prio},
        )


def load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.input == "-":
        text = sys.stdin.read()
    elif args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    elif args.signals:
        text = args.signals
    else:
        text = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not text.strip():
        raise SystemExit("No input. Pass --input file.json, --signals JSON, or pipe JSON on stdin.")
    payload = json.loads(text)
    if isinstance(payload, list):
        payload = {"signals": payload}
    if not isinstance(payload, dict):
        raise SystemExit("Input must be a JSON object or signal list")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generic fuzzy/Dempster-Shafer/centrality/priority-decay signal aggregator")
    ap.add_argument("--input", "-i", help="Input JSON file, or '-' for stdin")
    ap.add_argument("--signals", help="Inline JSON payload")
    ap.add_argument("--strategy", choices=["ensemble", "fuzzy_logic", "dempster_shafer", "priority_decay", "centrality_weighted", "max_signal"], help="Override strategy")
    ap.add_argument("--half-life-seconds", type=float, default=None, help="Override exponential decay half-life")
    ap.add_argument("--out", "-o", help="Write JSON receipt to file instead of stdout")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--example", action="store_true", help="Print a sample payload and exit")
    args = ap.parse_args(argv)
    if args.example:
        example = {
            "signals": {
                "friction_overload": {"score": 0.85, "reliability": 0.9, "tier": 1, "entity": "node-a", "hypotheses": {"CONSOLIDATE": 0.7, "RECOVER": 0.3}},
                "extraction_window": {"score": 0.40, "reliability": 0.8, "tier": 2, "entity": "node-b", "hypotheses": {"EXPLOIT": 1.0}},
            },
            "graph": {"entity_centrality": {"node-a": 0.8, "node-b": 0.2}},
            "config": {"strategy": "ensemble", "half_life_seconds": 30},
        }
        print(json.dumps(example, indent=2, sort_keys=True))
        return 0
    payload = load_payload(args)
    now = safe_float(payload.get("now"), time.time())
    cfg = Config.from_payload(payload, args)
    signals = parse_signals(payload, now=now)
    rec = GenericSignalAggregator(cfg).aggregate(signals, payload.get("graph") or payload.get("local_graph_subset"))
    rec["input_signal_count"] = len(signals)
    rec["config"] = {
        "strategy": cfg.strategy,
        "half_life_seconds": cfg.half_life_seconds,
        "suppression_threshold": cfg.suppression_threshold,
        "contradiction_threshold": cfg.contradiction_threshold,
        "contradiction_margin": cfg.contradiction_margin,
        "ensemble_weights": cfg.ensemble_weights,
    }
    text = json.dumps(rec, indent=2 if args.pretty else None, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
