# ALGOS — Deterministic Algorithm Arsenal

This directory contains ~80+ algorithms that rank, gate, score, and route. They cannot write truth.

## Key Rules

- **Algorithms can rank/gate/score/route; they cannot write canonical truth.**
- **No network calls.** Pure local execution only.
- **No database writes from algorithms.** Results go to the caller, who decides what to persist.
- **Deterministic by default.** Randomness must be seeded and documented.
- **Each algorithm must have a schema declaration** in its docstring or a companion JSON.

## Categories

- `rete_bandit_gate.py` — RETE + bandit routing (the main gate)
- `percyphon*` — identity system (128-slot xxhash128)
- `hoeffding_tree.py` — LiteTree decision trees
- `ternary*` — ternary routing
- `ltc.py` — Liquid Time Constant
- `evolved/` — evolutionary algorithm survivors (GAUNTLET)
- `distilled/` — distilled/compressed algorithms
