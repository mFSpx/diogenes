# Possum / OPOSSUM / Capybara Algorithm Primitive Intake

Date: 2026-05-13

Operator supplied five more local algorithm primitives. Wrapped as dependency-light Python modules under `ALGOS/` and wired into `scripts/lucidota_algos_smoke.py`.

## Added modules

- `ALGOS/possum_filter.py` — local-diversity suppression for near-duplicate entities using Haversine distance plus category/address signature.
- `ALGOS/rbf_surrogate.py` — OPOSSUM-style radial-basis-function surrogate fitting and candidate grid search.
- `ALGOS/thanatosis.py` — simulated-annealing/dormancy acceptance probability and cooling decision.
- `ALGOS/capybara_optimization.py` — COA social-interaction and predator-evasion movement primitives.
- `ALGOS/hoeffding_tree.py` — CapyMOA/MOA-style Hoeffding bound and stream split decision helper.

## Wire status

- Smoke harness now exercises 20 algorithm modules.
- These are local primitives only: no Drive access, no external runtime dependency, no model weights.
- Candidate uses:
  - Possum filter: dedupe/prune pivot candidates by spatial/category similarity when location metadata exists.
  - RBF surrogate: cheap policy scoring approximation for expensive workflow choices.
  - Thanatosis: dormancy / backoff policy for stuck agents or noisy pivots.
  - Capybara movement: population search strategy for route/model/hyperparameter sweeps.
  - Hoeffding bound: stream split gate for future online classifier nodes.
