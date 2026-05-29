# Knowledge Card: CyberCrafter Deterministic Reasoning Framework (DRF)

- ID: `cybercrafter_drf`
- Authority class: `research_reference`
- Source: https://gitlab.com/cybercrafter/drf
- Local clone: `01_REPOS/cybercrafter_drf/`
- Cloned commit: `214ae660d44fb072d9bdad27444a815cd3beab54`
- License file: `01_REPOS/cybercrafter_drf/LICENSE`

## What it is

DRF is a deterministic decision layer intended to sit beside LLM agents. It uses the model for semantic understanding, evidence/criteria extraction, and presentation, while deterministic algorithms perform final ranking/classification. This aligns strongly with LUCIDOTA's deterministic-before-LLM routing law without becoming a global pseudolaw by itself.

## Core architecture learned

1. Split semantic extraction from decision authority.
   - LLM: intent/evidence/criteria extraction and response formatting.
   - Deterministic tools: MCDA, GAM, GP/symbolic regression, decision trees, statistical models.

2. Persist learned deterministic decisions as artifacts.
   - DRF has XAI/discovery modes that produce reusable JSON artifacts.
   - Inference then uses artifacts at near-zero latency and reproducible output.

3. Compare stochastic baselines against deterministic methods.
   - KAPPA mode repeats LLM and MCDA runs and scores reproducibility with Cohen's kappa and Spearman correlation.
   - Reported DRF result: MCDA methods achieve perfect ranking consistency on the vehicle experiment while LLM rankings vary even at temperature zero.

4. Bias audit lesson.
   - In synthetic hiring, deterministic merit-based MCDA stays balanced when merits are equal.
   - LLM selections can concentrate along proxy demographic signals even when protected attributes are removed.

## Repository map

- `drf.py` — main CLI/controller; modes include baseline, KAPPA, XAI, artifact-backed inference.
- `core/mcda.py` — deterministic MCDA algorithms: TOPSIS, VIKOR, WSM, WPM, WASPAS, ARAS, MABAC, CODAS, EDAS, MOORA, ELECTRE-TRI, PromSort, FlowSort.
- `core/reasoners.py` — LLM, MCDA, GP, GAM, statistical, and tree reasoner wrappers.
- `core/reverse_engineering.py` — weight/model discovery methods: GA, PSO, differential evolution, SHAP/LIME-style importance, NSGA-II, CMAES, etc.
- `core/statistical.py` / `core/decision_tree.py` — deterministic classifiers/regressors.
- `core/dataclass.py` / `core/utility.py` — dataset configs, artifact serialization, memory helpers, metrics.
- `report_makers/` — analysis/report generators for kappa, classification, reverse-engineering, and bias experiments.
- `datasets/` — cars, WDBC, triage, candidates, malware/network datasets.
- `experiments/` — essay/bias outputs; clone contains 140 JSON experiment/artifact files and 101 PNG figures.

## LUCIDOTA integration stance

Keep DRF as a research/reference pattern, not hard law. Use it to shape future deterministic routing and fairness checks:

- Add a LUCIDOTA `deterministic_decision_layer` adapter only after identifying a narrow local decision problem (queue priority, candidate routing, graph promotion triage, or model-router selection).
- Use DRF-style reproducibility metrics for slop audits: repeated LLM route choice should be compared against deterministic router output.
- Treat any LLM-only high-impact ranking/classification as incomplete until a deterministic artifact or rule baseline exists.
- Consider a future `drf-inspired` micro-adapter rather than importing the whole repo directly; DRF's main controller is large and currently has external runtime dependencies.

## Verification / blockers

- Local syntax compile passed for `drf.py`, `core/*.py`, and `report_makers/*.py`.
- `python3 drf.py --help` is blocked in this environment by missing Python package `ollama`; do not claim runtime verification until dependencies are installed intentionally.
- Clone emitted Git LFS pointer warnings for PNG/media files; source code and README are present and inspectable.
