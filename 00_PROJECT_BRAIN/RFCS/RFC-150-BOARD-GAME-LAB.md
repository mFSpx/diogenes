# RFC-150: Board-Game / Simulation Lab

Status: DRAFT  
Subject ID: `board_game_lab`  
Normative role: defines board-game and simulation work as an externalized laboratory for rules, agents, search, training, and ontology experiments, not as active production truth-spine code.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

The board-game lab is real and valuable, but it belongs outside the active LUCIDOTA repo. Ahoy proved the pattern: a rules-heavy game simulation can produce legal-action engines, policy legality checks, training datasets, River/XGBoost/Treelite experiments, ontology adapters, and strategy receipts. That is useful. It is also not the main spine.

The correct relationship is: board games are simulation sandboxes for learning decision mechanics, adversarial planning, policy legality, leakage checks, and strategy evaluation. They may donate hardened patterns back to LUCIDOTA. They must not sit in the active repo as if their local game ontology, tests, schemas, or datasets are production doctrine.

## 2. Sources

### Local sources

- `/home/mfspx/BOARD_GAMES/AHOY/` — externalized Ahoy lab containing `ahoy_sim`, tests, policies, rules, training, ontology adapters, signals, and schemas.
- `05_OUTPUTS/filesystem_organization/root414_ahoy_extraction_20260524T005734Z.json` and `ahoy_residual_extraction_20260524T010132Z.json` — receipts proving Ahoy was moved out of the active repo.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — classifies board-game/simulation lab as a sandbox organ rather than core runtime truth.
- `00_PROJECT_BRAIN/RFCS/RFC-070-KRAMPUS-KORPUS.md` — requires Ahoy to remain outside active KORPUS/main-spine gravity unless explicitly promoted.
- Dev Library search evidence for `board`, `decision`, and `regret` — shows adjacent dashboard, decision, regret, and graph tools to reuse rather than reinvent.

### External Source Anchors

- OpenSpiel frames games as a research framework for reinforcement learning, game theory, and search; that supports treating game engines as laboratories for algorithms and evaluations: <https://arxiv.org/abs/1908.09453>.
- Monte Carlo Tree Search literature treats games as a major domain for search/planning method development; that supports simulations as method labs rather than final truth systems: <https://cris.maastrichtuniversity.nl/en/publications/a-survey-of-monte-carlo-tree-search-methods/>.
- AlphaZero-style self-play shows how strict game rules can generate strong policies from simulation; LUCIDOTA can borrow the lesson without pretending board-game policies are OSINT facts: <https://arxiv.org/abs/1712.01815>.
- Blueprint-first design supports explicit rule engines and workflows before model interpretation: <https://arxiv.org/abs/2508.02721>.

## 3. Lab Contract

A board-game lab MAY contain:

- rules engines,
- legal-action validators,
- turn/event logs,
- search policies,
- self-play or batch simulations,
- strategy datasets,
- leakage checks,
- ontology adapters,
- model export experiments,
- receipts and tests.

A board-game lab MUST NOT:

- mutate active GO graph truth,
- redefine active ontology terms,
- live inside the active repo as production code,
- smuggle game-specific assumptions into OSINT/investigation workflows,
- overwrite sovereign originals,
- be treated as proof about real-world actors.

Promotion is allowed only by copy/adapt/harden: take a pattern, not the whole lab gravity.

## 4. Whole-System Interaction

- **Dev Library:** indexes board-game lab patterns as reusable prior art.
- **KORPUS:** can ingest lab outputs as archived/reference artifacts, not production truth.
- **Constant learning:** can reuse simulation techniques, leakage checks, offline training discipline, and policy evaluation patterns.
- **ABBA63:** simulations can test abductive strategies and regret-aware decisions without risking real external effects.
- **Ontology:** game ontology adapters can teach mapping mechanics, but GO remains active ontology.
- **Artifact templates:** can turn simulation runs into reports/receipts.
- **Filesystem law:** requires `/home/mfspx/BOARD_GAMES/` as the external home for labs.

## 5. Benefit to the Whole System

The lab gives LUCIDOTA a safe arena for adversarial and strategic experimentation. It lets the operator build weird, hard, complete systems — rules, policies, signals, training, audits — without corrupting the main intelligence pipeline. That is exactly the right place for high-energy invention.

The benefit is transfer learning at the engineering-pattern level: legal action filters, event logs, leakage checks, search policies, simulation receipts, strategy explainers, and model export contracts can all inform LUCIDOTA without pretending a pirate-board-game state is an investigation state.

## 6. Correctness Argument

I believe this RFC is correct because the local filesystem now proves the intended boundary. The active repo no longer contains Ahoy's implementation tree; `/home/mfspx/BOARD_GAMES/AHOY/` contains the lab. The remaining receipts document the extraction. The lab itself shows real engineering depth — rules, tests, policies, training, ontology adapters — which proves it was not junk. Its externalized path proves it was also not main-spine code.

The correctness property is separation with reuse. If a board-game artifact is useful, promote the pattern or hardened copy. Do not drag the entire lab back into production gravity.

## 7. Falsifiers

This RFC is wrong if:

- active LUCIDOTA runtime imports Ahoy modules from the active repo,
- board-game schemas/tests reappear under active `06_SCHEMA/` or `tests/` without promotion receipts,
- game ontology terms override GO terms,
- simulation outputs are cited as real-world evidence,
- no practical pattern can be reused from the lab,
- the externalized folder lacks enough content to preserve the lab.

## 8. Filesystem / Runtime Consequences

- Board-game labs live under `/home/mfspx/BOARD_GAMES/`.
- Active repo may keep only pointer docs/RFCs/receipts unless a component is explicitly promoted.
- Promotion must copy/adapt/harden into a production lane with tests and receipts.
- `rfc_program_check.py` keeps Ahoy active-repo paths as boundary violations.
- Future game labs should get their own folders under `/home/mfspx/BOARD_GAMES/<NAME>/`.
