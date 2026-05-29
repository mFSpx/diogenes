# ALGO_MANIFEST — LUCIDOTA Algorithm Arsenal

> **One-line purpose:** The single, readable map of every algorithm implementation in the repo, bucketed by what it actually *does*, so you can pick the right primitive when designing a data-ingestion workflow.
>
> **Generated:** 2026-05-28
> **Files cataloged:** ~84 implementation files (`ALGOS/` = 57, `pypeline/math/` = 17 unique, algo-bearing `scripts/` = ~10). Most files expose *multiple* named primitives (e.g. `bandit_router` ships LinUCB + Thompson + epsilon-greedy; `sketches` ships Count-Min + HLL + MinHash-LSH), so the operator-felt **200+ algorithm functions** is real — this manifest indexes them at the file/primitive level.
>
> **Doctrine reminder (CLAUDE.md):** Algorithms can **rank / gate / score / route** — they **cannot write canonical graph truth.** Lane column tells you where each runs: `ALGO` = pure deterministic primitive (`ALGOS/`), `MATH` = pypeline math layer, `SCRIPT` = worker/gate/governor lane, `SQL` = belongs in the contract layer.

---

## 1. Routing / Selection (bandits, gates, routers)

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Bandit Router | `ALGOS/bandit_router.py` | LinUCB / Thompson-sampling / epsilon-greedy action selection with reward update | ALGO |
| RETE Bandit Gate | `ALGOS/rete_bandit_gate.py` | RETE forward-chaining deterministic prune + bandit/regret routing to fast/slow lane | ALGO |
| RETE Bandit Gate CLI | `scripts/rete_bandit_gate_cli.py` | CLI wrapper exposing the fastlane/slowlane router to workers | SCRIPT |
| Ternary Router | `ALGOS/ternary_router.py` | Always-on FairyFuse ternary (-1/0/+1) packet router with daemon mode | ALGO |
| Ternary Lens Router | `ALGOS/ternary_lens_router.py` | Command-envelope ternary router scaffold; payload-hash → primitive id → confidence bps | ALGO |
| Ternary Lens Audit | `ALGOS/ternary_lens_audit.py` | Offline audit enforcing the ternary fast-path rule over a manifest | ALGO |
| Tri-Algo Conduit | `ALGOS/tri_algo_conduit.py` | Pipeline: passive monitor → Hoeffding gate → self-righting recovery decision | ALGO |
| Workshare Allocator | `ALGOS/workshare_allocator.py` | Deterministic split of work across local/remote engines + savings summary | ALGO |
| Distributed Leader Election | `ALGOS/distributed_leader_election.py` | Randomized local leader election / maximal independent set over graph neighborhoods | ALGO |
| Voronoi Partition | `ALGOS/voronoi_partition.py` | Nearest-seed space partitioning / assignment | ALGO |
| Endpoint Circuit Breaker / Pool | `ALGOS/endpoint_circuit_breaker.py` | Circuit-breaker + dual-engine endpoint pool selection (open/half/closed) | ALGO |

## 2. Online / Streaming Learning

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Hoeffding Tree | `ALGOS/hoeffding_tree.py` | MOA-style Hoeffding-bound split decision for online incremental decision trees | ALGO |
| NLMS | `ALGOS/nlms.py` | Normalized least-mean-squares adaptive online linear predictor/update | ALGO |
| Fold-Change Detection | `ALGOS/fold_change_detection.py` | Streaming fold-change response (ratio-based adaptation) update equations | ALGO |
| River Governor | `scripts/lucidota_river_governor.py` | River-ML online governor — system learns its own resource/throughput limits | SCRIPT |
| ABSURD River Worker | `scripts/absurd_river_worker.py` | Queue-spine wrapper running River/Bytewax stream + GLiNER extraction | SCRIPT |
| River Reflex (legacy) | `scripts/legacy/lucidota_river_reflex.py` | Earlier River reflex loop (legacy provenance) | SCRIPT |
| Honeybee Store | `ALGOS/honeybee_store.py` | Decentralized common-store feedback for resource rate control ("dance duration") | ALGO |
| Pheromone | `ALGOS/pheromone.py` | Darwinian surface pheromone deposit + decay signal worker | ALGO |
| Physarum Network | `ALGOS/physarum_network.py` | Slime-mold flux-based conductance reinforcement update | ALGO |

## 3. Backoff / Recovery / Resilience / Resource Control

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Possum Filter | `ALGOS/possum_filter.py` | Local diversity/quiescence filter — drop near-duplicate candidates by signature+geo | ALGO |
| Serpentina Self-Righting | `ALGOS/serpentina_self_righting.py` | Morphology-based recovery-priority scoring (righting time from sphericity/flatness) | ALGO |
| Thanatosis | `ALGOS/thanatosis.py` | Simulated-annealing dormancy: acceptance prob + cooling schedule for back-off | ALGO |
| Decreasing Pruning | `ALGOS/decreasing_pruning.py` | Decreasing-rate pruning schedule (anneal edge/candidate retention) | ALGO |
| Resource Governor | `scripts/resource_governor.py` | Process collar — caps CPU/VRAM, registers pids, enforces resource policy | SCRIPT |
| Model VRAM Scheduler | `pypeline/math/model_vram_scheduler.py` | VRAM slot planning, dual-engine residency, preemption + flash eviction | MATH |
| Model Pool | `pypeline/math/model_pool.py` | Tiered model load/evict pool manager | MATH |
| Model Governor | `scripts/lucidota_model_governor.py` | VRAM/loadout governor for the model stack | SCRIPT |
| Feedback Governor | `scripts/lucidota_feedback_governor.py` | Feedback-loop governor over runtime dials | SCRIPT |
| Endpoint Health | `pypeline/math/endpoint_health.py` | Circuit-breaker + pool health for provider endpoints | MATH |
| Capybara Optimization | `ALGOS/capybara_optimization.py` | Predator-evasion / social-interaction movement primitive (metaheuristic) | ALGO |
| RBF Surrogate | `ALGOS/rbf_surrogate.py` | OPOSSUM-style radial-basis surrogate + grid search for cheap optimization | ALGO |

## 4. Feature Attribution / Scoring / Regret

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| SHAP / Shapley Attribution | `ALGOS/shap_attribution.py` | Exact + kernel Shapley values, tree-SHAP tensor, mean-abs-SHAP per feature | ALGO |
| Regret Engine | `ALGOS/regret_engine.py` / `pypeline/math/regret_engine.py` | Regret-weighted strategy + expected-value action ranking | ALGO/MATH |
| XGBoost Objective | `ALGOS/xgboost_objective.py` | Binary-logistic grad/hess, optimal leaf weight, split gain, sklearn wrapper | ALGO |
| Counterfactual Effects | `ALGOS/counterfactual_effects.py` | Causal/counterfactual effect estimate + refutation suite | ALGO |
| Decision Hygiene | `ALGOS/decision_hygiene.py` | Decision-quality scoring (feature counts, half-vs-half comparison) | ALGO |
| Cockpit Metrics | `ALGOS/cockpit_metrics.py` | Anti-slop ratio, cockpit honesty, audit-debt evidence-coverage metrics | ALGO |
| Fisher Localization | `ALGOS/fisher_localization.py` | Fisher-information scoring / best-angle for off-axis sensing | ALGO |
| Claim Support Score | `scripts/claim_support_score.py` | Score how well evidence supports a candidate claim | SCRIPT |
| Hard-Truth Math | `ALGOS/hard_truth_math.py` | Stylometry (LSM) features + hinge classifier for persona/honesty telemetry | ALGO |
| Epistemic Certainty | `pypeline/math/epistemic_certainty.py` | Certainty flagging across observation/extraction/hypothesis/contradiction | MATH |

## 5. Game-Theoretic / Allocation

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Regret-Weighted Strategy | `ALGOS/regret_engine.py` | Counterfactual-regret minimization style strategy weighting (CFR-lite) | ALGO |
| Minimum-Cost Tree | `ALGOS/minimum_cost_tree.py` | Length/path trade-off tree-cost scoring (Steiner-ish allocation) | ALGO |
| Workshare Allocator | `ALGOS/workshare_allocator.py` | Deterministic budget split (Blotto-style resource allocation across fronts) | ALGO |

## 6. Matching / Dedup / Hashing / Sketches

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| MinHash | `ALGOS/minhash.py` | MinHash signatures for approximate Jaccard set similarity | ALGO |
| Sketches | `ALGOS/sketches.py` | Count-Min sketch, HyperLogLog cardinality, MinHash-LSH index | ALGO |
| Perceptual Dedupe | `ALGOS/perceptual_dedupe.py` | dHash/pHash perceptual hashing + Hamming clustering for image/evidence dedup | ALGO |
| SSIM | `ALGOS/ssim.py` | Structural similarity index between grayscale samples | ALGO |
| Sparse WTA | `ALGOS/sparse_wta.py` | Sparse winner-take-all tags + Hamming for high-dim similarity | ALGO |
| HDC | `ALGOS/hdc.py` | Hyperdimensional computing: bind/bundle/permute/similarity on bipolar vectors | ALGO |
| Semantic Neighbors | `ALGOS/semantic_neighbors.py` | In-memory cosine semantic-neighbor enclave | ALGO |
| Korpus Text | `ALGOS/korpus_text.py` | Low-level text math: minhash, entropy, CKDOG vector literals | ALGO |
| Privacy | `ALGOS/privacy.py` | Reconstruction-risk score, anonymize-for-indexing, DP aggregate | ALGO |
| Claim Clusterer | `scripts/claim_clusterer.py` | Cluster near-duplicate/related claims | SCRIPT |

## 7. Graph / RETE / Reasoning

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| RETE Prune (in gate) | `ALGOS/rete_bandit_gate.py` | RETE-style forward-chaining deterministic node prune before routing | ALGO |
| Bayes Update | `ALGOS/bayes_update.py` / `pypeline/math/algos/bayes_update.py` | Bayesian marginal + posterior evidence update primitive | ALGO |
| Bayes Claim Kernel | `pypeline/math/bayes_claim_kernel.py` | Hypothesis posterior update + log-likelihood-ratio for claim reasoning | MATH |
| Krampus Brainmap | `ALGOS/krampus_brainmap.py` | Feature extraction + 3D brain-xyz projection of documents | ALGO |
| Indy Learning Vector | `ALGOS/indy_learning_vector.py` | Ontology-hit deterministic learning vectors + LoRA row builder | ALGO |
| Label Foundry | `ALGOS/label_foundry.py` | Weak-supervision labeling functions, probabilistic label aggregation, error find | ALGO |
| Validators | `pypeline/math/validators.py` | Registered numbered equation validators (EQ001…) for math claims | MATH |
| Validators Extended | `pypeline/math/validators_extended.py` | Additional registered validator equations (ratio/regression checks) | MATH |
| Math Types | `pypeline/math/types.py` | Typed Claim/Evidence/Hypothesis/Action/Counterfactual contracts | MATH |

## 8. Ranking / Reranking

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| EV Action Ranking | `ALGOS/regret_engine.py` | Rank actions by expected value | ALGO |
| Chrono Ranking Pass | `scripts/chrono_ranking_pass.py` | Build immutable chronological ranking-pass selections | SCRIPT |
| Mean-Abs-SHAP Ranking | `ALGOS/shap_attribution.py` | Rank features by mean absolute SHAP | ALGO |
| Gini Coefficient | `ALGOS/gini_coefficient.py` / `pypeline/math/algos/gini_coefficient.py` | Inequality coefficient (ranking-spread / fairness metric) | ALGO |

## 9. Sampling / Search / Information-Seeking

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Infotaxis | `ALGOS/infotaxis.py` | Gradient-free entropy-minimizing search (best-action by expected entropy drop) | ALGO |
| Shannon Entropy | `ALGOS/shannon_entropy.py` / `pypeline/math/algos/shannon_entropy.py` | Shannon entropy of observations / distributions | ALGO |
| Thompson Sampling | `ALGOS/bandit_router.py` | Posterior-sampling exploration arm (inside bandit router) | ALGO |
| RBF Grid Search | `ALGOS/rbf_surrogate.py` | Surrogate-guided grid search over a parameter space | ALGO |
| Percyphon | `ALGOS/percyphon.py` | Zero-VRAM procedural entity generator (deterministic sampling from SHA seeds) | ALGO |

## 10. Temporal / Burst / Capture

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Temporal Motifs | `ALGOS/temporal_motifs.py` | Sessionize events, detect bursts, mine temporal motifs | ALGO |
| Chelydrid Ambush | `ALGOS/chelydrid_ambush.py` | Ambush-strike kinematics + burst-admission score (burst capture gating) | ALGO |
| Krampus Chrono | `ALGOS/krampus_chrono.py` | Loose datetime parse, chrono candidates, circadian activity curve/match | ALGO |
| Poikilotherm Schoolfield | `ALGOS/poikilotherm_schoolfield.py` | Temperature-dependent rate (Schoolfield-Rollinson) for throttling/timing | ALGO |
| Doomsday Calendar | `ALGOS/doomsday_calendar.py` | Doomsday-rule weekday computation | ALGO |

## 11. Misc / Uncategorized

| Algorithm | File | Literal functionality | Lane |
|---|---|---|---|
| Krampus Stickers | `ALGOS/krampus_stickers.py` | Cognitive-document feature stickers (vibes/psyche/resilience extraction) | ALGO |
| Omni Chaotic Sprint | `ALGOS/omni_chaotic_sprint.py` | Seismic ray-tracer + fluidic triage + non-parametric triad text compiler | ALGO |
| RSA Cipher | `ALGOS/rsa_cipher.py` | Textbook RSA integer encrypt/decrypt — demos/tests only, not prod crypto | ALGO |
| Prompt Injection | `pypeline/math/prompt_injection.py` | Detect + neutralize prompt-injection patterns in ingested text | MATH |
| GLiNER Zero-Shot Extractor | `ALGOS/gliner_zero_shot_extractor.py` | Zero-shot span/entity extraction with local fallback | ALGO |
| Small Model Zoo | `pypeline/math/small_model_zoo.py` | SetFit / PEFT-LoRA train + ONNX quantize registry of tiny models | MATH |
| Golden Path Regression Gate | `scripts/golden_path_regression_gate.py` | Regression gate over the golden-path scenario suite | SCRIPT |

---

## INGESTION-RELEVANT SHORTLIST

When designing a **data-ingestion workflow**, reach for these first:

**Route batches to fast/slow lanes & pick engines**
- `ALGOS/rete_bandit_gate.py` — the canonical ingest router: RETE prune → bandit/regret → fast/slow lane + engine plan. **Start here.**
- `ALGOS/bandit_router.py` — adaptive arm selection when you have reward feedback on routing choices.
- `ALGOS/ternary_router.py` / `ternary_lens_router.py` — lightweight always-on packet routing by ternary signal.
- `ALGOS/workshare_allocator.py` — split a batch deterministically across local/remote engines.

**Back off & survive load spikes**
- `ALGOS/thanatosis.py` — annealing dormancy / acceptance back-off under pressure.
- `ALGOS/possum_filter.py` — drop near-dup candidates early (quiescence) so you ingest less under load.
- `scripts/resource_governor.py` + `pypeline/math/model_vram_scheduler.py` — hard CPU/VRAM collars on workers.
- `scripts/lucidota_river_governor.py` — online learning of the system's own throughput limits.
- `ALGOS/endpoint_circuit_breaker.py` / `pypeline/math/endpoint_health.py` — trip breakers on flaky source/provider endpoints.

**Dedup incoming artifacts**
- `ALGOS/minhash.py` + `ALGOS/sketches.py` (MinHash-LSH, Count-Min, HLL) — near-dup text + cardinality at scale.
- `ALGOS/perceptual_dedupe.py` + `ALGOS/ssim.py` — image/visual-channel dedup.
- `ALGOS/possum_filter.py` — signature+geo diversity filter on entity candidates.
- `scripts/claim_clusterer.py` — collapse related/duplicate claims.

**Dead-letter recovery & resilience**
- `ALGOS/serpentina_self_righting.py` — recovery-priority scoring for stuck/failed items.
- `ALGOS/tri_algo_conduit.py` — monitor → Hoeffding gate → self-right, end-to-end recovery decision.
- `ALGOS/decreasing_pruning.py` — anneal retention of retried candidates.

**Score / rank / gate candidates before promotion**
- `ALGOS/shap_attribution.py` + `ALGOS/xgboost_objective.py` — feature attribution + gradient-boosted scoring.
- `ALGOS/regret_engine.py` — rank candidate actions by expected value / regret.
- `pypeline/math/epistemic_certainty.py` + `pypeline/math/bayes_claim_kernel.py` — certainty flags + Bayesian belief update before authority/graph gate.
- `scripts/claim_support_score.py` — evidence-support scoring for claim candidates.
- `ALGOS/cockpit_metrics.py` — anti-slop / honesty coverage check on the batch.

**Streaming / online ingest**
- `ALGOS/hoeffding_tree.py` — online incremental classification of incoming stream.
- `scripts/absurd_river_worker.py` — the ABSURD queue-spine River/Bytewax + GLiNER stream worker.

> **Gate reminder:** every one of these *ranks/gates/scores/routes*. None of them write canonical graph truth — promotion still goes through evidence → authority class → graph-promotion preflight → command envelope (CLAUDE.md execution spine).
