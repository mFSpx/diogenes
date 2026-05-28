# TICKLETRUNK — LUCIDOTA PROOF HOARD

Canonical manifest and access layer for the operator's proof hoard: every sovereign tool, algo, model, workflow, book, LoRA, scraper, skill, plugin, service, surface, schema, reusable fragment, and weird experimental instrument in the filesystem.

## Hard Law Summary

- TICKLETRUNK makes the proof hoard findable, not respectable.
- The scattered folders are not junk, dead code, or production dependencies merely because they exist.
- Before writing new tools, search TICKLETRUNK first and copy/adapt from existing work when useful.
- Never delete, rename, move, normalize, or production-gate sovereign toolbox artifacts without explicit operator instruction naming the exact target.
- If an artifact graduates, copy/adapt it into production and harden the copy; the original remains sovereign.

## Regenerate

```bash
python3 scripts/tickletrunk_scan.py --execute
```

## Query Examples

```bash
python3 scripts/tickletrunk_scan.py --query krampus
python3 scripts/tickletrunk_scan.py --category ALGOS
python3 scripts/tickletrunk_scan.py --list
```

## Category Table

| Category | Count |
|---|---:|
| ALGOS | 59 |
| SCRIPTS | 727 |
| MODELS | 33 |
| LORAS | 80 |
| SCHEMAS | 13 |
| SKILLS | 39 |
| PLUGINS | 11 |
| SERVICES | 6 |
| BOOKS | 18 |
| SURFACES | 6 |
| SCRAPERS | 12 |
| WORKFLOWS | 129 |
| REPOS | 121 |
| RUNTIME | 61 |
| VAULT | 27 |
| OTHER | 214 |

## ALGOS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| ALGOS | algo | sandbox | experiment | `ALGOS` | UNKNOWN — needs operator label; filename suggests: ALGOS |
| __init__.py | algo | sandbox | experiment | `ALGOS/__init__.py` | Local algorithm primitives for LUCIDOTA. |
| bandit_router.py | algo | sandbox | experiment | `ALGOS/bandit_router.py` | LinUCB/Thompson/epsilon-greedy-lite action router. |
| bayes_update.py | algo | sandbox | experiment | `ALGOS/bayes_update.py` | Bayesian evidence update primitives. |
| capybara_optimization.py | algo | sandbox | experiment | `ALGOS/capybara_optimization.py` | Capybara Optimization Algorithm movement primitives. |
| chelydrid_ambush.py | algo | sandbox | experiment | `ALGOS/chelydrid_ambush.py` | Chelydrid ambush-strike kinematics primitive. Biology origin: neck-force acceleration opposed by quadratic drag. LUCIDOTA use: burst/action admission model: short decisive actions accelerate quickly, but cost/drag dominates if the action is |
| cockpit_metrics.py | algo | sandbox | experiment | `ALGOS/cockpit_metrics.py` | Cockpit honesty / evidence-coverage metrics. |
| counterfactual_effects.py | algo | sandbox | experiment | `ALGOS/counterfactual_effects.py` | Lightweight causal/counterfactual effect estimates. |
| decision_hygiene.py | algo | sandbox | experiment | `ALGOS/decision_hygiene.py` | Decision hygiene scoring algorithms. Reusable deterministic text-feature counts and scoring. Scripts own DB scans/writes. No LLM calls. |
| decreasing_pruning.py | algo | sandbox | experiment | `ALGOS/decreasing_pruning.py` | Decreasing-rate pruning schedule. |
| distributed_leader_election.py | algo | sandbox | experiment | `ALGOS/distributed_leader_election.py` | Local randomized leader election primitive for graph neighborhoods. |
| doomsday_calendar.py | algo | sandbox | experiment | `ALGOS/doomsday_calendar.py` | Doomsday/calendar weekday helper, 0=Sunday..6=Saturday. |
| endpoint_circuit_breaker.py | algo | sandbox | experiment | `ALGOS/endpoint_circuit_breaker.py` | Endpoint circuit-breaker and dual-engine pool selection primitives. |
| fisher_localization.py | algo | sandbox | experiment | `ALGOS/fisher_localization.py` | Fisher-information scoring for off-axis sensing. |
| fold_change_detection.py | algo | sandbox | experiment | `ALGOS/fold_change_detection.py` | Fold-change detection update equations. |
| gini_coefficient.py | algo | sandbox | experiment | `ALGOS/gini_coefficient.py` | Gini inequality coefficient. |
| gliner_zero_shot_extractor.py | algo | sandbox | experiment | `ALGOS/gliner_zero_shot_extractor.py` | Sovereign GLiNER zero-shot extraction instrument for the proof hoard. Purpose: - Accept raw text plus target labels. - Return exact character-offset spans: start, end, text, label, score. - Stay decoupled from production ABSURD/runtime impo |
| hard_truth_math.py | algo | sandbox | experiment | `ALGOS/hard_truth_math.py` | Hard-truth telemetry algorithms for LUCIDOTA. Pure/reusable math: LSM vectors, stylometry features/classifier helpers, ISO/vector parsing. Runtime scripts own DB reads/writes only. No LLM calls. |
| hdc.py | algo | sandbox | experiment | `ALGOS/hdc.py` | Hyperdimensional computing primitives using bipolar vectors. |
| hoeffding_tree.py | algo | sandbox | experiment | `ALGOS/hoeffding_tree.py` | CapyMOA/MOA-style Hoeffding bound helpers for stream splits. |
| honeybee_store.py | algo | sandbox | experiment | `ALGOS/honeybee_store.py` | Common-store feedback primitive for decentralized resource rate control. |
| indy_learning_vector.py | algo | sandbox | experiment | `ALGOS/indy_learning_vector.py` | INDY_READs deterministic book-learning vectors. Pure math/packet layer. Runtime scripts own I/O and receipts. No model calls. |
| infotaxis.py | algo | sandbox | experiment | `ALGOS/infotaxis.py` | Gradient-free entropy search helpers. |
| korpus_text.py | algo | sandbox | experiment | `ALGOS/korpus_text.py` | KORPUS low-level text math helpers: minhash, entropy, CKDOG vector literals. |
| krampus_brainmap.py | algo | sandbox | experiment | `ALGOS/krampus_brainmap.py` | Krampus brain-map projection algorithms. No runtime orchestration here. |
| krampus_chrono.py | algo | sandbox | experiment | `ALGOS/krampus_chrono.py` | KORPUS/KRAMPUS chronological date extraction algorithms. Reusable math/regex layer: scripts should call this; scripts should not own it. No LLM calls. |
| krampus_stickers.py | algo | sandbox | experiment | `ALGOS/krampus_stickers.py` | KORPUS/KRAMPUS cognitive-document sticker algorithms. Reusable deterministic feature math lives here; runtime scripts only orchestrate. These are document/communication telemetry metrics, not diagnosis. No LLM calls. |
| label_foundry.py | algo | sandbox | experiment | `ALGOS/label_foundry.py` | Weak supervision labeling primitives. |
| minhash.py | algo | sandbox | experiment | `ALGOS/minhash.py` | MinHash signatures for approximate Jaccard similarity. |
| minimum_cost_tree.py | algo | sandbox | experiment | `ALGOS/minimum_cost_tree.py` | Minimum-cost tree scoring for length/path trade-offs. |
| nlms.py | algo | sandbox | experiment | `ALGOS/nlms.py` | Normalized least mean squares update. |
| omni_chaotic_sprint.py | algo | sandbox | experiment | `ALGOS/omni_chaotic_sprint.py` | LUCIDOTA Chaotic Omni-Front Synthesis Core Wires: Seismic Ray-Tracer, Fluidic Triage, and Non-Parametric Triad Text Compilation Safeguards: Strict 1536MB DuckDB caps, 7,200 tok/sec Needle throttles, and draft_only gates. |
| perceptual_dedupe.py | algo | sandbox | experiment | `ALGOS/perceptual_dedupe.py` | Perceptual hash-lite dedupe helpers for visual/evidence channels. |
| percyphon.py | algo | sandbox | experiment | `ALGOS/percyphon.py` | Percyphon.ai: zero-VRAM procedural entity generator. |
| pheromone.py | algo | sandbox | experiment | `ALGOS/pheromone.py` | Darwinian surface pheromone worker. Executable lifecycle scaffold: records surface usage/promote/decay signals in lucidota_runtime.surface_pheromone. It never mutates canonical graph/state. |
| physarum_network.py | algo | sandbox | experiment | `ALGOS/physarum_network.py` | Flux-based conductance update primitive. |
| poikilotherm_schoolfield.py | algo | sandbox | experiment | `ALGOS/poikilotherm_schoolfield.py` | Schoolfield-Rollinson poikilotherm rate primitive. Biology origin: temperature-dependent embryo development. LUCIDOTA use: a nonlinear activity/admission curve for systems that should move fastest in a safe operating band and slow down when |
| possum_filter.py | algo | sandbox | experiment | `ALGOS/possum_filter.py` | Possum-style local diversity filter. Suppresses near-duplicate entities that share a category/address signature inside a distance threshold, keeping the first/highest-ranked item from the caller's ordering. |
| privacy.py | algo | sandbox | experiment | `ALGOS/privacy.py` | Privacy/anonymization scoring helpers. |
| rbf_surrogate.py | algo | sandbox | experiment | `ALGOS/rbf_surrogate.py` | Tiny OPOSSUM-style radial-basis surrogate model. |
| regret_engine.py | algo | sandbox | experiment | `ALGOS/regret_engine.py` | Regret-weighted strategy and EV ranking. |
| rete_bandit_gate.py | algo | sandbox | experiment | `ALGOS/rete_bandit_gate.py` | RETE-style deterministic pruning + bandit/regret routing for Bytewax/Absurd. Pure local execution only. No network calls. No canonical graph writes. |
| rsa_cipher.py | algo | sandbox | experiment | `ALGOS/rsa_cipher.py` | Tiny textbook RSA integer primitive for demos/tests only; not production crypto. |
| semantic_neighbors.py | algo | sandbox | experiment | `ALGOS/semantic_neighbors.py` | In-memory semantic-neighbor enclave. |
| serpentina_self_righting.py | algo | sandbox | experiment | `ALGOS/serpentina_self_righting.py` | Chelydra serpentina self-righting morphology primitive. Biology origin: shell shape and body mass influence flip time. LUCIDOTA use: recovery scoring for toppled workflows: long/flat/high-mass states need stronger assistance; compact/spheri |
| shannon_entropy.py | algo | sandbox | experiment | `ALGOS/shannon_entropy.py` | Shannon entropy for observations or probability distributions. |
| shap_attribution.py | algo | sandbox | experiment | `ALGOS/shap_attribution.py` | SHAP/Shapley attribution mathematics and tree-model helpers. Algorithm 2: SHAP (SHapley Additive exPlanations) For feature set F and model value function f_S(x_S), the Shapley value of feature j is \phi_j = \sum_{S \subseteq F \setminus \{j |
| sketches.py | algo | sandbox | experiment | `ALGOS/sketches.py` | Count-min, HLL-lite, and MinHash LSH helpers. |
| sparse_wta.py | algo | sandbox | experiment | `ALGOS/sparse_wta.py` | Sparse winner-take-all tags for high-dimensional similarity. |
| ssim.py | algo | sandbox | experiment | `ALGOS/ssim.py` | Structural similarity index for equal-length grayscale samples. |
| temporal_motifs.py | algo | sandbox | experiment | `ALGOS/temporal_motifs.py` | Temporal session, burst, and motif mining helpers. |
| ternary_lens_audit.py | algo | sandbox | experiment | `ALGOS/ternary_lens_audit.py` | Offline Ternary Lens Lab audit. No external network calls. Uses the local vendor manifest plus local path/reference scan. |
| ternary_lens_router.py | algo | sandbox | experiment | `ALGOS/ternary_lens_router.py` | Tiny ternary Command Envelope Router scaffold for FairyFuse/LUCIDOTA. No external network calls. No hard-coded DB user. Hardware/BitNet execution is STUB until a real FairyFuse/BitNet backend is wired and benchmarked. |
| ternary_router.py | algo | sandbox | experiment | `ALGOS/ternary_router.py` | Always-on FairyFuse ternary router. CPU-bound resident route for LUCIDOTA dual-engine inference. It keeps the FairyFuse backend initialized, memory-maps packed ternary weights when present, and exposes status/route/daemon commands for Bytew |
| thanatosis.py | algo | sandbox | experiment | `ALGOS/thanatosis.py` | Thanatosis / simulated-annealing dormancy primitives. |
| tri_algo_conduit.py | algo | sandbox | experiment | `ALGOS/tri_algo_conduit.py` | Tri-algo conduit: passive monitor -> Hoeffding gate -> self-righting recovery. Pure resource-efficiency primitive for capture/ingress nodes. It keeps heavy work dormant until a signal is statistically worth a burst, then supplies a recovery |
| voronoi_partition.py | algo | sandbox | experiment | `ALGOS/voronoi_partition.py` | Small Voronoi assignment helpers for space partitioning. |
| workshare_allocator.py | algo | sandbox | experiment | `ALGOS/workshare_allocator.py` | Deterministic Project 2501 workshare math. Pure allocator: scripts own receipts/DB/queue writes. No model calls. |
| xgboost_objective.py | algo | sandbox | experiment | `ALGOS/xgboost_objective.py` | XGBoost objective mathematics and a small sklearn-compatible wrapper. Algorithm 1: eXtreme Gradient Boosting (XGBoost) Ensemble prediction after t rounds: \hat y_i^{(t)} = \hat y_i^{(t-1)} + f_t(x_i), f_t \in \mathcal{F} Regularized objecti |

## SCRIPTS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| scripts | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts` | UNKNOWN — needs operator label; filename suggests: scripts |
| benchmark_mark2.py | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts/benchmark_mark2.py` | UNKNOWN — needs operator label; filename suggests: benchmark mark2 |
| diogenes_grpc_smoke.py | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts/diogenes_grpc_smoke.py` | UNKNOWN — needs operator label; filename suggests: diogenes grpc smoke |
| init_files_db.sql | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts/init_files_db.sql` | Files metadata database: claim_kernel_files |
| init_sqlite_state_db.sql | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts/init_sqlite_state_db.sql` | scope_id text primary key, |
| init_state_db.sql | script | sandbox | reusable_prior | `01_REPOS/doggystyle/scripts/init_state_db.sql` | State database: claim_kernel_state |
| scripts | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts` | UNKNOWN — needs operator label; filename suggests: scripts |
| validate-apps.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/apple/validate-apps.sh` | ./scripts/apple/validate-ios.sh |
| validate-ios.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/apple/validate-ios.sh` | validate-ios.sh - Validate iOS Application with embedded llama.xcframework using SwiftUI |
| validate-macos.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/apple/validate-macos.sh` | validate-macos.sh - Validate macOS Application with embedded llama.xcframework using SwiftUI |
| validate-tvos.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/apple/validate-tvos.sh` | validate-tvos.sh - Validate tvOS Application with embedded llama.xcframework using SwiftUI |
| validate-visionos.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/apple/validate-visionos.sh` | validate-visionos.sh - Validate visionOS Application with embedded llama.xcframework using SwiftUI |
| bench-models.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/bench-models.sh` | RESULTS="bench-models-results.txt" |
| build-info.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/build-info.sh` | bin/sh |
| check-requirements.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/check-requirements.sh` | check-requirements.sh checks all requirements files for each top-level |
| compare-commits.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/compare-commits.sh` | if [ $# -lt 2 ]; then |
| compare-llama-bench.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/compare-llama-bench.py` | UNKNOWN — needs operator label; filename suggests: compare llama bench |
| compare-logprobs.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/compare-logprobs.py` | UNKNOWN — needs operator label; filename suggests: compare logprobs |
| create_ops_docs.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/create_ops_docs.py` | This script parses docs/ops/*.csv and creates the ops.md, which is a table documenting supported operations on various ggml backends. |
| debug-test.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/debug-test.sh` | PROG=${0##*/} |
| gen-authors.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/gen-authors.sh` | printf "# date: $(date)\n" > AUTHORS |
| gen-unicode-data.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/gen-unicode-data.py` | UNKNOWN — needs operator label; filename suggests: gen unicode data |
| get-hellaswag.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/get-hellaswag.sh` | bin/sh |
| get-pg.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/get-pg.sh` | function usage { |
| get-wikitext-2.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/get-wikitext-2.sh` | bin/sh |
| get-winogrande.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/get-winogrande.sh` | bin/sh |
| get_chat_template.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/get_chat_template.py` | Fetches the Jinja chat template of a HuggingFace model. If a model has multiple chat templates, you can specify the variant name. Syntax: ./scripts/get_chat_template.py model_id [variant] Examples: ./scripts/get_chat_template.py CohereForAI |
| git-bisect-run.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/git-bisect-run.sh` | cmake_args=() |
| git-bisect.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/git-bisect.sh` | if [ $# -lt 2 ]; then |
| hf.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/hf.sh` | Shortcut for downloading HF models |
| gcn-cdna-vgpr-check.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/hip/gcn-cdna-vgpr-check.py` | UNKNOWN — needs operator label; filename suggests: gcn cdna vgpr check |
| jinja-tester.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/jinja/jinja-tester.py` | UNKNOWN — needs operator label; filename suggests: jinja tester |
| requirements.txt | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/jinja/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| pr2wt.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/pr2wt.sh` | initialize a new worktree from a PR number: |
| serve-static.js | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/serve-static.js` | const http = require('http'); |
| server-bench.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/server-bench.py` | UNKNOWN — needs operator label; filename suggests: server bench |
| server-test-function-call.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/server-test-function-call.py` | Test tool calling capability via chat completions endpoint. Each test case contains: - tools: list of tool definitions (OpenAI-compatible) - messages: initial conversation messages - mock_tool_responses: dict mapping tool_name -> callable(a |
| server-test-model.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/server-test-model.py` | UNKNOWN — needs operator label; filename suggests: server test model |
| server-test-parallel-tc.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/server-test-parallel-tc.py` | Test parallel tool-calling capability via chat completions endpoint. Only run this against models that actually support parallel tool calls — this script does not attempt to toggle that setting on the server. Each scenario is explicitly wor |
| server-test-structured.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/server-test-structured.py` | Test structured output capability via chat completions endpoint. Each test case contains: - response_format: OpenAI-compatible response_format specification. Both "json_schema" and "json_object" are accepted; with "json_object" a schema can |
| run-bench.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/adb/run-bench.sh` | bin/sh |
| run-cli.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/adb/run-cli.sh` | bin/sh |
| run-completion.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/adb/run-completion.sh` | bin/sh |
| run-mtmd.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/adb/run-mtmd.sh` | bin/sh |
| run-tool.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/adb/run-tool.sh` | bin/sh |
| ggml-hexagon-profile.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py` | UNKNOWN — needs operator label; filename suggests: ggml hexagon profile |
| requirements.txt | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| run_qdc_jobs.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/run_qdc_jobs.py` | Run llama.cpp Hexagon Android tests in a single QDC Appium job. Bundles test scripts into one artifact and submits a single QDC job: 1. run_bench_tests_posix.py — llama-cli and llama-bench on CPU / GPU / NPU (from scripts/snapdragon/qdc/) R |
| conftest.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/tests/conftest.py` | Shared pytest fixtures for QDC on-device test runners. |
| run_backend_ops_posix.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/tests/run_backend_ops_posix.py` | On-device test-backend-ops runner for llama.cpp (HTP0 backend). Executed by QDC's Appium test framework on the QDC runner. The runner has ADB access to the allocated device. |
| run_bench_tests_posix.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/tests/run_bench_tests_posix.py` | On-device bench and completion test runner for llama.cpp (CPU, GPU, NPU backends). Executed by QDC's Appium test framework on the QDC runner. The runner has ADB access to the allocated device. Placeholders replaced at artifact creation time |
| utils.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/tests/utils.py` | Shared helpers for QDC on-device test runners. |
| sync-ggml-am.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/sync-ggml-am.sh` | Synchronize ggml changes to llama.cpp |
| sync-ggml.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/sync-ggml.sh` | cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt |
| sync_vendor.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/sync_vendor.py` | UNKNOWN — needs operator label; filename suggests: sync vendor |
| tool_bench.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/tool_bench.py` | Simplistic tool call benchmarks for llama-server and ollama. Essentially runs the tests at server/tools/server/tests/unit/test_tool_call.py N times, at different temperatures and on different backends (current llama-server, baseline llama-s |
| tool_bench.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/tool_bench.sh` | cmake --build build -j |
| verify-checksum-models.py | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/verify-checksum-models.py` | UNKNOWN — needs operator label; filename suggests: verify checksum models |
| wc2wt.sh | script | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/wc2wt.sh` | initialize a new worktree from a branch name: |
| 001_lucidota_control.sql | script | sandbox | reference | `06_SCHEMA/001_lucidota_control.sql` | LUCIDOTA control-plane schema. |
| 002_model_runtime.sql | script | sandbox | reference | `06_SCHEMA/002_model_runtime.sql` | Model runtime registry for LUCIDOTA. |
| 004_learning_reflex.sql | script | sandbox | reference | `06_SCHEMA/004_learning_reflex.sql` | LUCIDOTA learning/reflex schema. |
| 005_cas_manifest.sql | script | sandbox | reference | `06_SCHEMA/005_cas_manifest.sql` | LUCIDOTA local CAS manifest/index. |
| 006_workflow_registry.sql | script | sandbox | reference | `06_SCHEMA/006_workflow_registry.sql` | LUCIDOTA workflow registry: small inspectable catalog for ABSURD-owned workflows. |
| 007_bytewax_stream.sql | script | sandbox | reference | `06_SCHEMA/007_bytewax_stream.sql` | Bytewax mini-stream proof: live dataflow outputs durable bounded hints. |
| 008_hop_pivot.sql | script | sandbox | reference | `06_SCHEMA/008_hop_pivot.sql` | Queue-backed Hop Pivot v1. |
| 009_treelite_router.sql | script | sandbox | reference | `06_SCHEMA/009_treelite_router.sql` | Treelite routing hints: compiled/lightweight model outputs stay advisory. |
| 010_wake_bus.sql | script | sandbox | reference | `06_SCHEMA/010_wake_bus.sql` | Local-first Wake Bus. |
| 012_authorized_extractors.sql | script | sandbox | reference | `06_SCHEMA/012_authorized_extractors.sql` | Authorized extractor registry. |
| 013_signal_ingress.sql | script | sandbox | reference | `06_SCHEMA/013_signal_ingress.sql` | Tri-algo signal ingress decisions: passive monitor -> statistical gate -> recovery. |
| 014_indy_runtime.sql | script | sandbox | reference | `06_SCHEMA/014_indy_runtime.sql` | Indy_Reads runtime support tables. |
| 016_go_graph_core.sql | script | sandbox | reference | `06_SCHEMA/016_go_graph_core.sql` | GO graph core schema. |
| 017_indy_reads_library.sql | script | sandbox | reference | `06_SCHEMA/017_indy_reads_library.sql` | Indy_READs library ingestion, chunking, embedding, and LoRA staging. |
| 018_investigation_artifact.sql | script | sandbox | reference | `06_SCHEMA/018_investigation_artifact.sql` | DIOGENES investigative case/artifact workflow. |
| 019_korpus_krampii.sql | script | sandbox | reference | `06_SCHEMA/019_korpus_krampii.sql` | KORPUS KRAMPII: deterministic mass-ingestion substrate. |
| 020_chat_dump_timeline.sql | script | sandbox | reference | `06_SCHEMA/020_chat_dump_timeline.sql` | LUCIDOTA chat dump timeline: deterministic OpenAI/Claude export normalization. |
| 020_korpus_derived_compute_queue.sql | script | sandbox | reference | `06_SCHEMA/020_korpus_derived_compute_queue.sql` | KORPUS derived compute queue. |
| 021_hard_truth_math.sql | script | sandbox | reference | `06_SCHEMA/021_hard_truth_math.sql` | LUCIDOTA hard-math telemetry: LSM, state transitions, stylometry, semantic isolation. |
| 022_comm_dump_timeline.sql | script | sandbox | reference | `06_SCHEMA/022_comm_dump_timeline.sql` | LUCIDOTA universal communications dump timeline. |
| 023_etl_pipeline.sql | script | sandbox | reference | `06_SCHEMA/023_etl_pipeline.sql` | LUCIDOTA ETL Pipeline ABSURD substrate. |
| 025_chrono_ledger_core.sql | script | sandbox | reference | `06_SCHEMA/025_chrono_ledger_core.sql` | FILE: 06_SCHEMA/025_chrono_ledger_core.sql |
| 026_chrono_absurd_triggers.sql | script | sandbox | reference | `06_SCHEMA/026_chrono_absurd_triggers.sql` | FILE: 06_SCHEMA/026_chrono_absurd_triggers.sql |
| 027_chrono_phase_c_ops.sql | script | sandbox | reference | `06_SCHEMA/027_chrono_phase_c_ops.sql` | FILE: 06_SCHEMA/027_chrono_phase_c_ops.sql |
| 028_ternary_lens_lab.sql | script | sandbox | reference | `06_SCHEMA/028_ternary_lens_lab.sql` | FILE: 06_SCHEMA/028_ternary_lens_lab.sql |
| 029_darwinian_surfaces.sql | script | sandbox | reference | `06_SCHEMA/029_darwinian_surfaces.sql` | FILE: 06_SCHEMA/029_darwinian_surfaces.sql |
| 030_phase05_brain_archaeology.sql | script | sandbox | reference | `06_SCHEMA/030_phase05_brain_archaeology.sql` | FILE: 06_SCHEMA/030_phase05_brain_archaeology.sql |
| 031_graph_write_barrier_plan.sql | script | sandbox | reference | `06_SCHEMA/031_graph_write_barrier_plan.sql` | FILE: 06_SCHEMA/031_graph_write_barrier_plan.sql |
| 034_graph_promotion_pipeline.sql | script | sandbox | reference | `06_SCHEMA/034_graph_promotion_pipeline.sql` | FILE: 06_SCHEMA/034_graph_promotion_pipeline.sql |
| 035_absurd_queue_spine.sql | script | sandbox | reference | `06_SCHEMA/035_absurd_queue_spine.sql` | FILE: 06_SCHEMA/035_absurd_queue_spine.sql |
| 036_absurd_chrono_wrapper.sql | script | sandbox | reference | `06_SCHEMA/036_absurd_chrono_wrapper.sql` | FILE: 06_SCHEMA/036_absurd_chrono_wrapper.sql |
| 037_absurd_krampus_wrapper.sql | script | sandbox | reference | `06_SCHEMA/037_absurd_krampus_wrapper.sql` | FILE: 06_SCHEMA/037_absurd_krampus_wrapper.sql |
| 038_absurd_river_wrapper.sql | script | sandbox | reference | `06_SCHEMA/038_absurd_river_wrapper.sql` | FILE: 06_SCHEMA/038_absurd_river_wrapper.sql |
| 039_absurd_real_work_loop.sql | script | sandbox | reference | `06_SCHEMA/039_absurd_real_work_loop.sql` | FILE: 06_SCHEMA/039_absurd_real_work_loop.sql |
| 040_graph_write_barrier_enforcement.sql | script | sandbox | reference | `06_SCHEMA/040_graph_write_barrier_enforcement.sql` | FILE: 06_SCHEMA/040_graph_write_barrier_enforcement.sql |
| 041_boring_beast_loop_contracts.sql | script | sandbox | reference | `06_SCHEMA/041_boring_beast_loop_contracts.sql` | FILE: 06_SCHEMA/041_boring_beast_loop_contracts.sql |
| 042_fairyfuse_v0_backend.sql | script | sandbox | reference | `06_SCHEMA/042_fairyfuse_v0_backend.sql` | FILE: 06_SCHEMA/042_fairyfuse_v0_backend.sql |
| 043_absurd_remaining_worker_contracts.sql | script | sandbox | reference | `06_SCHEMA/043_absurd_remaining_worker_contracts.sql` | FILE: 06_SCHEMA/043_absurd_remaining_worker_contracts.sql |
| 044_graph_promotion_policy_roles.sql | script | sandbox | reference | `06_SCHEMA/044_graph_promotion_policy_roles.sql` | FILE: 06_SCHEMA/044_graph_promotion_policy_roles.sql |
| 045_document_ingestion_pipeline.sql | script | sandbox | reference | `06_SCHEMA/045_document_ingestion_pipeline.sql` | FILE: 06_SCHEMA/045_document_ingestion_pipeline.sql |
| 046_catchme_sensitivity_map.sql | script | sandbox | reference | `06_SCHEMA/046_catchme_sensitivity_map.sql` | FILE: 06_SCHEMA/046_catchme_sensitivity_map.sql |
| 047_simplemem_eval_promotion_bridge.sql | script | sandbox | reference | `06_SCHEMA/047_simplemem_eval_promotion_bridge.sql` | FILE: 06_SCHEMA/047_simplemem_eval_promotion_bridge.sql |
| 048_phase05_allowlisted_ingest.sql | script | sandbox | reference | `06_SCHEMA/048_phase05_allowlisted_ingest.sql` | FILE: 06_SCHEMA/048_phase05_allowlisted_ingest.sql |
| 049_absurd_intake_wrapper.sql | script | sandbox | reference | `06_SCHEMA/049_absurd_intake_wrapper.sql` | FILE: 06_SCHEMA/049_absurd_intake_wrapper.sql |
| 050_document_claim_packet_bridge.sql | script | sandbox | reference | `06_SCHEMA/050_document_claim_packet_bridge.sql` | FILE: 06_SCHEMA/050_document_claim_packet_bridge.sql |
| 051_phase05_design_atom_runtime.sql | script | sandbox | reference | `06_SCHEMA/051_phase05_design_atom_runtime.sql` | FILE: 06_SCHEMA/051_phase05_design_atom_runtime.sql |
| 052_graph_promotion_materialization.sql | script | sandbox | reference | `06_SCHEMA/052_graph_promotion_materialization.sql` | FILE: 06_SCHEMA/052_graph_promotion_materialization.sql |
| 053_simplemem_candidate_index.sql | script | sandbox | reference | `06_SCHEMA/053_simplemem_candidate_index.sql` | FILE: 06_SCHEMA/053_simplemem_candidate_index.sql |
| 054_demem_runtime_enforcement.sql | script | sandbox | reference | `06_SCHEMA/054_demem_runtime_enforcement.sql` | FILE: 06_SCHEMA/054_demem_runtime_enforcement.sql |
| 055_catchme_context_guard.sql | script | sandbox | reference | `06_SCHEMA/055_catchme_context_guard.sql` | FILE: 06_SCHEMA/055_catchme_context_guard.sql |
| 056_audit_verdict_runtime_enforcement.sql | script | sandbox | reference | `06_SCHEMA/056_audit_verdict_runtime_enforcement.sql` | FILE: 06_SCHEMA/056_audit_verdict_runtime_enforcement.sql |
| 057_tracer_claim_packet_bridge.sql | script | sandbox | reference | `06_SCHEMA/057_tracer_claim_packet_bridge.sql` | FILE: 06_SCHEMA/057_tracer_claim_packet_bridge.sql |
| 058_chrono_queue_event_bridge.sql | script | sandbox | reference | `06_SCHEMA/058_chrono_queue_event_bridge.sql` | FILE: 06_SCHEMA/058_chrono_queue_event_bridge.sql |
| 059_graph_promotion_gate_runtime.sql | script | sandbox | reference | `06_SCHEMA/059_graph_promotion_gate_runtime.sql` | FILE: 06_SCHEMA/059_graph_promotion_gate_runtime.sql |
| 060_idempotency_duplicate_suppression.sql | script | sandbox | reference | `06_SCHEMA/060_idempotency_duplicate_suppression.sql` | FILE: 06_SCHEMA/060_idempotency_duplicate_suppression.sql |
| 061_absurd_queue_hardening_v2.sql | script | sandbox | reference | `06_SCHEMA/061_absurd_queue_hardening_v2.sql` | FILE: 06_SCHEMA/061_absurd_queue_hardening_v2.sql |
| 062_work_order_loader_validator.sql | script | sandbox | reference | `06_SCHEMA/062_work_order_loader_validator.sql` | FILE: 06_SCHEMA/062_work_order_loader_validator.sql |
| 064_queue_transition_law_v2.sql | script | sandbox | reference | `06_SCHEMA/064_queue_transition_law_v2.sql` | FILE: 06_SCHEMA/064_queue_transition_law_v2.sql |
| 065_graph_materialization_helper_v2.sql | script | sandbox | reference | `06_SCHEMA/065_graph_materialization_helper_v2.sql` | FILE: 06_SCHEMA/065_graph_materialization_helper_v2.sql |
| 066_phase05_workflow_blueprint_synthesis.sql | script | sandbox | reference | `06_SCHEMA/066_phase05_workflow_blueprint_synthesis.sql` | FILE: 06_SCHEMA/066_phase05_workflow_blueprint_synthesis.sql |
| 067_chrono_queue_event_bridge_automation.sql | script | sandbox | reference | `06_SCHEMA/067_chrono_queue_event_bridge_automation.sql` | FILE: 06_SCHEMA/067_chrono_queue_event_bridge_automation.sql |
| 068_conversation_command_acceptance.sql | script | sandbox | reference | `06_SCHEMA/068_conversation_command_acceptance.sql` | FILE: 06_SCHEMA/068_conversation_command_acceptance.sql |
| 069_graph_promotion_approval_state_machine.sql | script | sandbox | reference | `06_SCHEMA/069_graph_promotion_approval_state_machine.sql` | FILE: 06_SCHEMA/069_graph_promotion_approval_state_machine.sql |
| 070_master_eye_runtime_review.sql | script | sandbox | reference | `06_SCHEMA/070_master_eye_runtime_review.sql` | FILE: 06_SCHEMA/070_master_eye_runtime_review.sql |
| 071_chrono_conservation_verifier.sql | script | sandbox | reference | `06_SCHEMA/071_chrono_conservation_verifier.sql` | FILE: 06_SCHEMA/071_chrono_conservation_verifier.sql |
| 072_marrow_cep_bridge.sql | script | sandbox | reference | `06_SCHEMA/072_marrow_cep_bridge.sql` | FILE: 06_SCHEMA/072_marrow_cep_bridge.sql |
| 073_absurd_river_claim_packet_job.sql | script | sandbox | reference | `06_SCHEMA/073_absurd_river_claim_packet_job.sql` | FILE: 06_SCHEMA/073_absurd_river_claim_packet_job.sql |
| 074_graph_journal_write_barrier.sql | script | sandbox | reference | `06_SCHEMA/074_graph_journal_write_barrier.sql` | FILE: 06_SCHEMA/074_graph_journal_write_barrier.sql |
| 075_phase05_workflow_blueprint_queue.sql | script | sandbox | reference | `06_SCHEMA/075_phase05_workflow_blueprint_queue.sql` | FILE: 06_SCHEMA/075_phase05_workflow_blueprint_queue.sql |
| 076_absurd_document_parse_worker.sql | script | sandbox | reference | `06_SCHEMA/076_absurd_document_parse_worker.sql` | FILE: 06_SCHEMA/076_absurd_document_parse_worker.sql |
| 077_graph_promotion_packet_dedupe.sql | script | sandbox | reference | `06_SCHEMA/077_graph_promotion_packet_dedupe.sql` | FILE: 06_SCHEMA/077_graph_promotion_packet_dedupe.sql |
| 078_operator_ontology_fidelity_runtime.sql | script | sandbox | reference | `06_SCHEMA/078_operator_ontology_fidelity_runtime.sql` | FILE: 06_SCHEMA/078_operator_ontology_fidelity_runtime.sql |
| 079_intake_custody_job_bridge.sql | script | sandbox | reference | `06_SCHEMA/079_intake_custody_job_bridge.sql` | FILE: 06_SCHEMA/079_intake_custody_job_bridge.sql |
| 080_sticker_feature_vector_extractor_v1.sql | script | sandbox | reference | `06_SCHEMA/080_sticker_feature_vector_extractor_v1.sql` | FILE: 06_SCHEMA/080_sticker_feature_vector_extractor_v1.sql |
| 081_cep_conversation_command_dedupe.sql | script | sandbox | reference | `06_SCHEMA/081_cep_conversation_command_dedupe.sql` | FILE: 06_SCHEMA/081_cep_conversation_command_dedupe.sql |
| 082_absurd_worker_contract_registry_enforcement.sql | script | sandbox | reference | `06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql` | FILE: 06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql |
| 083_topology_finding_extractor.sql | script | sandbox | reference | `06_SCHEMA/083_topology_finding_extractor.sql` | FILE: 06_SCHEMA/083_topology_finding_extractor.sql |
| 084_absurd_dead_letter_review.sql | script | sandbox | reference | `06_SCHEMA/084_absurd_dead_letter_review.sql` | FILE: 06_SCHEMA/084_absurd_dead_letter_review.sql |
| 085_graph_promotion_evidence_resolver.sql | script | sandbox | reference | `06_SCHEMA/085_graph_promotion_evidence_resolver.sql` | FILE: 06_SCHEMA/085_graph_promotion_evidence_resolver.sql |
| 086_telemetry_finding_worker.sql | script | sandbox | reference | `06_SCHEMA/086_telemetry_finding_worker.sql` | FILE: 06_SCHEMA/086_telemetry_finding_worker.sql |
| 087_chrono_ranking_pass.sql | script | sandbox | reference | `06_SCHEMA/087_chrono_ranking_pass.sql` | FILE: 06_SCHEMA/087_chrono_ranking_pass.sql |
| 088_cruelty_protocols_output.sql | script | sandbox | reference | `06_SCHEMA/088_cruelty_protocols_output.sql` | FILE: 06_SCHEMA/088_cruelty_protocols_output.sql |
| 089_marrow_absurd_work_order_bridge.sql | script | sandbox | reference | `06_SCHEMA/089_marrow_absurd_work_order_bridge.sql` | FILE: 06_SCHEMA/089_marrow_absurd_work_order_bridge.sql |
| 090_authority_class_mapper.sql | script | sandbox | reference | `06_SCHEMA/090_authority_class_mapper.sql` | FILE: 06_SCHEMA/090_authority_class_mapper.sql |
| 091_chrono_temporal_claim_invalidation.sql | script | sandbox | reference | `06_SCHEMA/091_chrono_temporal_claim_invalidation.sql` | FILE: 06_SCHEMA/091_chrono_temporal_claim_invalidation.sql |
| 092_surface_promotion_lineage_guard.sql | script | sandbox | reference | `06_SCHEMA/092_surface_promotion_lineage_guard.sql` | FILE: 06_SCHEMA/092_surface_promotion_lineage_guard.sql |
| 093_graph_edge_materialization_guard.sql | script | sandbox | reference | `06_SCHEMA/093_graph_edge_materialization_guard.sql` | FILE: 06_SCHEMA/093_graph_edge_materialization_guard.sql |
| 094_workflow_foundry_runtime.sql | script | sandbox | reference | `06_SCHEMA/094_workflow_foundry_runtime.sql` | FILE: 06_SCHEMA/094_workflow_foundry_runtime.sql |
| 095_cep_graph_packet_staging.sql | script | sandbox | reference | `06_SCHEMA/095_cep_graph_packet_staging.sql` | FILE: 06_SCHEMA/095_cep_graph_packet_staging.sql |
| 096_worker_command_registry.sql | script | sandbox | reference | `06_SCHEMA/096_worker_command_registry.sql` | FILE: 06_SCHEMA/096_worker_command_registry.sql |
| 097_conversation_command_status_transition.sql | script | sandbox | reference | `06_SCHEMA/097_conversation_command_status_transition.sql` | FILE: 06_SCHEMA/097_conversation_command_status_transition.sql |
| 098_absurd_retry_policy_registry.sql | script | sandbox | reference | `06_SCHEMA/098_absurd_retry_policy_registry.sql` | FILE: 06_SCHEMA/098_absurd_retry_policy_registry.sql |
| 099_phase05_contradiction_ledger.sql | script | sandbox | reference | `06_SCHEMA/099_phase05_contradiction_ledger.sql` | LUCIDOTA Round 73: Phase 0.5 contradiction ledger worker. |
| 100_chrono_timeline_projection_refresh.sql | script | sandbox | reference | `06_SCHEMA/100_chrono_timeline_projection_refresh.sql` | LUCIDOTA Round 74: current Chrono projection refresh target. |
| 101_semantic_handle_registry.sql | script | sandbox | reference | `06_SCHEMA/101_semantic_handle_registry.sql` | LUCIDOTA Round 78: fungible semantic handle registry. |
| 102_chrono_temporal_conflict_detector.sql | script | sandbox | reference | `06_SCHEMA/102_chrono_temporal_conflict_detector.sql` | LUCIDOTA Round 79: temporal conflict detector. Additive findings only. |
| 103_longmemeval_seed_generator.sql | script | sandbox | reference | `06_SCHEMA/103_longmemeval_seed_generator.sql` | LUCIDOTA Round 83: LongMemEval style seed table. |
| 104_post_gate_target_selection.sql | script | sandbox | reference | `06_SCHEMA/104_post_gate_target_selection.sql` | LUCIDOTA post Mega-Gate target selection framework. |
| 105_recovery_receipt.sql | script | sandbox | reference | `06_SCHEMA/105_recovery_receipt.sql` | receipt_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(), |
| 106_production_readiness.sql | script | sandbox | reference | `06_SCHEMA/106_production_readiness.sql` | readiness_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(), |
| 107_system_telemetry_rollup.sql | script | sandbox | reference | `06_SCHEMA/107_system_telemetry_rollup.sql` | rollup_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(), |
| 108_bytewax_mtime_lora_policy.sql | script | sandbox | reference | `06_SCHEMA/108_bytewax_mtime_lora_policy.sql` | Bytewax temporal-fragility runtime guard. |
| 109_chrono_lane_split_projection_gate.sql | script | sandbox | reference | `06_SCHEMA/109_chrono_lane_split_projection_gate.sql` | RFC-CHRONO-001: Temporal Claim Ledger -> Lane-Split Projection -> Graph Promotion Gate |
| 110_omni_front_chrono_graph_materialization_lock.sql | script | sandbox | reference | `06_SCHEMA/110_omni_front_chrono_graph_materialization_lock.sql` | LUCIDOTA omni-front graph materialization lock. |
| absurd_job.schema.json | script | sandbox | reference | `06_SCHEMA/absurd_job.schema.json` | UNKNOWN — needs operator label; filename suggests: absurd job.schema |
| bytewax_signal_packet.schema.json | script | sandbox | reference | `06_SCHEMA/bytewax_signal_packet.schema.json` | UNKNOWN — needs operator label; filename suggests: bytewax signal packet.schema |
| chrono_claim.schema.json | script | sandbox | reference | `06_SCHEMA/chrono_claim.schema.json` | UNKNOWN — needs operator label; filename suggests: chrono claim.schema |
| contradiction_record.schema.json | script | sandbox | reference | `06_SCHEMA/contradiction_record.schema.json` | UNKNOWN — needs operator label; filename suggests: contradiction record.schema |
| conversation_command.schema.json | script | sandbox | reference | `06_SCHEMA/conversation_command.schema.json` | UNKNOWN — needs operator label; filename suggests: conversation command.schema |
| dev_order_matrix_policy.v1.json | script | sandbox | reference | `06_SCHEMA/dev_order_matrix_policy.v1.json` | UNKNOWN — needs operator label; filename suggests: dev order matrix policy.v1 |
| document_parse_result.schema.json | script | sandbox | reference | `06_SCHEMA/document_parse_result.schema.json` | UNKNOWN — needs operator label; filename suggests: document parse result.schema |
| entity_candidate_registry.schema.json | script | sandbox | reference | `06_SCHEMA/entity_candidate_registry.schema.json` | UNKNOWN — needs operator label; filename suggests: entity candidate registry.schema |
| event_candidate_registry.schema.json | script | sandbox | reference | `06_SCHEMA/event_candidate_registry.schema.json` | UNKNOWN — needs operator label; filename suggests: event candidate registry.schema |
| graph_promotion_packet.schema.json | script | sandbox | reference | `06_SCHEMA/graph_promotion_packet.schema.json` | UNKNOWN — needs operator label; filename suggests: graph promotion packet.schema |
| import_policy.schema.json | script | sandbox | reference | `06_SCHEMA/import_policy.schema.json` | UNKNOWN — needs operator label; filename suggests: import policy.schema |
| krampus_custody.schema.json | script | sandbox | reference | `06_SCHEMA/krampus_custody.schema.json` | UNKNOWN — needs operator label; filename suggests: krampus custody.schema |
| memory_candidate.schema.json | script | sandbox | reference | `06_SCHEMA/memory_candidate.schema.json` | UNKNOWN — needs operator label; filename suggests: memory candidate.schema |
| model_invocation_receipt.schema.json | script | sandbox | reference | `06_SCHEMA/model_invocation_receipt.schema.json` | UNKNOWN — needs operator label; filename suggests: model invocation receipt.schema |
| ontology_staging.schema.json | script | sandbox | reference | `06_SCHEMA/ontology_staging.schema.json` | UNKNOWN — needs operator label; filename suggests: ontology staging.schema |
| operator_decision_receipt.schema.json | script | sandbox | reference | `06_SCHEMA/operator_decision_receipt.schema.json` | UNKNOWN — needs operator label; filename suggests: operator decision receipt.schema |
| pipeline_stage.schema.json | script | sandbox | reference | `06_SCHEMA/pipeline_stage.schema.json` | UNKNOWN — needs operator label; filename suggests: pipeline stage.schema |
| proof_object.schema.json | script | sandbox | reference | `06_SCHEMA/proof_object.schema.json` | UNKNOWN — needs operator label; filename suggests: proof object.schema |
| source_bundle.schema.json | script | sandbox | reference | `06_SCHEMA/source_bundle.schema.json` | UNKNOWN — needs operator label; filename suggests: source bundle.schema |
| status_ledger_entry.schema.json | script | sandbox | reference | `06_SCHEMA/status_ledger_entry.schema.json` | UNKNOWN — needs operator label; filename suggests: status ledger entry.schema |
| work_order.schema.json | script | sandbox | reference | `06_SCHEMA/work_order.schema.json` | UNKNOWN — needs operator label; filename suggests: work order.schema |
| scripts | script | sandbox | reusable_prior | `scripts` | UNKNOWN — needs operator label; filename suggests: scripts |
| dbos_fault_injector_20260527T034022927312Z.json | script | sandbox | reusable_prior | `scripts/05_OUTPUTS/chaos/dbos_fault_injector_20260527T034022927312Z.json` | UNKNOWN — needs operator label; filename suggests: dbos fault injector 20260527T034022927312Z |
| dbos_river_wrapper_worker-once_20260526T205102405690Z.json | script | sandbox | reusable_prior | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205102405690Z.json` | UNKNOWN — needs operator label; filename suggests: dbos river wrapper worker once 20260526T205102405690Z |
| dbos_river_wrapper_worker-once_20260526T205321153628Z.json | script | sandbox | reusable_prior | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205321153628Z.json` | UNKNOWN — needs operator label; filename suggests: dbos river wrapper worker once 20260526T205321153628Z |
| dbos_river_wrapper_worker-once_20260526T205401745296Z.json | script | sandbox | reusable_prior | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205401745296Z.json` | UNKNOWN — needs operator label; filename suggests: dbos river wrapper worker once 20260526T205401745296Z |
| CORPSE_MANIFEST.jsonl | script | sandbox | reusable_prior | `scripts/CORPSE_MANIFEST.jsonl` | UNKNOWN — needs operator label; filename suggests: CORPSE MANIFEST |
| KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl | script | sandbox | reusable_prior | `scripts/KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl` | UNKNOWN — needs operator label; filename suggests: KRAMPUSCHEWING SCRIPT CORPSES |
| SCRIPT_AUDIT_MANIFEST.jsonl | script | sandbox | reusable_prior | `scripts/SCRIPT_AUDIT_MANIFEST.jsonl` | UNKNOWN — needs operator label; filename suggests: SCRIPT AUDIT MANIFEST |
| abductive_db_os_gate.py | script | sandbox | reusable_prior | `scripts/abductive_db_os_gate.py` | Abductive DB OS megagate. |
| abductive_db_os_health_check.py | script | sandbox | reusable_prior | `scripts/abductive_db_os_health_check.py` | Fast health check for the file-backed Abductive DB OS lane. |
| abductive_db_os_ledger.py | script | sandbox | reusable_prior | `scripts/abductive_db_os_ledger.py` | File-backed abductive DB OS operating ledger. |
| abductive_next_move_engine.py | script | sandbox | reusable_prior | `scripts/abductive_next_move_engine.py` | Rank next safe moves from abductive DB OS board state. |
| absurd_abductive_ledger.py | script | sandbox | reusable_prior | `scripts/absurd_abductive_ledger.py` | File-backed ABSURD abductive operating ledger. |
| absurd_chrono_worker.py | script | sandbox | reusable_prior | `scripts/absurd_chrono_worker.py` | ABSURD queue-spine wrapper for Chrono-Ledger. The wrapper observes Chrono service health and writes ABSURD queue/job/workflow receipts. It never deletes, overwrites, invalidates, or inserts temporal claims. |
| absurd_consume_one.py | script | sandbox | reusable_prior | `scripts/absurd_consume_one.py` | Consume one ABSURD queue job with kernel-authorization enforcement. This is the ABSURD/Postgres successor to the legacy DBOS consume-one worker. It claims exactly one queued row, validates kernel authorization before handler execution, writ |
| absurd_corpus_job_bridge.py | script | sandbox | reusable_prior | `scripts/absurd_corpus_job_bridge.py` | ABSURD-compatible corpus lane job bridge for KORPUS/KRAMPUS custody work. |
| absurd_gate.py | script | sandbox | reusable_prior | `scripts/absurd_gate.py` | ABSURD abductive megagate. |
| absurd_graph_promotion_worker.py | script | sandbox | reusable_prior | `scripts/absurd_graph_promotion_worker.py` | ABSURD graph-promotion worker. Consumes only registry-approved graph_promotion jobs. Packet staging is routed through scripts/graph_promotion_gate.py so this worker cannot bypass evidence, authority, or preflight checks. Canonical graph mat |
| absurd_health_check.py | script | sandbox | reusable_prior | `scripts/absurd_health_check.py` | UNKNOWN — needs operator label; filename suggests: absurd health check |
| absurd_intake_worker.py | script | sandbox | reusable_prior | `scripts/absurd_intake_worker.py` | ABSURD custody wrapper for the supervised Rust lucidota-intake service. Observes service/drop-dir state and writes ABSURD workflow/custody receipts. It does not move drops. |
| absurd_kernel_authorization.py | script | sandbox | reusable_prior | `scripts/absurd_kernel_authorization.py` | Compatibility wrapper for scripts/spine_kernel_authorization.py. |
| absurd_momentary_flow.py | script | sandbox | reusable_prior | `scripts/absurd_momentary_flow.py` | PocketFlow-style momentary graph runner for durable ABSURD jobs. Shared state exists only during the job. Completion returns training evidence, transition trace, and hashes; the shared store itself is deliberately discarded. |
| absurd_next_move_engine.py | script | sandbox | reusable_prior | `scripts/absurd_next_move_engine.py` | Rank next safe moves from ABSURD abductive board state. |
| absurd_queue_spine.py | script | sandbox | reusable_prior | `scripts/absurd_queue_spine.py` | LUCIDOTA ABSURD-compatible durable queue spine. Dry-run is default for actions that can write. Execute mode writes only queue-spine state (jobs/events/dead-letter/workflow_event), never canonical graph tables. |
| absurd_remaining_worker_contract_alignment_check.py | script | sandbox | reusable_prior | `scripts/absurd_remaining_worker_contract_alignment_check.py` | Static contract-alignment check for remaining ABSURD workers. This covers the current open blocker scope: Chrono, Surface/CEP, document_parse, and KORPUS. It proves source-visible contract validation/rejection hooks and SQL registry job-kin |
| absurd_river_worker.py | script | sandbox | reusable_prior | `scripts/absurd_river_worker.py` | ABSURD queue-spine wrapper for River/Bytewax and Phase 0.5 GLiNER extraction. Primary queue for this pass: absurd.phase05_streaming_brain / gliner_zero_shot_extract Safety laws: - Writes ABSURD job/event/workflow receipts and lucidota_learn |
| absurd_worker_contracts.py | script | sandbox | reusable_prior | `scripts/absurd_worker_contracts.py` | ABSURD worker contract validation helpers. A queued job is executable only when a live contract row names its queue and job kind. This is intentionally small and boring so worker code can call it before side effects. |
| absurd_worker_runner.py | script | sandbox | reusable_prior | `scripts/absurd_worker_runner.py` | Drain local file-backed ABSURD pipeline jobs with legal state transitions. |
| activity_tree_ingest_dry_run.py | script | sandbox | reusable_prior | `scripts/activity_tree_ingest_dry_run.py` | activity_event/activity_tree_node shaped JSON; consent/revocation/sensitivity scopes required Research-only dry-run scaffold. No DB writes. No graph writes. |
| adhd_slow_lane_divergence.py | script | sandbox | reusable_prior | `scripts/adhd_slow_lane_divergence.py` | ADHD slow lane: divergent frames, local lanes, Treelite pruning. |
| apply_lucidota_control_schema.sh | script | sandbox | reusable_prior | `scripts/apply_lucidota_control_schema.sh` | DB_URL="${LUCIDOTA_CONTROL_DATABASE_URL:-postgresql://mfspx@/lucidota_state}" |
| authority_class_mapper.py | script | sandbox | reusable_prior | `scripts/authority_class_mapper.py` | Map and enforce authority_class on extraction outputs. Hardening role: every model/extractor output must carry an explicit authority class before it can move toward command, graph, or evidence workflows. This tool is intentionally bounded:  |
| behavioral_memory_dry_run.py | script | sandbox | reusable_prior | `scripts/behavioral_memory_dry_run.py` | Mem0-style behavior observation, not blind reinforcement Research-only dry-run scaffold. No DB writes. No graph writes. |
| bitloops_airlock_audit.py | script | sandbox | reusable_prior | `scripts/bitloops_airlock_audit.py` | Sovereign airlock audit for Bitloops before any LUCIDOTA install/enable. |
| bitloops_automation_loop.py | script | sandbox | reusable_prior | `scripts/bitloops_automation_loop.py` | Receipt-gated Bitloops automation loop for ABSURD momentary flows. This is the thin orchestration layer: verified cases run through the Bitloops -> Bytewax -> River -> ternary momentary flow; unverified cases are preserved for quarantine/re |
| bitloops_full_reingest_manifest.py | script | sandbox | reusable_prior | `scripts/bitloops_full_reingest_manifest.py` | Full-world reingestion manifest for LUCI/KRAMPUS/PONY/KORPUS. Indexes files into Bitloops/River-compatible JSONL rows. It never deletes, purges, mutates canonical graph tables, or calls models. |
| bitloops_red_team_suite.py | script | sandbox | reusable_prior | `scripts/bitloops_red_team_suite.py` | LUCIDOTA / Bitloops Airlock Red-Team Swarm Payload Safe-by-default, deterministic red-team fixtures for automation-loop airlocks. This is a ready-to-review payload only; it does not integrate with production. Sovereignty guarantees: - no te |
| board_effect_doctrine.py | script | sandbox | reusable_prior | `scripts/board_effect_doctrine.py` | Deterministic Santa/Krampus board-effect doctrine. Santa and Krampus are LUCIDOTA-local route personas/metaphors, not external technical terms. They gate how much board position a move must touch and how negative/audit claims are allowed to |
| boring_beast.py | script | sandbox | reusable_prior | `scripts/boring_beast.py` | Executable LUCIDOTA Boring Beast work-loop runtime. This is not a report generator. It provides DB-backed commands for queue-spine hardening, legal state transitions, work-order validation, consume-one workers, idempotency, retry/DLQ, execu |
| boring_beast_full_e2e.py | script | sandbox | reusable_prior | `scripts/boring_beast_full_e2e.py` | Full Boring Beast E2E verifier. Runs the core Boring Beast E2E, proves duplicate suppression, verifies direct graph writes stay blocked, derives status ledger facts, and emits one compact machine report. |
| build_bonsai_ternary_llama_cuda.sh | script | sandbox | reusable_prior | `scripts/build_bonsai_ternary_llama_cuda.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| build_llama_cuda.sh | script | sandbox | reusable_prior | `scripts/build_llama_cuda.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| bytewax_abductive_blender.py | script | sandbox | reusable_prior | `scripts/bytewax_abductive_blender.py` | LUCIDOTA Bytewax-compatible abductive hypothesis blender. Continuous local eventflow over command envelopes, workflow events, ABSURD rows, and optional ActivityWatch-style telemetry windows. The implementation uses real Bytewax when the pac |
| bytewax_absurd_stream_audit.py | script | sandbox | reusable_prior | `scripts/bytewax_absurd_stream_audit.py` | UNKNOWN — needs operator label; filename suggests: bytewax absurd stream audit |
| bytewax_chrono_lane_normalizer.py | script | sandbox | reusable_prior | `scripts/bytewax_chrono_lane_normalizer.py` | Chrono lane normalization stream runner. RFC-CHRONO-001 requires Bytewax-owned normalization semantics, no Docker, and no legacy queue-runtime dependency. Bytewax is optional at runtime on this host; this runner keeps the same append -> nor |
| bytewax_db_os_stream_audit.py | script | sandbox | reusable_prior | `scripts/bytewax_db_os_stream_audit.py` | Audit the Project2501/Bytewax stream for the Abductive DB OS lane. |
| canonical_graph_write_scanner.py | script | sandbox | reusable_prior | `scripts/canonical_graph_write_scanner.py` | LUCIDOTA Canonical Graph Write Scanner. Audits source code directories to catch direct database writes. Updated to seamlessly parse both legacy INSERT/UPDATE/DELETE syntax and high-speed Absurd binary COPY stream protocols. |
| case_dashboard_data.py | script | sandbox | reusable_prior | `scripts/case_dashboard_data.py` | UNKNOWN — needs operator label; filename suggests: case dashboard data |
| case_packet_compiler.py | script | sandbox | reusable_prior | `scripts/case_packet_compiler.py` | UNKNOWN — needs operator label; filename suggests: case packet compiler |
| case_packet_renderer.py | script | sandbox | reusable_prior | `scripts/case_packet_renderer.py` | UNKNOWN — needs operator label; filename suggests: case packet renderer |
| case_pipeline_dispatch.py | script | sandbox | reusable_prior | `scripts/case_pipeline_dispatch.py` | UNKNOWN — needs operator label; filename suggests: case pipeline dispatch |
| case_workspace.py | script | sandbox | reusable_prior | `scripts/case_workspace.py` | UNKNOWN — needs operator label; filename suggests: case workspace |
| catchme_scope_contract_dry_run.py | script | sandbox | reusable_prior | `scripts/catchme_scope_contract_dry_run.py` | Define CatchMe / Personal Context Spine consent, revocation, and sensitivity scopes. Dry-run only. No DB writes. No graph writes. |
| cep_builder.py | script | sandbox | reusable_prior | `scripts/cep_builder.py` | UNKNOWN — needs operator label; filename suggests: cep builder |
| cep_full_e2e.py | script | sandbox | reusable_prior | `scripts/cep_full_e2e.py` | CEP full E2E: surface instruction -> conversation_command -> ABSURD work order. Default mode is a graph-safe golden-path dry-run. Execute mode preserves the existing write path for staged conversation_command + ABSURD queue creation. |
| cep_to_kernel_route.py | script | sandbox | reusable_prior | `scripts/cep_to_kernel_route.py` | UNKNOWN — needs operator label; filename suggests: cep to kernel route |
| check_all_lucidota_services.py | script | sandbox | reusable_prior | `scripts/check_all_lucidota_services.py` | UNKNOWN — needs operator label; filename suggests: check all lucidota services |
| check_chrono_ledger_service.sh | script | sandbox | reusable_prior | `scripts/check_chrono_ledger_service.sh` | DB="${DATABASE_URL:-postgresql:///lucidota_storage}" |
| check_lucidota_intake_service.sh | script | sandbox | reusable_prior | `scripts/check_lucidota_intake_service.sh` | SERVICE="lucidota-intake.service" |
| chroma_gliner_bounded_probe.py | script | sandbox | reusable_prior | `scripts/chroma_gliner_bounded_probe.py` | Bounded read-only Chroma -> GLiNER claim-packet dry-run probe. Reads Chroma SQLite with keyset pagination, never loads the full DB/table, and runs the existing GLiNER claim-packet dry-run in bounded mini-batches. |
| chrono_audit_db_report.py | script | sandbox | reusable_prior | `scripts/chrono_audit_db_report.py` | Generate Chrono audit report from live DB facts only. |
| chrono_claim_chain_replay_audit_gate.py | script | sandbox | reusable_prior | `scripts/chrono_claim_chain_replay_audit_gate.py` | Chrono claim-chain/replay audit final gate. |
| chrono_conservation_verify.py | script | sandbox | reusable_prior | `scripts/chrono_conservation_verify.py` | Chrono conservation verifier CLI. Verifies that the current best-time projection links back to temporal_claim rows and that low-confidence/conflicting temporal evidence remains queryable. No temporal evidence is deleted, overwritten, or col |
| chrono_dead_letter_replay.py | script | sandbox | reusable_prior | `scripts/chrono_dead_letter_replay.py` | Chrono dead-letter replay safety CLI. Dry-run by default. Execute runs the Chrono backfill safely and marks a dead-letter resolved only if the target file gains a temporal_claim during the replay. It never deletes dead-letter rows and never |
| chrono_freeze_mtime.py | script | sandbox | reusable_prior | `scripts/chrono_freeze_mtime.py` | Dry-run/default mtime evidence freezer for Chrono-Ledger. Default mode is dry-run. Execute mode appends temporal_claim rows only and never mutates file_object/file_occurrence custody rows. |
| chrono_from_inventory.py | script | sandbox | reusable_prior | `scripts/chrono_from_inventory.py` | UNKNOWN — needs operator label; filename suggests: chrono from inventory |
| chrono_full_conservation_gate.py | script | sandbox | reusable_prior | `scripts/chrono_full_conservation_gate.py` | Full Chrono conservation gate after expanded ingest. |
| chrono_lane_split_projection_gate.py | script | sandbox | reusable_prior | `scripts/chrono_lane_split_projection_gate.py` | RFC-CHRONO-001 lane split, projection, and graph-promotion gate. This is an Absurd-compatible durable worker step: it preserves temporal_claim as the append-only ledger, writes only derived classification/projection/candidate rows, and refu |
| chrono_missed_event_simulation.py | script | sandbox | reusable_prior | `scripts/chrono_missed_event_simulation.py` | Simulate a missed Chrono LISTEN/NOTIFY event and prove replay/backfill recovery. |
| chrono_mtime_freeze_discrepancy.py | script | sandbox | reusable_prior | `scripts/chrono_mtime_freeze_discrepancy.py` | Explain Chrono processed_count vs mtime-freeze candidate row delta without mutation. |
| chrono_mtime_snapshot_audit.py | script | sandbox | reusable_prior | `scripts/chrono_mtime_snapshot_audit.py` | Audit mtime_snapshot_v1 temporal claims remain queryable and linked. |
| chrono_phase_b_validation.sql | script | sandbox | reusable_prior | `scripts/chrono_phase_b_validation.sql` | FILE: scripts/chrono_phase_b_validation.sql |
| chrono_phase_c_conservation_report.py | script | sandbox | reusable_prior | `scripts/chrono_phase_c_conservation_report.py` | Generate Chrono-Ledger Phase C report with Temporal Conservation Law checks. This script never deletes temporal evidence. It appends a ranking_pass and immutable ranking_result rows, then writes a JSON report. |
| chrono_phase_c_validation.sql | script | sandbox | reusable_prior | `scripts/chrono_phase_c_validation.sql` | FILE: scripts/chrono_phase_c_validation.sql |
| chrono_pre_deployment_check.sql | script | sandbox | reusable_prior | `scripts/chrono_pre_deployment_check.sql` | FILE: scripts/chrono_pre_deployment_check.sql |
| chrono_projection_claim_verifier.py | script | sandbox | reusable_prior | `scripts/chrono_projection_claim_verifier.py` | UNKNOWN — needs operator label; filename suggests: chrono projection claim verifier |
| chrono_queue_event_bridge.py | script | sandbox | reusable_prior | `scripts/chrono_queue_event_bridge.py` | Append ABSURD queue events into Chrono as temporal evidence. |
| chrono_ranking_pass.py | script | sandbox | reusable_prior | `scripts/chrono_ranking_pass.py` | Create immutable Chrono ranking pass selections. |
| chrono_replay_cursor_validator.py | script | sandbox | reusable_prior | `scripts/chrono_replay_cursor_validator.py` | Validate Chrono replay cursor rows and pending replay target counts. |
| chrono_service_activation_report.py | script | sandbox | reusable_prior | `scripts/chrono_service_activation_report.py` | Write Chrono-Ledger service activation report. Read-only except report file. |
| chrono_snapshot_line_slicer.py | script | sandbox | reusable_prior | `scripts/chrono_snapshot_line_slicer.py` | Newline-safe slicer for the Chrono master snapshot. The slicer is intentionally boring: - read bytes from the source file, - accumulate complete lines only, - rotate output parts before a line would cross the target size, - never split a JS |
| chrono_source_trust_validator.py | script | sandbox | reusable_prior | `scripts/chrono_source_trust_validator.py` | Validate Chrono evidence_source trust weights. |
| chrono_temporal_claim_invalidate.py | script | sandbox | reusable_prior | `scripts/chrono_temporal_claim_invalidate.py` | Invalidate temporal claims with evidence without deleting them. |
| chrono_temporal_conflict_detector.py | script | sandbox | reusable_prior | `scripts/chrono_temporal_conflict_detector.py` | Detect disputed Chrono files from preserved temporal claims. |
| chrono_timeline_projection_refresh.py | script | sandbox | reusable_prior | `scripts/chrono_timeline_projection_refresh.py` | Refresh derived current Chrono timeline projection with selected claim UUIDs. |
| chronological_migrator.sh | script | sandbox | reusable_prior | `scripts/chronological_migrator.sh` | KORPUS KRAMPII chronological migrator. |
| chunk_to_staging.py | script | sandbox | reusable_prior | `scripts/chunk_to_staging.py` | UNKNOWN — needs operator label; filename suggests: chunk to staging |
| ckdog_kernel_events.py | script | sandbox | reusable_prior | `scripts/ckdog_kernel_events.py` | UNKNOWN — needs operator label; filename suggests: ckdog kernel events |
| ckdog_kernel_route_plan.py | script | sandbox | reusable_prior | `scripts/ckdog_kernel_route_plan.py` | UNKNOWN — needs operator label; filename suggests: ckdog kernel route plan |
| claim_clusterer.py | script | sandbox | reusable_prior | `scripts/claim_clusterer.py` | UNKNOWN — needs operator label; filename suggests: claim clusterer |
| claim_support_score.py | script | sandbox | reusable_prior | `scripts/claim_support_score.py` | UNKNOWN — needs operator label; filename suggests: claim support score |
| claim_table_compiler.py | script | sandbox | reusable_prior | `scripts/claim_table_compiler.py` | UNKNOWN — needs operator label; filename suggests: claim table compiler |
| cohere_chat_cli.py | script | sandbox | reusable_prior | `scripts/cohere_chat_cli.py` | Cohere Chat API bridge with receipts and dry-run default. The key is never printed. Use --execute to make a network call; otherwise the command validates the request shape and writes a dry-run receipt only. |
| content_store.py | script | sandbox | reusable_prior | `scripts/content_store.py` | UNKNOWN — needs operator label; filename suggests: content store |
| contradiction_report.py | script | sandbox | reusable_prior | `scripts/contradiction_report.py` | UNKNOWN — needs operator label; filename suggests: contradiction report |
| control_packet_ledger.py | script | sandbox | reusable_prior | `scripts/control_packet_ledger.py` | UNKNOWN — needs operator label; filename suggests: control packet ledger |
| conversation_command_accept_worker.py | script | sandbox | reusable_prior | `scripts/conversation_command_accept_worker.py` | Accept staged conversation_command rows into ABSURD queue work orders. Generated surfaces remain conversation instruments. This worker takes already staged plain-language instructions, validates the no-direct-mutation contract, queues a ABS |
| cruelty_protocols_guard.py | script | sandbox | reusable_prior | `scripts/cruelty_protocols_guard.py` | Cruelty Protocols executable schema/guardrail verifier. |
| decision_boundary_memory_dry_run.py | script | sandbox | reusable_prior | `scripts/decision_boundary_memory_dry_run.py` | DeMem preserves decision-changing distinctions Research-only dry-run scaffold. No DB writes. No graph writes. |
| demem_surface_boundary_bridge_dry_run.py | script | sandbox | reusable_prior | `scripts/demem_surface_boundary_bridge_dry_run.py` | Bridge DeMem decision boundaries into surface instruction compiler constraints. Dry-run only. No DB writes. No graph writes. Boundaries become guardrails, not mutation authority. |
| dev_journey_decision_points.py | script | sandbox | reusable_prior | `scripts/dev_journey_decision_points.py` | Compile dev-journey decision points into sticker + XGBoost/Treelite training rows. This is advisory ML substrate only. It does not declare truth and does not write canonical graph state. It turns receipts/logs/hunches into repeatable weak l |
| dev_library_scan.py | script | sandbox | reusable_prior | `scripts/dev_library_scan.py` | LUCIDOTA Dev Library scanner CLI. Human-facing wrapper around the legacy manifest implementation. Use this name in new docs and operator workflows; keep the old implementation until a full receipt-backed rename is safe. |
| dev_order_gate.py | script | sandbox | reusable_prior | `scripts/dev_order_gate.py` | UNKNOWN — needs operator label; filename suggests: dev order gate |
| dev_order_matrix_wrapper.py | script | sandbox | reusable_prior | `scripts/dev_order_matrix_wrapper.py` | UNKNOWN — needs operator label; filename suggests: dev order matrix wrapper |
| diogenes_memory_gate.py | script | sandbox | reusable_prior | `scripts/diogenes_memory_gate.py` | DIOGENES / CKDOG1 low-RAM gate. Runs the local doggystyle CKDOG1 hot-path benches plus PercyphonAI's zero-VRAM procedural scaffold and writes one compact receipt. |
| diogenes_stub_kernel.py | script | sandbox | reusable_prior | `scripts/diogenes_stub_kernel.py` | Marrow Loop v0 command receipt kernel. Default is dry-run. A receipt is always written. DB writes are disabled unless --execute and --execute-db are both explicitly passed. |
| document_claim_packet_worker.py | script | sandbox | reusable_prior | `scripts/document_claim_packet_worker.py` | Document parse span -> GLiNER-style claim packet bridge. This is executable production-path plumbing, not graph truth: - reads parsed document spans from lucidota_korpus.document_parse_* tables - extracts exact matched text from parser Mark |
| document_parse_bakeoff.py | script | sandbox | reusable_prior | `scripts/document_parse_bakeoff.py` | Isolated document parser bakeoff for the LUCIDOTA tech bench. Bench-only: no production mutation, no parser output treated as truth. Checks MinerU, Docling, and SmolDocling installability/import state and, when --execute is used, runs only  |
| document_parse_ingest.py | script | sandbox | reusable_prior | `scripts/document_parse_ingest.py` | Production document parse ingestion path. Local-first parser: text/markdown/json/csv become Markdown+JSON provenance records. Binary/unsupported documents are deferred unless a real parser backend is installed. Parser output is NOT truth; i |
| document_parse_router.py | script | sandbox | reusable_prior | `scripts/document_parse_router.py` | UNKNOWN — needs operator label; filename suggests: document parse router |
| durable_workflow_decision_check.py | script | sandbox | reusable_prior | `scripts/durable_workflow_decision_check.py` | Check that ABSURD is current and DBOS is legacy-compatibility only. |
| entity_candidate_registry.py | script | sandbox | reusable_prior | `scripts/entity_candidate_registry.py` | UNKNOWN — needs operator label; filename suggests: entity candidate registry |
| epistemic_trace_dry_run.py | script | sandbox | reusable_prior | `scripts/epistemic_trace_dry_run.py` | TRACER-lite labels epistemic moves including PFM Research-only dry-run scaffold. No DB writes. No graph writes. |
| event_candidate_registry.py | script | sandbox | reusable_prior | `scripts/event_candidate_registry.py` | UNKNOWN — needs operator label; filename suggests: event candidate registry |
| experienced_colleague_eval_seed_dry_run.py | script | sandbox | reusable_prior | `scripts/experienced_colleague_eval_seed_dry_run.py` | LongMemEval-style seeds after timeline ingestion Research-only dry-run scaffold. No DB writes. No graph writes. |
| export_bundle.py | script | sandbox | reusable_prior | `scripts/export_bundle.py` | UNKNOWN — needs operator label; filename suggests: export bundle |
| fast_recall_scout_dry_run.py | script | sandbox | reusable_prior | `scripts/fast_recall_scout_dry_run.py` | SimpleMem-style fast recall scout; candidates only Research-only dry-run scaffold. No DB writes. No graph writes. |
| fast_slow_lane_gate.py | script | sandbox | reusable_prior | `scripts/fast_slow_lane_gate.py` | Deterministic fastlane/slowlane metadata gate with cache-to-slowlane flushes. PocketFlow lesson: tiny packet -> decision -> post hook. Syd lesson: extract and validate facts before any model. llm-router lesson: route metadata, not vibes. Th |
| full_system_soak_audit.py | script | sandbox | reusable_prior | `scripts/full_system_soak_audit.py` | Receipt-backed full-system soak auditor. This does not run a daemon and does not mutate the graph. It answers one narrow question: do the required runtime lanes have PASS receipts spanning at least N hours, while canonical graph writes stay |
| gauntlet_state_promoter.py | script | sandbox | reusable_prior | `scripts/gauntlet_state_promoter.py` | Promote RGAUNTLET work-order states from validated proof packets. Streaming rules: - proof packet ingest is bounded by --max-proofs - master gauntlet is streamed line-by-line - source gauntlet is never mutated; a timestamped replacement JSO |
| generate_bucket_deep_passes.py | script | sandbox | reusable_prior | `scripts/generate_bucket_deep_passes.py` | Generate per-bucket deep-pass artifacts from existing bucket pass markdown reports. |
| generate_lucidota_architecture_manifest.py | script | sandbox | reusable_prior | `scripts/generate_lucidota_architecture_manifest.py` | UNKNOWN — needs operator label; filename suggests: generate lucidota architecture manifest |
| gliner_claim_packet_dry_run.py | script | sandbox | reusable_prior | `scripts/gliner_claim_packet_dry_run.py` | Bench-only GLiNER span -> ClaimPacket dry-run layer. Takes existing GLiNER/proof-hoard extraction behavior and emits claim candidates with evidence refs. These are not graph truth and are not written to canonical graph tables. |
| goal_agent_packet.py | script | sandbox | reusable_prior | `scripts/goal_agent_packet.py` | Export one coding-only GOALS packet for Codex/Claude/OpenCode/Aider/Continue/llxprt-code/generic agents. |
| goal_chain.py | script | sandbox | reusable_prior | `scripts/goal_chain.py` | Tiny GOALS next-goal packet queue. |
| goal_dev_control.py | script | sandbox | reusable_prior | `scripts/goal_dev_control.py` | Tiny GOALS cadence/effective-LOC/supply router; no model calls. |
| goal_handoff.py | script | sandbox | reusable_prior | `scripts/goal_handoff.py` | Tiny GOALS handoff helper; deliberately under 100 non-comment code lines. |
| goal_model_fabric_control.py | script | sandbox | reusable_prior | `scripts/goal_model_fabric_control.py` | Tiny GOAL 3 local model fabric status/start/stop helper; no daemon. |
| goal_model_fabric_orchestrate.py | script | sandbox | reusable_prior | `scripts/goal_model_fabric_orchestrate.py` | Queue model-fabric bringup/proof jobs through ABSURD/Postgres. |
| goal_scenario_batch.py | script | sandbox | reusable_prior | `scripts/goal_scenario_batch.py` | Deterministic asymmetric-wargame scenario batch harness. Runs cheap scenario passes through the existing language router, compresses the observed decision pairs, and writes a receipt-backed batch report. |
| goal_scenario_compare.py | script | sandbox | reusable_prior | `scripts/goal_scenario_compare.py` | Compare scenario-batch/holdout receipts and propose the next seed. |
| goal_swarm_brief.py | script | sandbox | reusable_prior | `scripts/goal_swarm_brief.py` | Emit compact GOALS swarm packets from the current swarm brief. |
| goal_swarm_dispatch.py | script | sandbox | reusable_prior | `scripts/goal_swarm_dispatch.py` | Bridge GOALS packets into durable ABSURD queue jobs with receipts. |
| goal_system_index.py | script | sandbox | reusable_prior | `scripts/goal_system_index.py` | Tiny system/function index for GOALS dev mode. |
| goal_telemetry.py | script | sandbox | reusable_prior | `scripts/goal_telemetry.py` | Tiny CPU/RAM/VRAM/PCI/temp telemetry sampler for GOALS. |
| golden_path_regression_gate.py | script | sandbox | reusable_prior | `scripts/golden_path_regression_gate.py` | UNKNOWN — needs operator label; filename suggests: golden path regression gate |
| gpu_runtime_budget.py | script | sandbox | reusable_prior | `scripts/gpu_runtime_budget.py` | UNKNOWN — needs operator label; filename suggests: gpu runtime budget |
| graph_candidate.py | script | sandbox | reusable_prior | `scripts/graph_candidate.py` | UNKNOWN — needs operator label; filename suggests: graph candidate |
| graph_canonical_mutation_detector.py | script | sandbox | reusable_prior | `scripts/graph_canonical_mutation_detector.py` | Detect canonical graph mutation around graph worker commands. |
| graph_edge_materialize.py | script | sandbox | reusable_prior | `scripts/graph_edge_materialize.py` | Materialize a graph edge through the governed promotion path. |
| graph_edge_write_blocker_probe.py | script | sandbox | reusable_prior | `scripts/graph_edge_write_blocker_probe.py` | Regression probe: direct graph_edge writes are blocked outside promotion path. |
| graph_item_approval_guard_probe.py | script | sandbox | reusable_prior | `scripts/graph_item_approval_guard_probe.py` | Probe graph_item approved guard: operator UUID, scope, time required. |
| graph_journal_completeness_check.py | script | sandbox | reusable_prior | `scripts/graph_journal_completeness_check.py` | Audit graph promotion materializations for journal/command/evidence completeness. |
| graph_journal_replay_audit.py | script | sandbox | reusable_prior | `scripts/graph_journal_replay_audit.py` | Replay/audit graph promotion journal materialization receipts. |
| graph_materialization_helper.py | script | sandbox | reusable_prior | `scripts/graph_materialization_helper.py` | Reusable hardening helper for graph promotion materialization. This helper does not provide a direct graph-write shortcut. It wraps the existing guarded promotion path and then proves the required receipt chain: conversation_command -> prom |
| graph_materialization_receipt_query.py | script | sandbox | reusable_prior | `scripts/graph_materialization_receipt_query.py` | Query and validate graph promotion materialization receipts. |
| graph_materialization_rollback_probe.py | script | sandbox | reusable_prior | `scripts/graph_materialization_rollback_probe.py` | Prove graph promotion execute path fails safely on invalid payloads. |
| graph_promotion.py | script | sandbox | reusable_prior | `scripts/graph_promotion.py` | UNKNOWN — needs operator label; filename suggests: graph promotion |
| graph_promotion_approval_worker.py | script | sandbox | reusable_prior | `scripts/graph_promotion_approval_worker.py` | Graph promotion approval state machine. Enforces candidate -> defer/reject/operator_confirmed -> materialized. It writes promotion packet/decision/transition receipts only. Materialized state requires an existing graph_promotion_materializa |
| graph_promotion_dry_run.py | script | sandbox | reusable_prior | `scripts/graph_promotion_dry_run.py` | Graph promotion dry-run scaffold. Writes a report only. Performs no DB writes and no graph mutation. |
| graph_promotion_execute.py | script | sandbox | reusable_prior | `scripts/graph_promotion_execute.py` | Execute-gated graph promotion packet/decision writer. This is the DB execute path for the promotion pipeline only. It never writes canonical graph_item, graph_edge, or graph_journal rows. |
| graph_promotion_full_e2e.py | script | sandbox | reusable_prior | `scripts/graph_promotion_full_e2e.py` | Full graph promotion E2E: surface instruction -> command -> promotion helper -> receipts. |
| graph_promotion_gate.py | script | sandbox | reusable_prior | `scripts/graph_promotion_gate.py` | Runtime graph promotion gate: evidence + authority + preflight before packet creation. |
| graph_promotion_materialize.py | script | sandbox | reusable_prior | `scripts/graph_promotion_materialize.py` | Materialize an evidence-gated graph promotion packet into canonical graph. This is the promotion path, not a direct graph-write shortcut: - verifies a conversation_command UUID exists in the control database - runs graph_promotion_preflight |
| graph_promotion_orphan_detector.py | script | sandbox | reusable_prior | `scripts/graph_promotion_orphan_detector.py` | Detect orphan graph promotion packets/decisions/materializations. |
| graph_promotion_packet_reviewer.py | script | sandbox | reusable_prior | `scripts/graph_promotion_packet_reviewer.py` | CLI reviewer for graph promotion packets: list/defer/reject/operator_confirmed. |
| graph_promotion_policy_check.py | script | sandbox | reusable_prior | `scripts/graph_promotion_policy_check.py` | UNKNOWN — needs operator label; filename suggests: graph promotion policy check |
| graph_promotion_soak_test.py | script | sandbox | reusable_prior | `scripts/graph_promotion_soak_test.py` | Graph promotion soak: invalid packets blocked, valid staged packets accepted without direct canonical mutation. |
| graph_role_policy_validator.py | script | sandbox | reusable_prior | `scripts/graph_role_policy_validator.py` | Validate graph role promotion policy matrix in executable code. |
| graph_store_adapter.py | script | sandbox | reusable_prior | `scripts/graph_store_adapter.py` | UNKNOWN — needs operator label; filename suggests: graph store adapter |
| graph_symbol_compare.py | script | sandbox | reusable_prior | `scripts/graph_symbol_compare.py` | Compare two symbol-condensation receipts and propose the next seed. |
| graph_symbol_condensation.py | script | sandbox | reusable_prior | `scripts/graph_symbol_condensation.py` | Confidence-scored condensation of symbol streams into evidence-linked claims. |
| graph_symbol_dispatch.py | script | sandbox | reusable_prior | `scripts/graph_symbol_dispatch.py` | Turn a symbol-compare receipt into compact GO-25 worker packets. |
| graph_write_blocker_probe.py | script | sandbox | reusable_prior | `scripts/graph_write_blocker_probe.py` | Probe canonical graph write barrier without mutating graph state. |
| groq_chat_cli.py | script | sandbox | reusable_prior | `scripts/groq_chat_cli.py` | Groq Chat Completions bridge with receipts and dry-run default. Uses Groq's OpenAI-compatible Chat Completions endpoint. API keys are read from environment only and are never printed or written to receipts. |
| groq_env.py | script | sandbox | reusable_prior | `scripts/groq_env.py` | Load Groq credentials from .env when the shell did not export them. |
| groq_goal_delegate.py | script | sandbox | reusable_prior | `scripts/groq_goal_delegate.py` | Bound one GOALS slice for Groq; writes wrapper receipt and subreceipt. |
| groq_model_catalog.py | script | sandbox | reusable_prior | `scripts/groq_model_catalog.py` | List/rank live Groq models with redacted receipts; dry-run by default. |
| hipporag_locked_gate_report.py | script | sandbox | reusable_prior | `scripts/hipporag_locked_gate_report.py` | NOT ALLOWED TO PLAY WITH UNTIL YOU CLEAN YOUR GRAPH PONYBOY. Research-only dry-run scaffold. No DB writes. No graph writes. |
| hunch_hypertimeline_audit.py | script | sandbox | reusable_prior | `scripts/hunch_hypertimeline_audit.py` | Deterministic hunch/hypertimeline audit packetizer. This does not decide truth. It extracts hunch records, preserves source hashes, compares parsed counts against stated ledgers, and emits observation-center state for downstream graph/Indy/ |
| hunch_postgres_ingest.py | script | sandbox | reusable_prior | `scripts/hunch_postgres_ingest.py` | Ingest hunch audit records into Postgres without promoting graph truth. |
| hypertimeline_engine.py | script | sandbox | reusable_prior | `scripts/hypertimeline_engine.py` | LUCIDOTA ABSURD hypertimeline sweep, ingest, and circadian engine test. |
| ignite_pillars.sh | script | sandbox | reusable_prior | `scripts/ignite_pillars.sh` | bin/bash |
| import_export_bundle.py | script | sandbox | reusable_prior | `scripts/import_export_bundle.py` | UNKNOWN — needs operator label; filename suggests: import export bundle |
| import_policy.py | script | sandbox | reusable_prior | `scripts/import_policy.py` | UNKNOWN — needs operator label; filename suggests: import policy |
| indy_book_learning_pipeline.py | script | sandbox | reusable_prior | `scripts/indy_book_learning_pipeline.py` | INDY_READs book learning pipeline: fetch/read -> <=500-token chunks -> GO packets -> LoRA job. |
| indy_reads.py | script | sandbox | reusable_prior | `scripts/indy_reads.py` | INDY_READs — GO-25 page-by-page reading game. INDY_READs is a she: reading companion, margin-noter, judgment collector. Dynamic library: /home/mfspx/LUCIDOTA/BOOKS State/data: /home/mfspx/LUCIDOTA/BOOKS/.indy_reads No page rewind. One page  |
| indy_reads_absurd_brief.py | script | sandbox | reusable_prior | `scripts/indy_reads_absurd_brief.py` | UNKNOWN — needs operator label; filename suggests: indy reads absurd brief |
| indy_reads_db_os_brief.py | script | sandbox | reusable_prior | `scripts/indy_reads_db_os_brief.py` | Summarize Indy_READS evidence posture from the Abductive DB OS board. |
| indy_reads_safe_books_batch.py | script | sandbox | reusable_prior | `scripts/indy_reads_safe_books_batch.py` | Safe INDY_READs BOOKS batch wrapper: local extract -> <=500 chunks -> LoRA staging receipts. This wrapper deliberately reuses the legacy local BOOKS extractors, then feeds extracted UTF-8 text into scripts.indy_book_learning_pipeline. It pe |
| install_chrono_ledger_service.sh | script | sandbox | reusable_prior | `scripts/install_chrono_ledger_service.sh` | ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| install_lucidota_intake_service.sh | script | sandbox | reusable_prior | `scripts/install_lucidota_intake_service.sh` | ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| instruction_conflict_scanner.py | script | sandbox | reusable_prior | `scripts/instruction_conflict_scanner.py` | UNKNOWN — needs operator label; filename suggests: instruction conflict scanner |
| interpretability_eval_bench_dry_run.py | script | sandbox | reusable_prior | `scripts/interpretability_eval_bench_dry_run.py` | CE-Bench-style future validation; no pretty latent ghosts Research-only dry-run scaffold. No DB writes. No graph writes. |
| kernel_control_packet.py | script | sandbox | reusable_prior | `scripts/kernel_control_packet.py` | CKDOG1 control-packet helpers for Python workers. This mirrors lucidota-core ControlPacket hashing so ABSURD/Postgres worker scripts can require kernel-style authorization receipts before enqueueing executable work. |
| kernel_packet_cli.py | script | sandbox | reusable_prior | `scripts/kernel_packet_cli.py` | UNKNOWN — needs operator label; filename suggests: kernel packet cli |
| knowledge_library_check.py | script | sandbox | reusable_prior | `scripts/knowledge_library_check.py` | UNKNOWN — needs operator label; filename suggests: knowledge library check |
| korpus_derived_compute_worker.py | script | sandbox | reusable_prior | `scripts/korpus_derived_compute_worker.py` | KORPUS derived-compute queue worker. Native Postgres queue with single-row leases (FOR UPDATE SKIP LOCKED). Designed for post-raw-ingest enrichment so River, MinHash, and graph promotion never hold the evidence storage pump hostage again. |
| korpus_embedding_worker.sh | script | sandbox | reusable_prior | `scripts/korpus_embedding_worker.sh` | ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}" |
| korpus_krampii.py | script | sandbox | reusable_prior | `scripts/korpus_krampii.py` | UNKNOWN — needs operator label; filename suggests: korpus krampii |
| krampus_bounded_inventory.py | script | sandbox | reusable_prior | `scripts/krampus_bounded_inventory.py` | Bounded KRAMPUSCHEWING inventory with safe hashing and error capture. |
| krampus_chrono_ledger.py | script | sandbox | reusable_prior | `scripts/krampus_chrono_ledger.py` | KORPUS KRAMPII chrono-ledger builder. Builds the Time Machine ledger: absolute_path, iso_timestamp, source, confidence_bps, size_bytes, suffix Order is deterministic and content-aware: explicit frontmatter/date fields -> filename/folder dat |
| krampus_extension_policy.py | script | sandbox | reusable_prior | `scripts/krampus_extension_policy.py` | UNKNOWN — needs operator label; filename suggests: krampus extension policy |
| krampus_inventory.py | script | sandbox | reusable_prior | `scripts/krampus_inventory.py` | UNKNOWN — needs operator label; filename suggests: krampus inventory |
| krampus_rechronologize.py | script | sandbox | reusable_prior | `scripts/krampus_rechronologize.py` | Forensic re-chronologizer and River replay for KORPUS KRAMPII. Purpose: - Do NOT require the whole corpus to arrive before ingestion. - Preserve raw/CAS first, then improve original-file dates as more metadata lands. - Rebuild DBSTREAM/brai |
| krampus_time_machine.py | script | sandbox | reusable_prior | `scripts/krampus_time_machine.py` | KORPUS KRAMPII Time Machine orchestrator. One command for the historical corpus run: 1. build chrono-ledger from content/path/frontmatter dates; 2. feed brain map in exact chronological order; 3. run durable KORPUS ingest sequentially so Ri |
| krampuschewing_absurd_adapter.py | script | sandbox | reusable_prior | `scripts/krampuschewing_absurd_adapter.py` | UNKNOWN — needs operator label; filename suggests: krampuschewing absurd adapter |
| krampuschewing_archive_unpack.py | script | sandbox | reusable_prior | `scripts/krampuschewing_archive_unpack.py` | Controlled KRAMPUSCHEWING archive unpack into staging only. |
| krampuschewing_chrono_graph.py | script | sandbox | reusable_prior | `scripts/krampuschewing_chrono_graph.py` | Build KRAMPUSCHEWING chronology graph packets from an index JSONL. |
| krampuschewing_db_os_adapter.py | script | sandbox | reusable_prior | `scripts/krampuschewing_db_os_adapter.py` | Expose KRAMPUSCHEWING receipts as Abductive DB OS typed counts. |
| krampuschewing_derived_recovery_gate.py | script | sandbox | reusable_prior | `scripts/krampuschewing_derived_recovery_gate.py` | Derived KRAMPUSCHEWING recovery gate from existing index only. |
| krampuschewing_graph_materialize.py | script | sandbox | reusable_prior | `scripts/krampuschewing_graph_materialize.py` | KRAMPUSCHEWING Streaming Graph Materializer (Absurd COPY Pipeline) Utilizes native Postgres COPY. Dynamically registers unknown ontology terms to satisfy strict Foreign Key constraints. Spoofs transaction variables to appease PL/pgSQL trigg |
| krampuschewing_graph_stage.py | script | sandbox | reusable_prior | `scripts/krampuschewing_graph_stage.py` | Emit KRAMPUSCHEWING graph-staging candidates only; no DB writes. |
| krampuschewing_large_file_validate.py | script | sandbox | reusable_prior | `scripts/krampuschewing_large_file_validate.py` | Validate large KRAMPUSCHEWING files that were intentionally not fully hashed. This is a metadata/partial-fingerprint pass only: - no archive unpacking - no full-file hashing - no file moves/deletes - no graph materialization/writes |
| krampuschewing_master_index.py | script | sandbox | reusable_prior | `scripts/krampuschewing_master_index.py` | Build a non-mutating KRAMPUSCHEWING master index. |
| krampuschewing_reclassify_index.py | script | sandbox | reusable_prior | `scripts/krampuschewing_reclassify_index.py` | Normalize/reclassify an existing KRAMPUSCHEWING index without touching files. |
| krampuschewing_recovery_gate.py | script | sandbox | reusable_prior | `scripts/krampuschewing_recovery_gate.py` | Run non-mutating KRAMPUSCHEWING recovery engine outputs. |
| krampuschewing_river_rows.py | script | sandbox | reusable_prior | `scripts/krampuschewing_river_rows.py` | Emit River-ready KRAMPUSCHEWING training-row candidates; does not train. |
| krampuschewing_role_discovery.py | script | sandbox | reusable_prior | `scripts/krampuschewing_role_discovery.py` | KRAMPUSCHEWING executable-spine / role discovery. Read-only: does not start, stop, restart, move files, delete files, train models, or materialize canonical graph rows. |
| krampuschewing_watcher.sh | script | sandbox | reusable_prior | `scripts/krampuschewing_watcher.sh` | ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}" |
| language_router.py | script | sandbox | reusable_prior | `scripts/language_router.py` | Deterministic language subsystem: messy text -> GO/JSON lanes/work orders. |
| latent_mode_registry_scan_dry_run.py | script | sandbox | reusable_prior | `scripts/latent_mode_registry_scan_dry_run.py` | Scan model custody gaps; do not run models Research-only dry-run scaffold. No DB writes. No graph writes. |
| DBOS_LEGACY_ARCHIVE_MANIFEST.json | script | sandbox | reusable_prior | `scripts/legacy/DBOS_LEGACY_ARCHIVE_MANIFEST.json` | UNKNOWN — needs operator label; filename suggests: DBOS LEGACY ARCHIVE MANIFEST |
| README.md | script | sandbox | reusable_prior | `scripts/legacy/README.md` | Archived DBOS legacy utilities |
| absurd_worker_script.py | script | sandbox | reusable_prior | `scripts/legacy/absurd_worker_script.py` | LUCIDOTA Absurd Workflow Engine Spine. Replaces DBOS with a native, state-machine driven Postgres workflow engine. No more SKIP LOCKED contention. Workflows are durable, resumable state functions. |
| absurd_worker_spine.py | script | sandbox | reusable_prior | `scripts/legacy/absurd_worker_spine.py` | LUCIDOTA Absurd Workflow Engine Spine. Small durable state-machine queue on Postgres. Critical rule: the control-table row is locked only while it is claimed or finished. Handler code runs outside the control transaction so long CAS/file/IO |
| absured_worker_script.py | script | sandbox | reusable_prior | `scripts/legacy/absured_worker_script.py` | LUCIDOTA Absurd Workflow Engine Spine. Replaces DBOS with a native, state-machine driven Postgres workflow engine. No more SKIP LOCKED contention. Workflows are durable, resumable state functions. |
| chrono_ledger_duplicate_reclaimer.py | script | sandbox | reusable_prior | `scripts/legacy/active_root_pruned/chrono_ledger_duplicate_reclaimer.py` | Ledger-only duplicate reclamation planner for LUCIDOTA. This script does not hash source files and does not read source file contents. It trusts the already-built Chrono ledger's source_sha256 fields as the hash oracle, then resolves one ma |
| lucidota_fabric_press.py | script | sandbox | reusable_prior | `scripts/legacy/active_root_pruned/lucidota_fabric_press.py` | LUCIDOTA fabric press diagnostic + orchestration sweep. This is a thin coordinator over existing production scripts. It does not invent duplicate simulator or runtime infrastructure; it only wires what already exists and emits a structured  |
| ouroboros_filesystem_audit.py | script | sandbox | reusable_prior | `scripts/legacy/active_root_pruned/ouroboros_filesystem_audit.py` | Generate whole-filesystem audit maps for the Ouroboros script survival loop. |
| script_survival_batch_audit.py | script | sandbox | reusable_prior | `scripts/legacy/active_root_pruned/script_survival_batch_audit.py` | Batch baseline survival-audit every TICKLETRUNK script entry. This is a coverage repair tool: it appends deterministic audit rows for every TICKLETRUNK SCRIPTS path not already present in SCRIPT_AUDIT_MANIFEST.jsonl. It never deletes or mov |
| agentsi.py | script | sandbox | reusable_prior | `scripts/legacy/agentsi.py` | agentSI — Agent Self-Sovereign Job Fair MVP. A deterministic/local MVP for agent identity, career matching, growth journaling, and job-fair outputs. No external model calls. |
| audit_verdict_enforcer.py | script | sandbox | reusable_prior | `scripts/legacy/audit_verdict_enforcer.py` | Strict audit verdict contract enforcer. |
| boring_beast.py | script | sandbox | reusable_prior | `scripts/legacy/boring_beast.py` | Executable LUCIDOTA Boring Beast work-loop runtime. This is not a report generator. It provides DB-backed commands for queue hardening, legal state transitions, work-order validation, consume-one workers, idempotency, retry/DLQ, execution r |
| boring_beast_event_chain_verify.py | script | sandbox | reusable_prior | `scripts/legacy/boring_beast_event_chain_verify.py` | UNKNOWN — needs operator label; filename suggests: boring beast event chain verify |
| boring_beast_loop_runner.py | script | sandbox | reusable_prior | `scripts/legacy/boring_beast_loop_runner.py` | Run Boring Beast real-code loops against executable DB/runtime paths. |
| catchme_context_guard.py | script | sandbox | reusable_prior | `scripts/legacy/catchme_context_guard.py` | Runtime CatchMe consent/sensitivity guard for context use. |
| catchme_sensitivity_map.py | script | sandbox | reusable_prior | `scripts/legacy/catchme_sensitivity_map.py` | UNKNOWN — needs operator label; filename suggests: catchme sensitivity map |
| cep_graph_packet_stager.py | script | sandbox | reusable_prior | `scripts/legacy/cep_graph_packet_stager.py` | Stage graph promotion packets from CEP conversation_command rows. |
| chrono_daemon_activation_refresher.py | script | sandbox | reusable_prior | `scripts/legacy/chrono_daemon_activation_refresher.py` | Refresh Chrono daemon activation report and write a runtime fact. |
| chrono_queue_event_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/chrono_queue_event_bridge.py` | Append DBOS queue events into Chrono as temporal evidence. |
| chrono_service_health_recorder.py | script | sandbox | reusable_prior | `scripts/legacy/chrono_service_health_recorder.py` | Record Chrono service health as runtime facts and update STATUS_LEDGER from facts. |
| conversation_command_accept_worker.py | script | sandbox | reusable_prior | `scripts/legacy/conversation_command_accept_worker.py` | Accept staged conversation_command rows into DBOS queue work orders. Generated surfaces remain conversation instruments. This worker takes already staged plain-language instructions, validates the no-direct-mutation contract, queues a DBOS  |
| conversation_command_idempotency_probe.py | script | sandbox | reusable_prior | `scripts/legacy/conversation_command_idempotency_probe.py` | Probe CEP conversation_command content idempotency. |
| conversation_command_status_worker.py | script | sandbox | reusable_prior | `scripts/legacy/conversation_command_status_worker.py` | Enforce conversation_command legal status transitions. |
| daemon_supervision_preflight.py | script | sandbox | reusable_prior | `scripts/legacy/daemon_supervision_preflight.py` | Daemon launch/supervision preflight with machine-readable failure records. MVP control-plane rule: inspect and optionally start missing local loops; never kill healthy daemons and never reset queues from this script. |
| dbos_chrono_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_chrono_worker.py` | DBOS queue-spine wrapper for Chrono-Ledger. The wrapper observes Chrono service health and writes DBOS queue/job/workflow receipts. It never deletes, overwrites, invalidates, or inserts temporal claims. |
| dbos_consume_one.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_consume_one.py` | Generic consume-one DBOS worker path for a named queue. |
| dbos_corpus_job_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_corpus_job_bridge.py` | DBOS-compatible corpus lane job bridge for KORPUS/KRAMPUS custody work. |
| dbos_dead_letter_review.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_dead_letter_review.py` | DBOS queue dead-letter review CLI. Lists, classifies, and retries DBOS queue dead-letter rows only with explicit --execute for mutations. It never deletes DLQ rows and never marks a DLQ row resolved as a side effect of retry creation. |
| dbos_document_parse_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_document_parse_worker.py` | DBOS wrapper for document parse ingestion. Queues document parser jobs, consumes one job safely, calls the production local parser ingestion path, and records DBOS queue events. Parser output is NOT truth and canonical graph writes are neve |
| dbos_fault_injector.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_fault_injector.py` | UNKNOWN — needs operator label; filename suggests: dbos fault injector |
| dbos_graph_promotion_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_graph_promotion_worker.py` | DBOS wrapper for graph promotion packets. Writes promotion packet/decision only; no direct canonical materialization. |
| dbos_intake_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_intake_worker.py` | DBOS custody wrapper for the supervised Rust lucidota-intake service. Observes service/drop-dir state and writes DBOS workflow/custody receipts. It does not move drops. |
| dbos_krampus_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_krampus_worker.py` | DBOS queue-spine wrapper for KRAMPUSCHEWING/KORPUS. This is an observation/health wrapper. It writes DBOS queue/job/workflow receipts only. It never ingests drops, moves files, deletes files, mutates temporal claims, or mutates KORPUS custo |
| dbos_queue_hardening_check.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_queue_hardening_check.py` | Apply/check DBOS queue schema hardening v2. |
| dbos_queue_metrics_exporter.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_queue_metrics_exporter.py` | Export DBOS queue metrics into lucidota_control.runtime_status_fact. |
| dbos_queue_scheduler_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_queue_scheduler_probe.py` | Probe DBOS queue scheduler run_after/priority selection across queues. |
| dbos_queue_soak_test.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_queue_soak_test.py` | DBOS queue spine soak: many jobs, retries, duplicate suppression. |
| dbos_queue_spine.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_queue_spine.py` | LUCIDOTA DBOS-compatible durable queue spine. Dry-run is default for actions that can write. Execute mode writes only queue-spine state (jobs/events/dead-letter/workflow_event), never canonical graph tables. |
| dbos_retry_dead_letter_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_retry_dead_letter_probe.py` | Probe DBOS retry/dead-letter behavior with a bounded failing job. |
| dbos_retry_policy.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_retry_policy.py` | Configure and validate per-queue DBOS retry policy. |
| dbos_river_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_river_worker.py` | DBOS queue-spine wrapper for River/Bytewax and Phase 0.5 GLiNER extraction. Primary queue for this pass: dbos.phase05_streaming_brain / gliner_zero_shot_extract Safety laws: - Writes DBOS job/event/workflow receipts and lucidota_learning GL |
| dbos_skip_locked_contention_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_skip_locked_contention_probe.py` | Prove DBOS queue multi-worker contention uses SKIP LOCKED. |
| dbos_stale_lock_recovery_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_stale_lock_recovery_probe.py` | Create and recover a stale in-flight DBOS job. |
| dbos_surface_cep_worker.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_surface_cep_worker.py` | DBOS wrapper for Surface/CEP fan-in. Writes queue/workflow receipts and optionally stages conversation_command rows. No canonical graph mutation. |
| dbos_transition_law_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_transition_law_probe.py` | Apply and probe table-driven DBOS transition law. |
| dbos_worker_contract_registry_check.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_worker_contract_registry_check.py` | Validate required DBOS worker contracts for core LUCIDOTA workers. |
| dbos_worker_harness.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_worker_harness.py` | Generic DBOS worker harness: claim/execute/record/retry primitives. |
| dbos_worker_heartbeat_probe.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_worker_heartbeat_probe.py` | Probe worker heartbeat writes and stale recovery by last_heartbeat_at. |
| dbos_worker_supervisor_preflight.py | script | sandbox | reusable_prior | `scripts/legacy/dbos_worker_supervisor_preflight.py` | Preflight registered DBOS workers before supervisor launch. |
| demem_runtime_guard.py | script | sandbox | reusable_prior | `scripts/legacy/demem_runtime_guard.py` | Executable DeMem runtime boundary guard. Given a plain-language instruction, checks active lucidota_control.demem_boundary rules and returns allow/warn/rewrite/block. Execute mode records the decision. |
| diogenes_30_phase_audit.py | script | sandbox | reusable_prior | `scripts/legacy/diogenes_30_phase_audit.py` | DIOGENES / LUCIDOTA 30-phase architecture audit. This is an evidence audit, not a build loop. It reads filesystem, SQL schemas, status/TICKLETRUNK manifests, local DB counts, local GPU/model/dependency state, and emits one finding block per |
| execution_record_writer.py | script | sandbox | reusable_prior | `scripts/legacy/execution_record_writer.py` | Standalone execution record writer for worker outputs. Writes structured execution receipts into lucidota_control.boring_execution_record using the same table Boring Beast workers use. Optional Chrono append records the worker-output event  |
| graph_promotion_evidence_resolver.py | script | sandbox | reusable_prior | `scripts/legacy/graph_promotion_evidence_resolver.py` | Resolve graph-promotion evidence refs before packet creation. Resolved evidence is a precondition for governed graph promotion. This resolver checks files and known DB-backed refs and can append resolution receipts; it does not create promo |
| graph_promotion_materialize.py | script | sandbox | reusable_prior | `scripts/legacy/graph_promotion_materialize.py` | Materialize an evidence-gated graph promotion packet into canonical graph. This is the promotion path, not a direct graph-write shortcut: - verifies a conversation_command UUID exists in the control database - runs graph_promotion_preflight |
| idempotency_duplicate_probe.py | script | sandbox | reusable_prior | `scripts/legacy/idempotency_duplicate_probe.py` | Prove normalized idempotency duplicate suppression in the DBOS queue. |
| intake_custody_job_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/intake_custody_job_bridge.py` | Bridge Rust lucidota-intake UnifiedMetadata JSONL custody records into DBOS jobs. |
| korpus_krampii.py | script | sandbox | reusable_prior | `scripts/legacy/korpus_krampii.py` | UNKNOWN — needs operator label; filename suggests: korpus krampii |
| lucidota_artifact_ingest.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_artifact_ingest.py` | DIOGENES investigative case + artifact workflow. Deterministic local software, not agent sprawl: - Create proper case folders/files. - Hash artifact bytes and lock them into CAS. - Extract metadata/EXIF/OCR/document/video text. - Normalize  |
| lucidota_basic_workflows.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_basic_workflows.py` | Basic DIOGENES/LUCIDOTA workflows. Small, boring, inspectable workflows: audit inventory capabilities and gaps clawd-route route a CLI-style message onto GO graph + DBOS event lane-dispatch materialize deterministic model-lane workflow even |
| lucidota_big_board.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_big_board.py` | LUCIDOTA terminal Big Board v0. Read-only dashboard: docs are truth for bars; Postgres supplies live counters. No Drive access, no writes, no secrets. |
| lucidota_bytewax_mini.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_bytewax_mini.py` | Mini Bytewax live-stream proof over committed workflow events. |
| lucidota_chatdump_timeline.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_chatdump_timeline.py` | Streaming OpenAI/Claude data-export timeline ingester. Built for very large exports (950MB+): ZIP members and top-level JSON arrays are streamed one conversation at a time. No LLM calls. |
| lucidota_clawd_runtime.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_clawd_runtime.py` | Local graph-first runtime for the hard-modded Claw CLI. This is intentionally small: - no Anthropic/OpenAI/xAI keys - route the user message through CKDOG1's baked-in deterministic embedding path - reduce concepts to GO-25 terms - write the |
| lucidota_commdump_timeline.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_commdump_timeline.py` | Universal communications dump ingester. Handles non-exhaustive personal data dumps: Facebook messages, email JSON, SMS/text JSON/XML-ish shapes, JSONL, ZIPs, and directories. It is heuristic and schema-tolerant by design: preserve what can  |
| lucidota_dbos_big_board.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_big_board.py` | DBOS workflow wrapper for Big Board snapshots. Big Board stays read-only by default. This wrapper is the explicit workflow-feed path: collect a local snapshot, persist a compact workflow_event, and return the same compact report for UI/repl |
| lucidota_dbos_brain_child.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_brain_child.py` | Restartable DBOS wrapper for one Krampus brain-sidecar child. This does not replace the live shell watcher yet. It gives each per-file brain sidecar run a deterministic DBOS workflow ID, timeout envelope, retry/resume surface, and LUCIDOTA  |
| lucidota_dbos_dispatch.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_dispatch.py` | DBOS workflow dispatcher with signoff + retry. This is the one-hour test lane: 1. pick a registered workflow, 2. require or auto-create operator signoff, 3. run the registered command as a DBOS workflow step, 4. emit workflow_event rows for |
| lucidota_dbos_drive_map.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_drive_map.py` | DBOS Drive-map workflow, local-records only. No Google Drive connector is touched here. This makes the current safe Drive import/map scaffold a DBOS-owned, replayable workflow. |
| lucidota_dbos_external_draft.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_external_draft.py` | DBOS external-write draft gate. Creates a pending governance gate for any proposed external write without performing the write. Approval can be handled by lucidota_dbos_signoff.py. |
| lucidota_dbos_replay.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_replay.py` | Replay/inspect workflow_event history for DBOS signoff and dispatch tests. |
| lucidota_dbos_signoff.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_signoff.py` | DBOS-backed workflow signoff lane. Purpose: make workflow approval testable in one command surface. - request: create a governance_gate + waiting_user workflow_event - approve/deny: decide the gate + terminal workflow_event - smoke: request |
| lucidota_dbos_smoke.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_smoke.py` | UNKNOWN — needs operator label; filename suggests: lucidota dbos smoke |
| lucidota_dbos_watcher.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dbos_watcher.py` | DBOS scheduled watcher v0. Seeds a tiny local schedule table and records due workflow ticks. It does not daemonize; cron/systemd can call this. The test path proves due detection, locking-by-next-run update, and workflow_event emission. |
| lucidota_dev_preflight.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_dev_preflight.py` | Runtime preflight for LUCIDOTA development discipline. Checks the minimum live gates before coding/worker passes: - TICKLETRUNK manifest exists and parses - status ledger checker passes - direct canonical graph writes are blocked - Boring B |
| lucidota_hard_truth_math.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_hard_truth_math.py` | Hard-math telemetry for LUCIDOTA. Implements four factual-ish signal families: 1) Linguistic Style Matching (LSM) over chat turns. 2) Observed-state Markov/HMM-proxy transitions over brain-map states. 3) Stylometry with deterministic hashed |
| lucidota_hop_pivot.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_hop_pivot.py` | Queue-backed bounded Hop Pivot v1. Uses Survey as the sensor; persists job/node/promotion state; emits workflow events. |
| lucidota_indy_brief.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_indy_brief.py` | Indy_Reads runtime brief + quiet task memory. Reads active machine truth from README/JSON/SQL/Postgres, plus quiet runtime memory tables. Legacy project-brain Markdown is optional archive context only. No Drive/Gmail/Calendar/network reads. |
| lucidota_indy_library_ingest.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_indy_library_ingest.py` | Ingest Indy_READs books into the live Postgres graph. - Scans BOOKS for epub/pdf/mobi/txt/md. - Extracts text locally. - Chunks to <=500 approximate tokens. - Writes book/chunk graph items plus BOOK_HAS_CHUNK edges. - Writes deterministic 3 |
| lucidota_indy_polycareer.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_indy_polycareer.py` | INDY_READs polycareer workflow router + Glow Watch hook. Deterministic/no-LLM helper for the Polycareer Workflow Wizard subproject. It watches local CLAWD/workflow/agent artifacts for: - the right boring workflow mode; and - weird, high-val |
| lucidota_indy_reads_watcher.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_indy_reads_watcher.py` | INDY_READs book watcher. Polls /BOOKS for new or changed books, then runs the simple ingestion lane: extract -> <=500 token chunks -> CKDOG1 embeddings -> GO graph -> LoRA cartridge dataset. No external services. This is the "Mamba watches, |
| lucidota_indy_regression.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_indy_regression.py` | Indy_Reads regression smoke: brief, corrections, auth, queue, corpus. |
| lucidota_litellm_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_litellm_bridge.py` | LiteLLM-compatible model management bridge for LUCIDOTA. This is intentionally a dry inventory/validation command. It does not import litellm, open a model, call a provider API, or run inference. It reports the model/provider configuration  |
| lucidota_model_artifact_readiness.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_model_artifact_readiness.py` | Truthful local model artifact readiness check: registry intent vs actual weight files. |
| lucidota_progress_monitor.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_progress_monitor.py` | LUCIDOTA active-loop progress monitor. Reads process state, ABSURD queue rows, Bytewax/ABCD ledgers, and GO ontology fidelity without mutating worker queues or canonical graph tables. Default mode prints one terminal frame; `loop` repeats f |
| lucidota_river_reflex.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_river_reflex.py` | Tiny River online-learning bridge for committed LUCIDOTA workflow events. This is intentionally small: DBOS/Postgres commits events; this script learns success/failure reflex hints with River and writes bounded scores back to Postgres. It i |
| lucidota_ruthless_gauntlet.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_ruthless_gauntlet.py` | Ruthless 30x20 DIOGENES audit gauntlet generator/executor. |
| lucidota_source_policy_seed.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_source_policy_seed.py` | Seed operator-supervised source policy rows for DIOGENES/LUCIDOTA. |
| lucidota_stream_river_worker.sh | script | sandbox | reusable_prior | `scripts/legacy/lucidota_stream_river_worker.sh` | ROOT="/home/mfspx/LUCIDOTA" |
| lucidota_treelite_router.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_treelite_router.py` | Tiny Treelite routing-hint proof. Builds a minimal Treelite tree with no XGBoost runtime dependency, stores the compiled/lightweight artifact in the ignored vault, runs a prediction, and persists the advisory result. DBOS policy remains the |
| lucidota_wake_bus.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_wake_bus.py` | Local-first Wake Bus using a Postgres outbox. Truth stays in workflow_event and related tables. This worker handles only small wake-up refs and marks outbox rows delivered. It is intentionally boring: durable DB rows, idempotent readers, no |
| lucidota_workflow_registry.py | script | sandbox | reusable_prior | `scripts/legacy/lucidota_workflow_registry.py` | UNKNOWN — needs operator label; filename suggests: lucidota workflow registry |
| marrow_cep_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/marrow_cep_bridge.py` | Bridge a Marrow Loop receipt into a staged conversation_command. |
| marrow_dbos_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/marrow_dbos_bridge.py` | Bridge a Marrow command receipt into a DBOS work-order job. |
| phase05_brain_archaeology_prep.py | script | sandbox | reusable_prior | `scripts/legacy/phase05_brain_archaeology_prep.py` | Phase 0.5 Brain Archaeology scaffold/prep runner. Default mode is dry-run. This script does not perform full corpus ingestion. |
| phase05_workflow_blueprint_generator.py | script | sandbox | reusable_prior | `scripts/legacy/phase05_workflow_blueprint_generator.py` | Generate DBOS implementation-candidate work orders from Phase 0.5 workflow blueprints. |
| post_gate_target_selector.py | script | sandbox | reusable_prior | `scripts/legacy/post_gate_target_selector.py` | Post Mega-Gate target selector. Executable target-selection framework: - Reads TICKLETRUNK and status ledger before decisions. - Scores/selects targets deterministically. - Requires evidence refs for selected/executed states. - Writes DB au |
| production_readiness_eval.py | script | sandbox | reusable_prior | `scripts/legacy/production_readiness_eval.py` | UNKNOWN — needs operator label; filename suggests: production readiness eval |
| runtime_status_ledger_updater.py | script | sandbox | reusable_prior | `scripts/legacy/runtime_status_ledger_updater.py` | Update STATUS_LEDGER from runtime DB facts, not hand-written claims. |
| simplemem_candidate_index.py | script | sandbox | reusable_prior | `scripts/legacy/simplemem_candidate_index.py` | Executable SimpleMem candidate index. Indexes high-recall candidates from existing claim/design-atom tables into lucidota_control.simplemem_candidate. Every row is constrained as NOT_TRUTH and promotion_allowed=false; this is recall substra |
| simplemem_promotion_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/simplemem_promotion_bridge.py` | UNKNOWN — needs operator label; filename suggests: simplemem promotion bridge |
| surface_instruction_compile_dry_run.py | script | sandbox | reusable_prior | `scripts/legacy/surface_instruction_compile_dry_run.py` | Dry-run compiler from generated-surface interaction to conversation instruction. This is not button telemetry. It turns an operator selection/rejection/refinement into a plain-language instruction plus auditable command envelope for later c |
| system_runtime_facts_refresh.py | script | sandbox | reusable_prior | `scripts/legacy/system_runtime_facts_refresh.py` | System-wide runtime facts refresh from live DB/daemon evidence. |
| system_state_desync_detector.py | script | sandbox | reusable_prior | `scripts/legacy/system_state_desync_detector.py` | UNKNOWN — needs operator label; filename suggests: system state desync detector |
| system_telemetry_exporter.py | script | sandbox | reusable_prior | `scripts/legacy/system_telemetry_exporter.py` | UNKNOWN — needs operator label; filename suggests: system telemetry exporter |
| tracer_claim_packet_bridge.py | script | sandbox | reusable_prior | `scripts/legacy/tracer_claim_packet_bridge.py` | Attach TRACER-lite epistemic labels to document claim packets. |
| unified_absurd_ingest_worker.py | script | sandbox | reusable_prior | `scripts/legacy/unified_absurd_ingest_worker.py` | UNKNOWN — needs operator label; filename suggests: unified absurd ingest worker |
| work_order_loader.py | script | sandbox | reusable_prior | `scripts/legacy/work_order_loader.py` | Strict work-order loader/validator for DBOS queue jobs. |
| worker_command_registry.py | script | sandbox | reusable_prior | `scripts/legacy/worker_command_registry.py` | UNKNOWN — needs operator label; filename suggests: worker command registry |
| llxprt_groq_login_bind.sh | script | sandbox | reusable_prior | `scripts/llxprt_groq_login_bind.sh` | LUCIDOTA/LLxprt hard-binding: prefer Groq on login/session initialization. |
| llxprt_project2501.py | script | sandbox | reusable_prior | `scripts/llxprt_project2501.py` | Project 2501 LLxprt/Groq orchestrator setup and launcher. |
| local_deterministic_audit.py | script | sandbox | reusable_prior | `scripts/local_deterministic_audit.py` | Deterministic local audit fallback for signed five-task coverage. |
| local_model_chat_cli.py | script | sandbox | reusable_prior | `scripts/local_model_chat_cli.py` | Receipt-backed local model invocation probe for llama.cpp and Needle lanes. |
| longmemeval_seed_generator.py | script | sandbox | reusable_prior | `scripts/longmemeval_seed_generator.py` | Generate LongMemEval-style eval seeds after timeline/dev-pattern ingestion. |
| lucidota_acceptance.py | script | sandbox | reusable_prior | `scripts/lucidota_acceptance.py` | UNKNOWN — needs operator label; filename suggests: lucidota acceptance |
| lucidota_age_edges.py | script | sandbox | reusable_prior | `scripts/lucidota_age_edges.py` | Write Survey/Hop/CAS provenance edges into Apache AGE. |
| lucidota_algos_smoke.py | script | sandbox | reusable_prior | `scripts/lucidota_algos_smoke.py` | Smoke tests for local algorithm primitives. |
| lucidota_anthropic_llama_proxy.py | script | sandbox | reusable_prior | `scripts/lucidota_anthropic_llama_proxy.py` | Minimal Anthropic Messages API shim over local llama.cpp OpenAI-compatible server. |
| lucidota_auth_report.py | script | sandbox | reusable_prior | `scripts/lucidota_auth_report.py` | Create a redacted local auth surface report from tracked auth inventory only. |
| lucidota_bonsai_ternary_handler.py | script | sandbox | reusable_prior | `scripts/lucidota_bonsai_ternary_handler.py` | LOCAL BONSAI TERNARY runtime wiring. This is intentionally a thin, local-only handler around the PrismML llama.cpp fork because the Bonsai GGUF uses GGML type 42 / Q2_0 ternary weights that the stock llama.cpp build in this checkout rejecte |
| lucidota_brain_ingest.py | script | sandbox | reusable_prior | `scripts/lucidota_brain_ingest.py` | Standalone KrampusBrain DBSTREAM ingest. This is the brain-map sidecar: it reuses KORPUS' deterministic sticker extractors, feeds one document at a time to River DBSTREAM, and appends a JSONL point that can be plotted later. No LLM calls. |
| lucidota_bytewax_mini.py | script | sandbox | reusable_prior | `scripts/lucidota_bytewax_mini.py` | Bytewax live cursor over workflow_event rows; hard-fails on errors. |
| lucidota_cas_gc.py | script | sandbox | reusable_prior | `scripts/lucidota_cas_gc.py` | Report-first CAS GC for LUCIDOTA. Default mode never deletes and never moves bytes. It scans local CAS files, compares them to Postgres references, writes a durable report, and prints counts. Optional quarantine mode moves only orphan candi |
| lucidota_cas_index.py | script | sandbox | reusable_prior | `scripts/lucidota_cas_index.py` | Index and verify the local LUCIDOTA CAS vault. |
| lucidota_cas_journal.py | script | sandbox | reusable_prior | `scripts/lucidota_cas_journal.py` | Append durable local CAS ingest journal records. This is the filesystem half of dual-write recovery: if bytes land but the DB metadata commit dies, the ignored journal gives the reconciler source context. |
| lucidota_chaos_suite.py | script | sandbox | reusable_prior | `scripts/lucidota_chaos_suite.py` | UNKNOWN — needs operator label; filename suggests: lucidota chaos suite |
| lucidota_ci_gate.py | script | sandbox | reusable_prior | `scripts/lucidota_ci_gate.py` | UNKNOWN — needs operator label; filename suggests: lucidota ci gate |
| lucidota_clawd_runtime.py | script | sandbox | reusable_prior | `scripts/lucidota_clawd_runtime.py` | Active LUCIDOTA/Claw local graph runtime shim. Purpose: keep Claw wired to the vetted legacy GO/ABSURD runtime without copying its 417-line body back into active script space. The shim is the production entrypoint; the source implementation |
| lucidota_cli.py | script | sandbox | reusable_prior | `scripts/lucidota_cli.py` | UNKNOWN — needs operator label; filename suggests: lucidota cli |
| lucidota_cockpit.py | script | sandbox | reusable_prior | `scripts/lucidota_cockpit.py` | One-screen LUCIDOTA cockpit: compact bars, counters, Indy queues, governor. |
| lucidota_code_language_scan.py | script | sandbox | reusable_prior | `scripts/lucidota_code_language_scan.py` | Plain-language gate for executable/schema paths. Purpose: keep code, schemas, and harness text technical and boring. Project lore and immutable ontology records are out of scope for this scanner. |
| lucidota_cuda_inventory.sh | script | sandbox | reusable_prior | `scripts/lucidota_cuda_inventory.sh` | echo "LUCIDOTA CUDA Inventory" |
| lucidota_current_system_docs.py | script | sandbox | reusable_prior | `scripts/lucidota_current_system_docs.py` | Refresh the current human/robot operating docs from live repo + Postgres state. |
| lucidota_decision_delta.py | script | sandbox | reusable_prior | `scripts/lucidota_decision_delta.py` | Decision-quality delta analyzer for KORPUS/chat timelines. This does not diagnose a person. It measures text evidence of decision hygiene: evidence before action, planning, delay/de-escalation, support seeking, boundaries, completion/outcom |
| lucidota_deploy_dry_run.py | script | sandbox | reusable_prior | `scripts/lucidota_deploy_dry_run.py` | UNKNOWN — needs operator label; filename suggests: lucidota deploy dry run |
| lucidota_drive_import_manifest.py | script | sandbox | reusable_prior | `scripts/lucidota_drive_import_manifest.py` | Build a Drive import manifest skeleton from tracked local records only. No Google Drive connector, network, or secret material is used. The output is an operator-facing intake checklist: it identifies known source nuclei and records what ev |
| lucidota_drive_manifest.py | script | sandbox | reusable_prior | `scripts/lucidota_drive_manifest.py` | Local Drive/import manifest from existing records only; no Drive API access. |
| lucidota_extractor_registry.py | script | sandbox | reusable_prior | `scripts/lucidota_extractor_registry.py` | Authorized extractor registry: adapters first, browser fallback last. |
| lucidota_gap_workflow_compiler.py | script | sandbox | reusable_prior | `scripts/lucidota_gap_workflow_compiler.py` | Compile launch gaps into executable workflow cards; no DB/graph writes. |
| lucidota_go_ingest.py | script | sandbox | reusable_prior | `scripts/lucidota_go_ingest.py` | GO ingest/promote lane for DIOGENES/LUCIDOTA. Postgres-first. Uses psql so the project does not need a Python PG driver. Env: LUCIDOTA_GO_STORAGE_DSN default: postgresql:///lucidota_storage LUCIDOTA_GO_STATE_DSN default: postgresql:///lucid |
| lucidota_goal_audit.py | script | sandbox | reusable_prior | `scripts/lucidota_goal_audit.py` | Requirement-level audit for the LUCIDOTA RFC/organization goal. This script intentionally audits the user's stated DONE criteria against current files, registry subjects, and verifier outputs. It is not a replacement for the RFCs; it is the |
| lucidota_hitman_loop.sh | script | sandbox | reusable_prior | `scripts/lucidota_hitman_loop.sh` | LOG="${LUCIDOTA_HITMAN_LOOP_LOG:-/tmp/lucidota_hitman_loop.log}" |
| lucidota_indy_brief.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_brief.py` | Active/importable entrypoint for Indy_READs brief until the Rust port lands. |
| lucidota_indy_contract.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_contract.py` | Render Indy_Reads runtime contract from graph/DB first, Markdown if still present. |
| lucidota_indy_corpus.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_corpus.py` | Build a local Indy_Reads persona corpus/distillation from project-brain docs. No Drive, Gmail, Calendar, network, or ambient filesystem search: this reads only the small allow-listed project-brain files that define current Indy_Reads behavi |
| lucidota_indy_library_ingest.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_library_ingest.py` | Active/importable entrypoint for Indy_READs library ingest until Rust replaces it. |
| lucidota_indy_lora_train.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_lora_train.py` | Train an INDY_READs LoRA cartridge when a local HF base model is available. The current resident DeepSeek file is GGUF for inference. PEFT training needs a Hugging Face model directory/name, supplied as --base-model or LUCIDOTA_LORA_BASE_MO |
| lucidota_indy_polycareer.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_polycareer.py` | Active entrypoint for INDY_READs polycareer routing and Glow Watch. |
| lucidota_indy_reads_watcher.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_reads_watcher.py` | Active/importable entrypoint for the Indy_READs book watcher. |
| lucidota_indy_regression.py | script | sandbox | reusable_prior | `scripts/lucidota_indy_regression.py` | Active/importable entrypoint for Indy_READs regression smoke. |
| lucidota_ingest_watchdog.py | script | sandbox | reusable_prior | `scripts/lucidota_ingest_watchdog.py` | LUCIDOTA ingestion watchdog / observation dashboard. Cron-safe, conservative maintenance loop: - observe live KRAMPUS/KORPUS/brain sidecar state; - write JSON/Markdown/HTML observation dashboard; - apply only narrow safe remediation: termin |
| lucidota_ingest_watchdog_cron.sh | script | sandbox | reusable_prior | `scripts/lucidota_ingest_watchdog_cron.sh` | cd /home/mfspx/LUCIDOTA |
| lucidota_kernel_api_smoke.py | script | sandbox | reusable_prior | `scripts/lucidota_kernel_api_smoke.py` | CKDOG1 full gRPC API smoke: exercises every current KernelService RPC. |
| lucidota_markdown_ingest_archive.py | script | sandbox | reusable_prior | `scripts/lucidota_markdown_ingest_archive.py` | Ingest legacy Markdown breadcrumbs into Postgres graph and remove from active tree. Default is dry-run. Use --execute to move non-README Markdown to 03_VAULT/ingested_markdown. |
| lucidota_mega_gate.py | script | sandbox | reusable_prior | `scripts/lucidota_mega_gate.py` | Full LUCIDOTA mega-gate. Repairs v2: 1. compile critical gate scripts before running the gate; 2. regenerate TICKLETRUNK before checking it; 3. parse child JSON reports instead of trusting rc/stdout alone; 4. enforce cross-system invariants |
| lucidota_model_governor.py | script | sandbox | reusable_prior | `scripts/lucidota_model_governor.py` | LUCIDOTA model VRAM/loadout governor. Advisory only: observes GPU memory, estimates resident slot VRAM, writes the load/defer/reject decision to Postgres, and never imports or loads model weights. |
| lucidota_model_governor_smoke.py | script | sandbox | reusable_prior | `scripts/lucidota_model_governor_smoke.py` | Regression smoke for the advisory LUCIDOTA model governor. Uses synthetic slots/GPU observations only; it must not load model weights. |
| lucidota_model_registry.py | script | sandbox | reusable_prior | `scripts/lucidota_model_registry.py` | Print the current LUCIDOTA model runtime registry. |
| lucidota_model_turbine_overseer.py | script | sandbox | reusable_prior | `scripts/lucidota_model_turbine_overseer.py` | Bounded local-model overseer: health, RAM/VRAM guards, DB snapshot, JSON tasks. |
| lucidota_mps_start.sh | script | sandbox | reusable_prior | `scripts/lucidota_mps_start.sh` | PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/lucidota-mps}" |
| lucidota_mps_stop.sh | script | sandbox | reusable_prior | `scripts/lucidota_mps_stop.sh` | PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/lucidota-mps}" |
| lucidota_needle_worker.py | script | sandbox | reusable_prior | `scripts/lucidota_needle_worker.py` | Tiny HTTP worker for one resident Needle tool-call router instance. |
| lucidota_observation_live_loop.sh | script | sandbox | reusable_prior | `scripts/lucidota_observation_live_loop.sh` | Live updater for the canonical LUCIDOTA observation dashboard. |
| lucidota_observation_live_loop_ensure.sh | script | sandbox | reusable_prior | `scripts/lucidota_observation_live_loop_ensure.sh` | Ensure the canonical observation dashboard live loop stays up. |
| lucidota_omni_front_sprint_orchestrator.py | script | sandbox | reusable_prior | `scripts/lucidota_omni_front_sprint_orchestrator.py` | LUCIDOTA omni-front sprint orchestrator. Critical assumptions: - temporal_claim is archive/append-only; this daemon never updates/deletes it. - weak mtime is custody/runtime evidence only; graph promotion uses RFC-CHRONO-001 gate. - no IP/p |
| lucidota_operator_demo.py | script | sandbox | reusable_prior | `scripts/lucidota_operator_demo.py` | End-to-end operator demo script: cockpit, Survey file, model governor. |
| lucidota_ouroboros_loop.py | script | sandbox | reusable_prior | `scripts/lucidota_ouroboros_loop.py` | Bounded Ouroboros loop runner. Inspects real repo targets, classifies them, runs the smallest safe validation available for each target, and writes per-cycle JSONL plus a summary receipt. It never deletes or mutates target artifacts; runtim |
| lucidota_pipeline.py | script | sandbox | reusable_prior | `scripts/lucidota_pipeline.py` | UNKNOWN — needs operator label; filename suggests: lucidota pipeline |
| lucidota_pipeline_synthesis.py | script | sandbox | reusable_prior | `scripts/lucidota_pipeline_synthesis.py` | Executable LUCIDOTA pipeline synthesis map. This is the "map the force multipliers" layer: it reads TICKLETRUNK plus the already-existing pipeline contracts/job planner and emits an inspectable command map, gap list, and receipt. It perform |
| lucidota_production_signoff.py | script | sandbox | reusable_prior | `scripts/lucidota_production_signoff.py` | UNKNOWN — needs operator label; filename suggests: lucidota production signoff |
| lucidota_progress.py | script | sandbox | reusable_prior | `scripts/lucidota_progress.py` | Print LUCIDOTA build progress. Legacy BUILD_PLAN_AUDIT.md may be archived. Active fallback is the status ledger; missing legacy files are not a crash condition. |
| lucidota_record_runtime_inventory.py | script | sandbox | reusable_prior | `scripts/lucidota_record_runtime_inventory.py` | Capture current runtime inventory into lucidota_control.model_runtime_inventory. |
| lucidota_regression_dashboard.py | script | sandbox | reusable_prior | `scripts/lucidota_regression_dashboard.py` | Regression dashboard: compact status of core focused checks. |
| lucidota_release_checklist.py | script | sandbox | reusable_prior | `scripts/lucidota_release_checklist.py` | Release checklist gate: local verification signals only. |
| lucidota_release_gate.py | script | sandbox | reusable_prior | `scripts/lucidota_release_gate.py` | UNKNOWN — needs operator label; filename suggests: lucidota release gate |
| lucidota_release_manifest.py | script | sandbox | reusable_prior | `scripts/lucidota_release_manifest.py` | UNKNOWN — needs operator label; filename suggests: lucidota release manifest |
| lucidota_river_reflex.py | script | sandbox | reusable_prior | `scripts/lucidota_river_reflex.py` | River incremental online-learning tick over new workflow events only. |
| lucidota_runtime_smoke.py | script | sandbox | reusable_prior | `scripts/lucidota_runtime_smoke.py` | LUCIDOTA runtime smoke. This intentionally checks plumbing, not performance. Benchmarks come after the stack imports cleanly and the control-plane schema exists. |
| lucidota_safe_ops_env.sh | script | sandbox | reusable_prior | `scripts/lucidota_safe_ops_env.sh` | LUCIDOTA laptop safe-ops defaults. |
| lucidota_security_quarantine_gate.py | script | sandbox | reusable_prior | `scripts/lucidota_security_quarantine_gate.py` | LUCIDOTA security quarantine gate. Scans configured local roots for secret-bearing file classes. Does not print secret values, delete files, mutate DB, or call external services. |
| lucidota_start_anthropic_proxy.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_anthropic_proxy.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_bonsai_ternary_llama.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_bonsai_ternary_llama.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_deepseek_llama.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_deepseek_llama.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_indy_polycareer_watch.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_indy_polycareer_watch.sh` | ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}" |
| lucidota_start_indy_reads_watcher.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_indy_reads_watcher.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_mamba_gpu_partial.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_mamba_gpu_partial.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_mamba_llama.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_mamba_llama.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_needle_swarm.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_needle_swarm.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_start_strict_model_stack.sh | script | sandbox | reusable_prior | `scripts/lucidota_start_strict_model_stack.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_status_ledger.py | script | sandbox | reusable_prior | `scripts/lucidota_status_ledger.py` | Canonical LUCIDOTA status ledger maintainer. Hard law: If it exists, it is listed. If it ran, it has evidence. If it is blocked, the blocker is named. If it changed, the ledger changed. |
| lucidota_stream_river_worker.sh | script | sandbox | reusable_prior | `scripts/lucidota_stream_river_worker.sh` | ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}" |
| lucidota_strict_model_stack_admission.py | script | sandbox | reusable_prior | `scripts/lucidota_strict_model_stack_admission.py` | Admission gate for the strict local model stack. This does not start models. It proves the local DIOGENES/CKDOG1 hot path is under the RAM ceiling, checks the display/offload policy keeps graphics off the NVIDIA model card, verifies model/l |
| lucidota_surface_emit_command.py | script | sandbox | reusable_prior | `scripts/lucidota_surface_emit_command.py` | UNKNOWN — needs operator label; filename suggests: lucidota surface emit command |
| lucidota_surface_promote.py | script | sandbox | reusable_prior | `scripts/lucidota_surface_promote.py` | Promote generated surfaces with immutable lineage. Dry-run is default. Execute requires --execute and --confirm-promote; execution copies an artifact to promoted/ and appends lineage metadata only. No canonical graph writes. |
| lucidota_swarm_dashboard.py | script | sandbox | reusable_prior | `scripts/lucidota_swarm_dashboard.py` | LUCIDOTA terminal dashboard UI/UX payload. Scope / safety contract ----------------------- - Read-only frontend: this module only reads JSON receipts under 05_OUTPUTS. - No network calls, no database calls, no canonical graph writes, no des |
| lucidota_swarm_router.py | script | sandbox | reusable_prior | `scripts/lucidota_swarm_router.py` | Deterministic sovereign swarm routing payload for LUCIDOTA. This module is intentionally inert: it makes no provider calls, reads no secrets, opens no network sockets, shells out nowhere, and emits no telemetry. It is a ready-to-review rout |
| lucidota_synthesis_pass.py | script | sandbox | reusable_prior | `scripts/lucidota_synthesis_pass.py` | Guardrail-preserving autonomous synthesis pass wrapper. This does not replace the existing suite. It wraps native checks and the existing acceptance harness into one bounded, receipt-backed execution pass. |
| lucidota_synthesis_pass.sh | script | sandbox | reusable_prior | `scripts/lucidota_synthesis_pass.sh` | ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" |
| lucidota_usecase_proof.py | script | sandbox | reusable_prior | `scripts/lucidota_usecase_proof.py` | Bounded proof harness for case, campaign, code, and hypertimeline use-cases. |
| lucidota_validator_noise_stress.py | script | sandbox | reusable_prior | `scripts/lucidota_validator_noise_stress.py` | EQ-001..EQ-100 synthetic low-quality stress inventory. The imported math.zip validators are test-first material. This harness verifies that every EQ id is represented in the recovered tests and runs a deterministic synthetic low-quality cor |
| lucidota_wake_bus_audit.py | script | sandbox | reusable_prior | `scripts/lucidota_wake_bus_audit.py` | Static + DB smoke audit for Wake Bus batch delivery shape. Checks the worker still claims bounded rows with one CTE, row locks, and SKIP LOCKED instead of drifting into per-row/O(N) delivery loops. |
| lucidota_wiki_query.py | script | sandbox | reusable_prior | `scripts/lucidota_wiki_query.py` | File-backed VIBESCONTROL / Indy_Reads wiki query v0. |
| marrow_loop_apply.py | script | sandbox | reusable_prior | `scripts/marrow_loop_apply.py` | Apply a Marrow Loop command receipt to the append-only local marrow state. |
| marrow_loop_render_surface.py | script | sandbox | reusable_prior | `scripts/marrow_loop_render_surface.py` | Render Marrow Loop status surface from append-only marrow_state.md. |
| marrow_state_append_only_verify.py | script | sandbox | reusable_prior | `scripts/marrow_state_append_only_verify.py` | Verify Marrow state is append-only using a local hash-chain receipt. |
| master_eye_review_worker.py | script | sandbox | reusable_prior | `scripts/master_eye_review_worker.py` | Deterministic Master Eye runtime reviewer for Phase 0.5 artifacts. |
| matrix_stream_executor.py | script | sandbox | reusable_prior | `scripts/matrix_stream_executor.py` | Bounded LUCIDOTA matrix executor. Streams security quarantine manifests and RGAUNTLET JSONL without loading large inputs. Writes dead-list/unblock receipts and bounded gauntlet proof batches. |
| matrix_trace_checker.py | script | sandbox | reusable_prior | `scripts/matrix_trace_checker.py` | UNKNOWN — needs operator label; filename suggests: matrix trace checker |
| media_metadata.py | script | sandbox | reusable_prior | `scripts/media_metadata.py` | UNKNOWN — needs operator label; filename suggests: media metadata |
| mega_gate_fault_injector.py | script | sandbox | reusable_prior | `scripts/mega_gate_fault_injector.py` | UNKNOWN — needs operator label; filename suggests: mega gate fault injector |
| mega_gate_metrics_validator.py | script | sandbox | reusable_prior | `scripts/mega_gate_metrics_validator.py` | Validate Mega-Gate v2 report metrics and repair coverage. |
| mega_gate_regression_loop.py | script | sandbox | reusable_prior | `scripts/mega_gate_regression_loop.py` | Run Mega-Gate repeatedly and validate each report. |
| memory_candidates.py | script | sandbox | reusable_prior | `scripts/memory_candidates.py` | UNKNOWN — needs operator label; filename suggests: memory candidates |
| missing_evidence_checklist.py | script | sandbox | reusable_prior | `scripts/missing_evidence_checklist.py` | UNKNOWN — needs operator label; filename suggests: missing evidence checklist |
| model_answer_diff_dry_run.py | script | sandbox | reusable_prior | `scripts/model_answer_diff_dry_run.py` | Cross-model behavioral answer diff; no external APIs Research-only dry-run scaffold. No DB writes. No graph writes. |
| model_audit_absurd_adapter.py | script | sandbox | reusable_prior | `scripts/model_audit_absurd_adapter.py` | Convert model invocation audit receipts into ABSURD typed rows. |
| model_audit_db_adapter.py | script | sandbox | reusable_prior | `scripts/model_audit_db_adapter.py` | Convert model invocation audit receipts into Abductive DB OS typed rows. |
| model_generation_event_bridge.py | script | sandbox | reusable_prior | `scripts/model_generation_event_bridge.py` | Stage model generation receipts into the targeted async PG/ABSURD event lane. |
| model_inventory.py | script | sandbox | reusable_prior | `scripts/model_inventory.py` | UNKNOWN — needs operator label; filename suggests: model inventory |
| model_invocation_audit.py | script | sandbox | reusable_prior | `scripts/model_invocation_audit.py` | Audit every cloud/local model invocation receipt and five-task model-audit coverage. |
| model_invocation_trace.py | script | sandbox | reusable_prior | `scripts/model_invocation_trace.py` | Normalized model-generation telemetry for LUCIDOTA model runners. |
| model_output_contract_audit.py | script | sandbox | reusable_prior | `scripts/model_output_contract_audit.py` | Normalize and audit model receipts against the GO-25 worker output contract. |
| model_runner_cli.py | script | sandbox | reusable_prior | `scripts/model_runner_cli.py` | CLI front door for local model-runner config validation and STUB receipts. |
| model_runner_config.py | script | sandbox | reusable_prior | `scripts/model_runner_config.py` | UNKNOWN — needs operator label; filename suggests: model runner config |
| model_runner_stub.py | script | sandbox | reusable_prior | `scripts/model_runner_stub.py` | UNKNOWN — needs operator label; filename suggests: model runner stub |
| mudcrab_merchant_tui.py | script | sandbox | reusable_prior | `scripts/mudcrab_merchant_tui.py` | Mudcrab Merchant: small local TUI launcher for the Clawd desktop. No policy ceremony here: this is a convenience switchboard for the operator's local environment. It launches Claw, Firefox, maps, and file explorers using the best command av |
| mutation_safety_oracle.py | script | sandbox | reusable_prior | `scripts/mutation_safety_oracle.py` | UNKNOWN — needs operator label; filename suggests: mutation safety oracle |
| ncnn_edge_runtime_probe.py | script | sandbox | reusable_prior | `scripts/ncnn_edge_runtime_probe.py` | Read-only ncnn edge-runtime capability probe for LUCIDOTA. |
| next_action_compiler.py | script | sandbox | reusable_prior | `scripts/next_action_compiler.py` | UNKNOWN — needs operator label; filename suggests: next action compiler |
| no_delete_guard.py | script | sandbox | reusable_prior | `scripts/no_delete_guard.py` | Forward-only guard for tracked deletes: preserve or receipt before erasing. |
| ocr_backlog.py | script | sandbox | reusable_prior | `scripts/ocr_backlog.py` | UNKNOWN — needs operator label; filename suggests: ocr backlog |
| ocr_document_router.py | script | sandbox | reusable_prior | `scripts/ocr_document_router.py` | Route files into OCR/document parsing states without pretending OCR ran. |
| ocr_routing.py | script | sandbox | reusable_prior | `scripts/ocr_routing.py` | UNKNOWN — needs operator label; filename suggests: ocr routing |
| ontology_nudge_inventory_dry_run.py | script | sandbox | reusable_prior | `scripts/ontology_nudge_inventory_dry_run.py` | WOOOAAHH NELLIE THIS IS A BRAINSTORM. Research-only dry-run scaffold. No DB writes. No graph writes. |
| ontology_staging.py | script | sandbox | reusable_prior | `scripts/ontology_staging.py` | UNKNOWN — needs operator label; filename suggests: ontology staging |
| ontology_staging_contract.py | script | sandbox | reusable_prior | `scripts/ontology_staging_contract.py` | Create ontology staging candidates; never promotes truth directly. |
| openai_codex_prompt_guide_ingest.py | script | sandbox | reusable_prior | `scripts/openai_codex_prompt_guide_ingest.py` | Ingest OpenAI's Codex prompting guide as an INDY_READs prompt-policy source. |
| operator_command_router.py | script | sandbox | reusable_prior | `scripts/operator_command_router.py` | UNKNOWN — needs operator label; filename suggests: operator command router |
| operator_decisions.py | script | sandbox | reusable_prior | `scripts/operator_decisions.py` | UNKNOWN — needs operator label; filename suggests: operator decisions |
| operator_ontology_fidelity_guard.py | script | sandbox | reusable_prior | `scripts/operator_ontology_fidelity_guard.py` | Runtime Operator ontology fidelity guard for extraction outputs. Checks actual extraction/report outputs for forbidden softening or renamed labels before they can be used by archaeology, workflow, or graph promotion paths. |
| oracle_scope_enforcer.py | script | sandbox | reusable_prior | `scripts/oracle_scope_enforcer.py` | Independent filesystem scope oracle for implementation slices. |
| patch_runner.py | script | sandbox | reusable_prior | `scripts/patch_runner.py` | UNKNOWN — needs operator label; filename suggests: patch runner |
| path_redaction.py | script | sandbox | reusable_prior | `scripts/path_redaction.py` | UNKNOWN — needs operator label; filename suggests: path redaction |
| percyphon_kernel_bridge.py | script | sandbox | reusable_prior | `scripts/percyphon_kernel_bridge.py` | Route a PercyphonAI procedural scaffold through the Diogenes control-packet gate. |
| phase05_allowlisted_ingest.py | script | sandbox | reusable_prior | `scripts/phase05_allowlisted_ingest.py` | UNKNOWN — needs operator label; filename suggests: phase05 allowlisted ingest |
| phase05_contradiction_ledger_worker.py | script | sandbox | reusable_prior | `scripts/phase05_contradiction_ledger_worker.py` | Extract Phase 0.5 contradiction/boundary records from design atoms and operator laws. |
| phase05_design_atom_extractor.py | script | sandbox | reusable_prior | `scripts/phase05_design_atom_extractor.py` | Deterministic Phase 0.5 design atom extractor. This converts allowlisted custody artifacts into design_atom candidates using operator-doctrine/design-rule patterns. It does not call LLMs and does not mutate graph tables. |
| phase05_workflow_blueprint_synthesizer.py | script | sandbox | reusable_prior | `scripts/phase05_workflow_blueprint_synthesizer.py` | Synthesize workflow_blueprint rows from existing Phase 0.5 design_atom rows. This is executable synthesis, not prose. It reads already-captured design_atom claims, clusters them into concrete ABSURD-target workflow candidates, and writes wo |
| pipeline_contracts.py | script | sandbox | reusable_prior | `scripts/pipeline_contracts.py` | UNKNOWN — needs operator label; filename suggests: pipeline contracts |
| pipeline_run_store.py | script | sandbox | reusable_prior | `scripts/pipeline_run_store.py` | UNKNOWN — needs operator label; filename suggests: pipeline run store |
| product_intake.py | script | sandbox | reusable_prior | `scripts/product_intake.py` | UNKNOWN — needs operator label; filename suggests: product intake |
| product_parse_pipeline.py | script | sandbox | reusable_prior | `scripts/product_parse_pipeline.py` | UNKNOWN — needs operator label; filename suggests: product parse pipeline |
| product_proof_harness.py | script | sandbox | reusable_prior | `scripts/product_proof_harness.py` | UNKNOWN — needs operator label; filename suggests: product proof harness |
| project2501_admin_prompt.py | script | sandbox | reusable_prior | `scripts/project2501_admin_prompt.py` | Project 2501 admin-prompt compiler/enforcer for LUCIDOTA model calls. |
| project2501_board_move.py | script | sandbox | reusable_prior | `scripts/project2501_board_move.py` | Project 2501 board-move pipeline: EventEnvelope -> route -> WorkReceipt -> River row. |
| project2501_board_worker.py | script | sandbox | reusable_prior | `scripts/project2501_board_worker.py` | Project 2501 bounded board worker: claim one slow/audit work_order, emit receipt/River/watch_metric. |
| project2501_bytewax_board_stream.py | script | sandbox | reusable_prior | `scripts/project2501_bytewax_board_stream.py` | Bytewax-compatible stream over Project 2501 board-move tables. |
| project2501_bytewax_board_stream_service.py | script | sandbox | reusable_prior | `scripts/project2501_bytewax_board_stream_service.py` | Install/verify durable user service for the Project2501 Bytewax board stream. |
| project2501_script_audit_worker.py | script | sandbox | reusable_prior | `scripts/project2501_script_audit_worker.py` | Project 2501 audit-lane script classifier: script -> manifest/corpse evidence -> receipt. |
| project2501_watch_server.py | script | sandbox | reusable_prior | `scripts/project2501_watch_server.py` | Project 2501 realtime watch surface: batteries, task stream, subway map. |
| project2501_workshare_payload.py | script | sandbox | reusable_prior | `scripts/project2501_workshare_payload.py` | Emit Project 2501 GO-25 workshare JzLOADS and receipt. |
| project_brain_doc_authority_check.py | script | sandbox | reusable_prior | `scripts/project_brain_doc_authority_check.py` | Check 00_PROJECT_BRAIN document/file authority shape. This is not a metaphysical truth engine. It verifies the concrete filesystem claim the operator challenged: top-level doc sprawl is counted, active specs live in a named folder, and file |
| proof_kernel.py | script | sandbox | reusable_prior | `scripts/proof_kernel.py` | Proof Kernel v1: byte custody without truth promotion or graph mutation. |
| quality_work_order_compiler.py | script | sandbox | reusable_prior | `scripts/quality_work_order_compiler.py` | Compile subsystem-quality repair rows into bounded ABSURD work orders. |
| quarantine_review.py | script | sandbox | reusable_prior | `scripts/quarantine_review.py` | UNKNOWN — needs operator label; filename suggests: quarantine review |
| real_stress_test.py | script | sandbox | reusable_prior | `scripts/real_stress_test.py` | LUCIDOTA Real Throughput & Analytical Accuracy Stress Tester Recalibrates the biological safety gates to measure true hardware limits. Safeguards: Prevents false-positive Thanatosis data dropouts; logs raw performance. |
| receipt_exporter.py | script | sandbox | reusable_prior | `scripts/receipt_exporter.py` | UNKNOWN — needs operator label; filename suggests: receipt exporter |
| receipt_writer.py | script | sandbox | reusable_prior | `scripts/receipt_writer.py` | UNKNOWN — needs operator label; filename suggests: receipt writer |
| recovery_matrix.py | script | sandbox | reusable_prior | `scripts/recovery_matrix.py` | UNKNOWN — needs operator label; filename suggests: recovery matrix |
| report_retention_index.py | script | sandbox | reusable_prior | `scripts/report_retention_index.py` | UNKNOWN — needs operator label; filename suggests: report retention index |
| rete_bandit_gate_cli.py | script | sandbox | reusable_prior | `scripts/rete_bandit_gate_cli.py` | CLI wrapper for the existing RETE/bandit fastlane-slowlane router. This copies no algorithm code and mutates no sovereign ALGOS artifact. It makes ALGOS.rete_bandit_gate runnable from the operator shell and writes a receipt. |
| rfc_claim_discipline_check.py | script | sandbox | reusable_prior | `scripts/rfc_claim_discipline_check.py` | Enforce RFC claim discipline from the operator's #1/#2 corrections. ABBA3^5 is treated here as a local operator audit instruction, not as an established external field term. The checker is intentionally structural: it forces each subject RF |
| rfc_program_check.py | script | sandbox | reusable_prior | `scripts/rfc_program_check.py` | Verify LUCIDOTA RFC program structure and source evidence. |
| run_dev_order_methodology_checks.py | script | sandbox | reusable_prior | `scripts/run_dev_order_methodology_checks.py` | UNKNOWN — needs operator label; filename suggests: run dev order methodology checks |
| run_golden_path_hardening_checks.py | script | sandbox | reusable_prior | `scripts/run_golden_path_hardening_checks.py` | UNKNOWN — needs operator label; filename suggests: run golden path hardening checks |
| run_instruction_hygiene.py | script | sandbox | reusable_prior | `scripts/run_instruction_hygiene.py` | UNKNOWN — needs operator label; filename suggests: run instruction hygiene |
| run_lucidota_intake_watch.sh | script | sandbox | reusable_prior | `scripts/run_lucidota_intake_watch.sh` | ROOT_DIR="${LUCIDOTA_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}" |
| safe_stress_test.py | script | sandbox | reusable_prior | `scripts/safe_stress_test.py` | LUCIDOTA Live Throughput Safety Stress Test Read-only metric suite: - checks resident worker liveness, - measures local loop throughput, - samples GPU thermal / VRAM state, - checks Postgres queue posture, - prints a compact operator-facing |
| same_lineage_validator.py | script | sandbox | reusable_prior | `scripts/same_lineage_validator.py` | Validate one receipt lineage by hash-linked parent/child receipts. Accepted primary input: lucidota.lineage.receipt_bundle.v1. A bundle is a compact, deterministic proof that an operator instruction moved through the command, queue, worker, |
| script_bucket_manifest.py | script | sandbox | reusable_prior | `scripts/script_bucket_manifest.py` | Classify LUCIDOTA scripts without moving or deleting anything. Buckets: - KEEP_ACTIVE: entrypoint/check/workflow script referenced by tests, schemas, CI, or docs. - KEEP_LIBRARY: imported helper/library script with no strong standalone-entr |
| script_survival_audit.py | script | sandbox | reusable_prior | `scripts/script_survival_audit.py` | Append-only script survival audit manifest writer. This tool records ACTIVE_KEEP, ACTIVE_REPAIR, LEGACY_CORPSE, and UNKNOWN_HOLD script audit verdicts without deleting or moving scripts. Corpse verdicts also feed the Krampuschewing script-c |
| script_survival_coverage.py | script | sandbox | reusable_prior | `scripts/script_survival_coverage.py` | Report script survival-audit coverage against TICKLETRUNK script entries. |
| semantic_handle_generator.py | script | sandbox | reusable_prior | `scripts/semantic_handle_generator.py` | Generate operator-owned fungible semantic handles for tools/workflows/artifacts. |
| semantic_handles_dry_run.py | script | sandbox | reusable_prior | `scripts/semantic_handles_dry_run.py` | Fungible semantic IDs; operator namespace sovereignty Research-only dry-run scaffold. No DB writes. No graph writes. |
| signal_aggregator.py | script | sandbox | reusable_prior | `scripts/signal_aggregator.py` | Generic deterministic signal aggregation CLI. Purpose: - Combine overlapping alarms/sensor outputs/model scores into one advisory stance. - Stay domain-agnostic: Ahoy, ops, graph triage, monitoring, routing, etc. - Use only Python stdlib so |
| simplemem_candidate_index.py | script | sandbox | reusable_prior | `scripts/simplemem_candidate_index.py` | Executable SimpleMem candidate index. Indexes high-recall candidates from existing claim/design-atom tables into lucidota_control.simplemem_candidate. Every row is constrained as NOT_TRUTH and promotion_allowed=false; this is recall substra |
| simplemem_recall_expansion.py | script | sandbox | reusable_prior | `scripts/simplemem_recall_expansion.py` | Expand SimpleMem recall over claim packets/design atoms across pass kinds. |
| slop_audit_law.py | script | sandbox | reusable_prior | `scripts/slop_audit_law.py` | Blueprint-First / PocketFlow simplicity audit. Deterministic hygiene check: workflow path should live in code blueprints; models are bounded tools. PocketFlow's ~100 line core is the local simplicity yardstick. |
| slop_jsonl_to_parquet.py | script | sandbox | reusable_prior | `scripts/slop_jsonl_to_parquet.py` | Transcode large JSONL artifacts to machine-readable compact forms. - Input: repo-wide *.jsonl (>= threshold, excluding cleanup outputs) - Output: - 05_OUTPUTS/slop_cleanup/jsonl_to_parquet/.../<source>.parquet - 05_OUTPUTS/slop_cleanup/json |
| snapshot_slicer.py | script | sandbox | reusable_prior | `scripts/snapshot_slicer.py` | Line-safe Chrono snapshot slicer with per-part header lineage. |
| source_bundle.py | script | sandbox | reusable_prior | `scripts/source_bundle.py` | UNKNOWN — needs operator label; filename suggests: source bundle |
| source_quote_extractor.py | script | sandbox | reusable_prior | `scripts/source_quote_extractor.py` | UNKNOWN — needs operator label; filename suggests: source quote extractor |
| spine_authority_checker.py | script | sandbox | reusable_prior | `scripts/spine_authority_checker.py` | UNKNOWN — needs operator label; filename suggests: spine authority checker |
| spine_common.py | script | sandbox | reusable_prior | `scripts/spine_common.py` | UNKNOWN — needs operator label; filename suggests: spine common |
| spine_document_parse_worker.py | script | sandbox | reusable_prior | `scripts/spine_document_parse_worker.py` | ABSURD wrapper for document parse ingestion. Queues document parser jobs, consumes one job safely, calls the production local parser ingestion path, and records ABSURD queue events. Parser output is NOT truth and canonical graph writes are  |
| spine_job_adapter.py | script | sandbox | reusable_prior | `scripts/spine_job_adapter.py` | UNKNOWN — needs operator label; filename suggests: spine job adapter |
| spine_kernel_authorization.py | script | sandbox | reusable_prior | `scripts/spine_kernel_authorization.py` | ABSURD/Postgres kernel-authorization enforcement helpers. This is not a status/report layer. Workers import this before executing jobs whose payload can cause downstream custody, graph, parser, OCR, or ledger effects. |
| spine_krampus_worker.py | script | sandbox | reusable_prior | `scripts/spine_krampus_worker.py` | ABSURD queue-spine wrapper for KRAMPUSCHEWING/KORPUS. This is an observation/health wrapper. It writes ABSURD queue/job/workflow receipts only. It never ingests drops, moves files, deletes files, mutates temporal claims, or mutates KORPUS c |
| spine_queue_soak_test.py | script | sandbox | reusable_prior | `scripts/spine_queue_soak_test.py` | ABSURD queue spine soak: many jobs, retries, duplicate suppression. |
| spine_surface_cep_worker.py | script | sandbox | reusable_prior | `scripts/spine_surface_cep_worker.py` | ABSURD wrapper for Surface/CEP fan-in. Writes queue/workflow receipts and optionally stages conversation_command rows. No canonical graph mutation. |
| status_ledger_evidence_gate.py | script | sandbox | reusable_prior | `scripts/status_ledger_evidence_gate.py` | UNKNOWN — needs operator label; filename suggests: status ledger evidence gate |
| status_ledger_fault_injector.py | script | sandbox | reusable_prior | `scripts/status_ledger_fault_injector.py` | UNKNOWN — needs operator label; filename suggests: status ledger fault injector |
| sticker_feature_extractor_v1.py | script | sandbox | reusable_prior | `scripts/sticker_feature_extractor_v1.py` | Sticker feature vector extractor v1 over allowlisted parsed text custody. |
| subsystem_abcd_sweeper.py | script | sandbox | reusable_prior | `scripts/subsystem_abcd_sweeper.py` | Bounded ABCD sweeper for RGAUNTLET subsystems. Streams the gauntlet, runs one bounded ABCD proof bundle per subsystem, emits work-order proof packets for passing subsystems, and dead-letters failures. It never mutates KRAMPUSCHEWING/CHROMAD |
| subsystem_quality_audit.py | script | sandbox | reusable_prior | `scripts/subsystem_quality_audit.py` | Legal-authority-system quality sweep: verdict every subsystem/script. |
| surface_html_command_extractor.py | script | sandbox | reusable_prior | `scripts/surface_html_command_extractor.py` | Extract command-envelope JSON candidates from generated surface HTML. |
| surface_instruction_compile_dry_run.py | script | sandbox | reusable_prior | `scripts/surface_instruction_compile_dry_run.py` | Dry-run compiler from generated-surface interaction to conversation instruction. This is not button telemetry. It turns an operator selection/rejection/refinement into a plain-language instruction plus auditable command envelope for later c |
| surface_lineage_validator.py | script | sandbox | reusable_prior | `scripts/surface_lineage_validator.py` | Validate generated surface lineage records and artifacts. |
| surface_reuse_registry_validator.py | script | sandbox | reusable_prior | `scripts/surface_reuse_registry_validator.py` | Validate generated surface sidecars are discoverable through TICKLETRUNK. The registry is TICKLETRUNK. This checker keeps generated/reusable surfaces findable without mutating canonical graph or surface state. |
| surface_sidecar_validator.py | script | sandbox | reusable_prior | `scripts/surface_sidecar_validator.py` | Validate generated-surface sidecars and HTML affordances. Enforces the LUCIDOTA surface law: generated surfaces compile interactions into plain-language command envelopes and must not directly mutate DB/API/canonical state. |
| surfaces_marrow_cep_loop_proof.py | script | sandbox | reusable_prior | `scripts/surfaces_marrow_cep_loop_proof.py` | Integrated Surfaces/Marrow/CEP command loop proof. |
| swarm_usage_ledger.py | script | sandbox | reusable_prior | `scripts/swarm_usage_ledger.py` | Aggregate Groq/local/main-agent token receipts into the requested swarm ledger. |
| system_archaeology_evidence_audit.py | script | sandbox | reusable_prior | `scripts/system_archaeology_evidence_audit.py` | System archaeology evidence audit: custody -> claim -> atom -> review coverage. |
| system_graph_safety_audit.py | script | sandbox | reusable_prior | `scripts/system_graph_safety_audit.py` | System-wide graph safety audit: orphan/direct-write/materialization/journal checks. |
| system_runtime_facts_refresh.py | script | sandbox | reusable_prior | `scripts/system_runtime_facts_refresh.py` | System-wide runtime facts refresh from live DB/daemon evidence. |
| system_temporal_audit.py | script | sandbox | reusable_prior | `scripts/system_temporal_audit.py` | System-wide temporal audit: claims/source distribution/disputes/ranking. |
| telemetry_finding_worker.py | script | sandbox | reusable_prior | `scripts/telemetry_finding_worker.py` | Deterministic telemetry finding worker from Sticker feature vectors. |
| template_contract.py | script | sandbox | reusable_prior | `scripts/template_contract.py` | Small deterministic Jinja-like template contract for LUCIDOTA outputs. This is intentionally tiny: variable interpolation plus simple for-loops. It is not an expression engine and performs no network/model calls. |
| test_core_imports.py | script | sandbox | reusable_prior | `scripts/test_core_imports.py` | Subprocess gauntlet: core imports must exit 0 with clean stderr. |
| text_chunker.py | script | sandbox | reusable_prior | `scripts/text_chunker.py` | UNKNOWN — needs operator label; filename suggests: text chunker |
| tickletrunk_fault_injector.py | script | sandbox | reusable_prior | `scripts/tickletrunk_fault_injector.py` | UNKNOWN — needs operator label; filename suggests: tickletrunk fault injector |
| tickletrunk_scan.py | script | sandbox | reusable_prior | `scripts/tickletrunk_scan.py` | TICKLETRUNK proof-hoard scanner and access-layer builder. Default mode is dry-run. The scanner never deletes, moves, renames, or edits sovereign toolbox artifacts. Execute mode writes only manifest/access/report files. |
| timeline_compiler.py | script | sandbox | reusable_prior | `scripts/timeline_compiler.py` | UNKNOWN — needs operator label; filename suggests: timeline compiler |
| tool_function_bucket_manifest.py | script | sandbox | reusable_prior | `scripts/tool_function_bucket_manifest.py` | Build a repeatable, reusable function-level bucket manifest for tools and scripts. What this does: - Reads the canonical TICKLETRUNK manifest for non-destructive tool inventory. - Enriches with SCRIPT_AUDIT_MANIFEST function/role metadata w |
| topology_finding_extractor.py | script | sandbox | reusable_prior | `scripts/topology_finding_extractor.py` | Extract deterministic topology findings from existing design_atom rows. |
| tracer_claim_packet_bridge_dry_run.py | script | sandbox | reusable_prior | `scripts/tracer_claim_packet_bridge_dry_run.py` | Bridge TRACER-lite epistemic labels into claim-packet dry-run output. Dry-run only. No DB writes. No graph writes. Claim packets remain candidates. |
| updated_abcd_sequence_runner.py | script | sandbox | reusable_prior | `scripts/updated_abcd_sequence_runner.py` | Execute the operator-specified 100-item updated ABCD sequence with receipts. This runner exists to prevent bundle-counting. It records one JSONL row per operator sequence item and includes the exact ABCD permutation, target command, validat |
| villager_status.py | script | sandbox | reusable_prior | `scripts/villager_status.py` | Report Percyphon/villager status from receipts without model calls or graph writes. |
| work_order_importer.py | script | sandbox | reusable_prior | `scripts/work_order_importer.py` | UNKNOWN — needs operator label; filename suggests: work order importer |
| workflow_foundry_dry_run.py | script | sandbox | reusable_prior | `scripts/workflow_foundry_dry_run.py` | Mine repeated workflow variants; no auto-canonization Research-only dry-run scaffold. No DB writes. No graph writes. |
| workflow_foundry_runtime.py | script | sandbox | reusable_prior | `scripts/workflow_foundry_runtime.py` | Workflow Foundry runtime: design atoms -> invariant candidates. |
| working_reality_record.py | script | sandbox | reusable_prior | `scripts/working_reality_record.py` | Working Reality Law recorder: evidence -> hypothesis -> selected action frame. |

## MODELS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| models | model | unknown | reference | `01_REPOS/llama.cpp/models` | UNKNOWN — needs operator label; filename suggests: models |
| ggml-vocab-aquila.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-aquila.gguf` | Reference asset/file: ggml-vocab-aquila.gguf |
| ggml-vocab-baichuan.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-baichuan.gguf` | Reference asset/file: ggml-vocab-baichuan.gguf |
| ggml-vocab-bert-bge.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-bert-bge.gguf` | Reference asset/file: ggml-vocab-bert-bge.gguf |
| ggml-vocab-command-r.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-command-r.gguf` | Reference asset/file: ggml-vocab-command-r.gguf |
| ggml-vocab-deepseek-coder.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-deepseek-coder.gguf` | Reference asset/file: ggml-vocab-deepseek-coder.gguf |
| ggml-vocab-deepseek-llm.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-deepseek-llm.gguf` | Reference asset/file: ggml-vocab-deepseek-llm.gguf |
| ggml-vocab-falcon.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-falcon.gguf` | Reference asset/file: ggml-vocab-falcon.gguf |
| ggml-vocab-gemma-4.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-gemma-4.gguf` | Reference asset/file: ggml-vocab-gemma-4.gguf |
| ggml-vocab-gpt-2.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-gpt-2.gguf` | Reference asset/file: ggml-vocab-gpt-2.gguf |
| ggml-vocab-gpt-neox.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-gpt-neox.gguf` | Reference asset/file: ggml-vocab-gpt-neox.gguf |
| ggml-vocab-llama-bpe.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-llama-bpe.gguf` | Reference asset/file: ggml-vocab-llama-bpe.gguf |
| ggml-vocab-llama-spm.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-llama-spm.gguf` | Reference asset/file: ggml-vocab-llama-spm.gguf |
| ggml-vocab-mpt.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-mpt.gguf` | Reference asset/file: ggml-vocab-mpt.gguf |
| ggml-vocab-nomic-bert-moe.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-nomic-bert-moe.gguf` | Reference asset/file: ggml-vocab-nomic-bert-moe.gguf |
| ggml-vocab-phi-3.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-phi-3.gguf` | Reference asset/file: ggml-vocab-phi-3.gguf |
| ggml-vocab-qwen2.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-qwen2.gguf` | Reference asset/file: ggml-vocab-qwen2.gguf |
| ggml-vocab-refact.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-refact.gguf` | Reference asset/file: ggml-vocab-refact.gguf |
| ggml-vocab-starcoder.gguf | model | unknown | reference | `01_REPOS/llama.cpp/models/ggml-vocab-starcoder.gguf` | Reference asset/file: ggml-vocab-starcoder.gguf |
| Apriel-1.6-15b-Thinker-fixed.jinja | model | unknown | reference | `01_REPOS/llama.cpp/models/templates/Apriel-1.6-15b-Thinker-fixed.jinja` | UNKNOWN — needs operator label; filename suggests: Apriel 1.6 15b Thinker fixed |
| README.md | model | unknown | reference | `01_REPOS/llama.cpp/models/templates/README.md` | These templates can be updated with the following commands: |
| models | model | unknown | reference | `03_VAULT/models` | UNKNOWN — needs operator label; filename suggests: models |
| DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf | model | unknown | reference | `03_VAULT/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf` | Reference asset/file: DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf |
| model_source.json | model | unknown | reference | `03_VAULT/models/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/model_source.json` | UNKNOWN — needs operator label; filename suggests: model source |
| README.md | model | unknown | reference | `03_VAULT/models/gliner/urchade_gliner_small-v2.1/README.md` | license: apache-2.0 |
| gliner_config.json | model | unknown | reference | `03_VAULT/models/gliner/urchade_gliner_small-v2.1/gliner_config.json` | UNKNOWN — needs operator label; filename suggests: gliner config |
| pytorch_model.bin | model | unknown | reference | `03_VAULT/models/gliner/urchade_gliner_small-v2.1/pytorch_model.bin` | Reference asset/file: pytorch_model.bin |
| mamba-1.4b-hf-Q2_K.gguf | model | unknown | reference | `03_VAULT/models/mamba-1.4b-hf-Q2_K.gguf` | Reference asset/file: mamba-1.4b-hf-Q2_K.gguf |
| needle.pkl | model | unknown | reference | `03_VAULT/models/needle/needle.pkl` | Reference asset/file: needle.pkl |
| Ternary-Bonsai-4B-Q2_0.gguf | model | unknown | reference | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf` | Reference asset/file: Ternary-Bonsai-4B-Q2_0.gguf |
| model_source.json | model | unknown | reference | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/model_source.json` | UNKNOWN — needs operator label; filename suggests: model source |
| Falcon3-Mamba-7B-Instruct-Q2_K.gguf | model | unknown | reference | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf` | Reference asset/file: Falcon3-Mamba-7B-Instruct-Q2_K.gguf |
| model_source.json | model | unknown | reference | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/model_source.json` | UNKNOWN — needs operator label; filename suggests: model source |

## LORAS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| export-lora | lora | unknown | unknown | `01_REPOS/llama.cpp/tools/export-lora` | export-lora |
| CMakeLists.txt | lora | unknown | unknown | `01_REPOS/llama.cpp/tools/export-lora/CMakeLists.txt` | UNKNOWN — needs operator label; filename suggests: CMakeLists |
| README.md | lora | unknown | unknown | `01_REPOS/llama.cpp/tools/export-lora/README.md` | export-lora |
| export-lora.cpp | lora | unknown | unknown | `01_REPOS/llama.cpp/tools/export-lora/export-lora.cpp` | UNKNOWN — needs operator label; filename suggests: export lora |
| lora_cartridges | lora | unknown | unknown | `04_RUNTIME/lora_cartridges` | UNKNOWN — needs operator label; filename suggests: lora cartridges |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |
| manifest.json | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| train.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/train.jsonl` | UNKNOWN — needs operator label; filename suggests: train |
| validation.jsonl | lora | unknown | unknown | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/validation.jsonl` | UNKNOWN — needs operator label; filename suggests: validation |

## SCHEMAS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 06_SCHEMA | schema | sandbox | reference | `06_SCHEMA` | UNKNOWN — needs operator label; filename suggests: 06 SCHEMA |
| 111_model_generation_event_lane.sql | schema | sandbox | reference | `06_SCHEMA/111_model_generation_event_lane.sql` | FILE: 06_SCHEMA/111_model_generation_event_lane.sql |
| 112_project2501_core_board.sql | schema | sandbox | reference | `06_SCHEMA/112_project2501_core_board.sql` | FILE: 06_SCHEMA/112_project2501_core_board.sql |
| 113_project2501_bytewax_board_stream.sql | schema | sandbox | reference | `06_SCHEMA/113_project2501_bytewax_board_stream.sql` | FILE: 06_SCHEMA/113_project2501_bytewax_board_stream.sql |
| 114_project2501_board_worker.sql | schema | sandbox | reference | `06_SCHEMA/114_project2501_board_worker.sql` | FILE: 06_SCHEMA/114_project2501_board_worker.sql |
| 115_project2501_script_audit_worker.sql | schema | sandbox | reference | `06_SCHEMA/115_project2501_script_audit_worker.sql` | FILE: 06_SCHEMA/115_project2501_script_audit_worker.sql |
| 116_hunch_postgres_ingest.sql | schema | sandbox | reference | `06_SCHEMA/116_hunch_postgres_ingest.sql` | hunch_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(), |
| 117_working_reality_law.sql | schema | sandbox | reference | `06_SCHEMA/117_working_reality_law.sql` | move_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(), |
| abductive_db_os_objects.v1.json | schema | sandbox | reference | `06_SCHEMA/abductive_db_os/abductive_db_os_objects.v1.json` | UNKNOWN — needs operator label; filename suggests: abductive db os objects.v1 |
| absurd_abductive_objects.v1.json | schema | sandbox | reference | `06_SCHEMA/absurd_abductive/absurd_abductive_objects.v1.json` | UNKNOWN — needs operator label; filename suggests: absurd abductive objects.v1 |
| ontology_packs | schema | sandbox | reference | `BOOKS/ontology_packs` | UNKNOWN — needs operator label; filename suggests: ontology packs |
| README.md | schema | sandbox | reference | `BOOKS/ontology_packs/sio8/README.md` | SIO-8 Sovereign Intelligence Ontology Pack |
| registry.json | schema | sandbox | reference | `BOOKS/ontology_packs/sio8/registry.json` | UNKNOWN — needs operator label; filename suggests: registry |

## SKILLS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| skills | skill | unknown | reference | `/home/mfspx/.codex/skills` | UNKNOWN — needs operator label; filename suggests: skills |
| LICENSE.txt | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/LICENSE.txt` | UNKNOWN — needs operator label; filename suggests: LICENSE |
| SKILL.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/SKILL.md` | name: "imagegen" |
| openai.yaml | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/agents/openai.yaml` | UNKNOWN — needs operator label; filename suggests: openai |
| cli.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/references/cli.md` | CLI reference (`scripts/image_gen.py`) |
| codex-network.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/references/codex-network.md` | Codex network approvals / sandbox notes |
| image-api.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/references/image-api.md` | Image API quick reference |
| prompting.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/references/prompting.md` | Prompting best practices |
| sample-prompts.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/references/sample-prompts.md` | Sample prompts (copy/paste) |
| image_gen.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/scripts/image_gen.py` | Fallback CLI for explicit image generation or editing with GPT Image models. Used only when the user explicitly opts into CLI fallback mode, or when explicit transparent output requires the `gpt-image-1.5` fallback path. Defaults to gpt-ima |
| remove_chroma_key.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/imagegen/scripts/remove_chroma_key.py` | Remove a solid chroma-key background from an image. This helper supports the imagegen skill's built-in-first transparent workflow: generate an image on a flat key color, then convert that key color to alpha. |
| LICENSE.txt | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/LICENSE.txt` | UNKNOWN — needs operator label; filename suggests: LICENSE |
| SKILL.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/SKILL.md` | name: "openai-docs" |
| openai.yaml | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/agents/openai.yaml` | UNKNOWN — needs operator label; filename suggests: openai |
| latest-model.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/references/latest-model.md` | Latest model guide |
| prompting-guide.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/references/prompting-guide.md` | GPT-5.5 works best when prompts define the outcome and leave room for the model to choose an efficient solution path. Compared with earlier models, you can often use shorter, more outcome-oriented prompts: describe what good looks like, wha |
| upgrade-guide.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/references/upgrade-guide.md` | Upgrading to GPT-5.5 |
| resolve-latest-model-info.js | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/openai-docs/scripts/resolve-latest-model-info.js` | const fs = require("node:fs/promises"); |
| SKILL.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/SKILL.md` | name: plugin-creator |
| openai.yaml | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/agents/openai.yaml` | UNKNOWN — needs operator label; filename suggests: openai |
| installing-and-updating.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/references/installing-and-updating.md` | Updating Existing Local Plugins |
| plugin-json-spec.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/references/plugin-json-spec.md` | Plugin JSON sample spec |
| create_basic_plugin.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/scripts/create_basic_plugin.py` | Scaffold a plugin directory and optionally update marketplace.json. |
| read_marketplace_name.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/scripts/read_marketplace_name.py` | Print the top-level marketplace name from any marketplace.json file. |
| update_plugin_cachebuster.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/scripts/update_plugin_cachebuster.py` | Rewrite a local plugin version to a single Codex cachebuster suffix. |
| validate_plugin.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/plugin-creator/scripts/validate_plugin.py` | Validate a generated plugin against the plugin ingestion contract. |
| SKILL.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/SKILL.md` | name: skill-creator |
| openai.yaml | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/agents/openai.yaml` | UNKNOWN — needs operator label; filename suggests: openai |
| license.txt | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/license.txt` | UNKNOWN — needs operator label; filename suggests: license |
| openai_yaml.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/references/openai_yaml.md` | openai.yaml fields (full example + descriptions) |
| generate_openai_yaml.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/scripts/generate_openai_yaml.py` | OpenAI YAML Generator - Creates agents/openai.yaml for a skill folder. Usage: generate_openai_yaml.py <skill_dir> [--name <skill_name>] [--interface key=value] |
| init_skill.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/scripts/init_skill.py` | Skill Initializer - Creates a new skill from template Usage: init_skill.py <skill-name> --path <path> [--resources scripts,references,assets] [--examples] [--interface key=value] Examples: init_skill.py my-new-skill --path skills/public ini |
| quick_validate.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-creator/scripts/quick_validate.py` | Quick validation script for skills - minimal version |
| LICENSE.txt | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/LICENSE.txt` | UNKNOWN — needs operator label; filename suggests: LICENSE |
| SKILL.md | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/SKILL.md` | name: skill-installer |
| openai.yaml | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/agents/openai.yaml` | UNKNOWN — needs operator label; filename suggests: openai |
| github_utils.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/scripts/github_utils.py` | Shared GitHub helpers for skill install scripts. |
| install-skill-from-github.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py` | Install a skill from a GitHub repo path into $CODEX_HOME/skills. |
| list-skills.py | skill | unknown | reference | `/home/mfspx/.codex/skills/.system/skill-installer/scripts/list-skills.py` | List skills from a GitHub repo path. |

## PLUGINS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| plugins | plugin | unknown | reference | `/home/mfspx/.claw/plugins` | UNKNOWN — needs operator label; filename suggests: plugins |
| installed.json | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed.json` | UNKNOWN — needs operator label; filename suggests: installed |
| plugin.json | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/example-bundled-bundled/.claw-plugin/plugin.json` | UNKNOWN — needs operator label; filename suggests: plugin |
| post.sh | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/example-bundled-bundled/hooks/post.sh` | bin/sh |
| pre.sh | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/example-bundled-bundled/hooks/pre.sh` | bin/sh |
| plugin.json | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/sample-hooks-bundled/.claw-plugin/plugin.json` | UNKNOWN — needs operator label; filename suggests: plugin |
| post.sh | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/sample-hooks-bundled/hooks/post.sh` | bin/sh |
| pre.sh | plugin | unknown | reference | `/home/mfspx/.claw/plugins/installed/sample-hooks-bundled/hooks/pre.sh` | bin/sh |
| plugins | plugin | unknown | reference | `/home/mfspx/.codex/plugins` | UNKNOWN — needs operator label; filename suggests: plugins |
| CODE_OF_CONDUCT.md | plugin | unknown | reference | `/home/mfspx/.codex/plugins/cache/openai-curated/superpowers/11b5af68/CODE_OF_CONDUCT.md` | Contributor Covenant Code of Conduct |
| README.md | plugin | unknown | reference | `/home/mfspx/.codex/plugins/cache/openai-curated/superpowers/11b5af68/README.md` | Superpowers |

## SERVICES

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| services | service | unknown | candidate_for_promotion | `services` | UNKNOWN — needs operator label; filename suggests: services |
| __init__.py | service | unknown | candidate_for_promotion | `services/__init__.py` | UNKNOWN — needs operator label; filename suggests: init |
| __init__.py | service | unknown | candidate_for_promotion | `services/fairyfuse/__init__.py` | UNKNOWN — needs operator label; filename suggests: init |
| fairyfuse_backend.py | service | unknown | candidate_for_promotion | `services/fairyfuse/fairyfuse_backend.py` | FairyFuse resident ternary backend for LUCIDOTA. Local-only CPU residency layer: - memory-map packed ternary weights when present; - invoke an optional fused ternary GEMV shared library through ctypes; - otherwise keep a deterministic multi |
| README.md | service | unknown | candidate_for_promotion | `services/ternary_lab/README.md` | LUCIDOTA Ternary Lens Lab |
| vendor_manifest.json | service | unknown | candidate_for_promotion | `services/ternary_lab/vendor_manifest.json` | UNKNOWN — needs operator label; filename suggests: vendor manifest |

## BOOKS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| BOOKS | book | unknown | reference | `BOOKS` | UNKNOWN — needs operator label; filename suggests: BOOKS |
| indy_reads_judgments.csv | book | unknown | reference | `BOOKS/.indy_reads/indy_reads_judgments.csv` | Reference asset/file: indy_reads_judgments.csv |
| watch_state.json | book | unknown | reference | `BOOKS/.indy_reads/watch_state.json` | UNKNOWN — needs operator label; filename suggests: watch state |
| A Big Boy Did It and Ran Away -- Brookmyre, Christopher -- 2003 -- Abacus -- isbn13 9780316857437 -- 017896b0f90479cdc825e5bf9858e2a1 -- Anna’s Archive.mobi | book | unknown | reference | `BOOKS/A Big Boy Did It and Ran Away -- Brookmyre, Christopher -- 2003 -- Abacus -- isbn13 9780316857437 -- 017896b0f90479cdc825e5bf9858e2a1 -- Anna’s Archive.mobi` | Reference asset/file: A Big Boy Did It and Ran Away -- Brookmyre, Christopher -- 2003 -- Abacus -- isbn13 9780316857437 -- 017896b0f90479cdc825e5bf9858e2a1 -- Anna’s Archive.mobi |
| A Death in Malta - An Assassination and a Family's Quest for -- Paul Caruana Galizia -- 1, 2023 -- Penguin Publishing Group -- isbn13 9780593543733 -- 8f3f41586e1dc07c5d332d97a462519f -- Anna’s Archive.epub | book | unknown | reference | `BOOKS/A Death in Malta - An Assassination and a Family's Quest for -- Paul Caruana Galizia -- 1, 2023 -- Penguin Publishing Group -- isbn13 9780593543733 -- 8f3f41586e1dc07c5d332d97a462519f -- Anna’s Archive.epub` | Reference asset/file: A Death in Malta - An Assassination and a Family's Quest for -- Paul Caruana Galizia -- 1, 2023 -- Penguin Publishing Group -- isbn13 9780593543733 -- 8f3f41586e1dc07c5d332d97a462519f -- Anna’s Archive.epub |
| ACTIVE_ONTOLOGY.json | book | unknown | reference | `BOOKS/ACTIVE_ONTOLOGY.json` | Reference asset/file: ACTIVE_ONTOLOGY.json |
| Blood in the Machine_ The Origins of the Rebellion Against -- Brian Merchant -- Sep 26, 2023 -- Little, Brown and Company -- isbn13 9780316487733 -- 34659fa17bb29d21160839ac4be9c421 -- Anna’s Archive.epub | book | unknown | reference | `BOOKS/Blood in the Machine_ The Origins of the Rebellion Against -- Brian Merchant -- Sep 26, 2023 -- Little, Brown and Company -- isbn13 9780316487733 -- 34659fa17bb29d21160839ac4be9c421 -- Anna’s Archive.epub` | Reference asset/file: Blood in the Machine_ The Origins of the Rebellion Against -- Brian Merchant -- Sep 26, 2023 -- Little, Brown and Company -- isbn13 9780316487733 -- 34659fa17bb29d21160839ac4be9c421 -- Anna’s Archive.epub |
| GO_ACTIVE_TERMS.json | book | unknown | reference | `BOOKS/GO_ACTIVE_TERMS.json` | Reference asset/file: GO_ACTIVE_TERMS.json |
| GO_EXTENSIONS.json | book | unknown | reference | `BOOKS/GO_EXTENSIONS.json` | Reference asset/file: GO_EXTENSIONS.json |
| GO_GAME_GRADING_SCHEMA.json | book | unknown | reference | `BOOKS/GO_GAME_GRADING_SCHEMA.json` | Reference asset/file: GO_GAME_GRADING_SCHEMA.json |
| GO_GRAPH_EDGE_ENVELOPE.schema.json | book | unknown | reference | `BOOKS/GO_GRAPH_EDGE_ENVELOPE.schema.json` | Reference asset/file: GO_GRAPH_EDGE_ENVELOPE.schema.json |
| GO_GRAPH_IDENTITY_VERSIONING_RULES.json | book | unknown | reference | `BOOKS/GO_GRAPH_IDENTITY_VERSIONING_RULES.json` | Reference asset/file: GO_GRAPH_IDENTITY_VERSIONING_RULES.json |
| GO_GRAPH_ITEM_ENVELOPE.schema.json | book | unknown | reference | `BOOKS/GO_GRAPH_ITEM_ENVELOPE.schema.json` | Reference asset/file: GO_GRAPH_ITEM_ENVELOPE.schema.json |
| GO_GRAPH_STORAGE_PROMOTION_RULES.json | book | unknown | reference | `BOOKS/GO_GRAPH_STORAGE_PROMOTION_RULES.json` | Reference asset/file: GO_GRAPH_STORAGE_PROMOTION_RULES.json |
| GO_ONTOLOGY_SCHEMA.json | book | unknown | reference | `BOOKS/GO_ONTOLOGY_SCHEMA.json` | Reference asset/file: GO_ONTOLOGY_SCHEMA.json |
| One Day, Everyone Will Have Always Been Against This -- Omar El Akkad -- 2025 -- 5832a6537facbd0df292e3891f675469 -- Anna’s Archive.mobi | book | unknown | reference | `BOOKS/One Day, Everyone Will Have Always Been Against This -- Omar El Akkad -- 2025 -- 5832a6537facbd0df292e3891f675469 -- Anna’s Archive.mobi` | Reference asset/file: One Day, Everyone Will Have Always Been Against This -- Omar El Akkad -- 2025 -- 5832a6537facbd0df292e3891f675469 -- Anna’s Archive.mobi |
| Out of Darkness _ Essays on Corporate Power and Civic -- Ralph Nader, Lewis H_ Lapham -- Penguin Random House LLC (Publisher Services), New York, NY, -- isbn13 9781644213735 -- b8b55954e02caad6c02ddb49fc04553f -- Anna’s A.epub | book | unknown | reference | `BOOKS/Out of Darkness _ Essays on Corporate Power and Civic -- Ralph Nader, Lewis H_ Lapham -- Penguin Random House LLC (Publisher Services), New York, NY, -- isbn13 9781644213735 -- b8b55954e02caad6c02ddb49fc04553f -- Anna’s A.epub` | Reference asset/file: Out of Darkness _ Essays on Corporate Power and Civic -- Ralph Nader, Lewis H_ Lapham -- Penguin Random House LLC (Publisher Services), New York, NY, -- isbn13 9781644213735 -- b8b55954e02caad6c02ddb49fc04553f -- Anna’ |
| The Small and the Mighty_ Twelve Unsung Americans Who -- Sharon McMahon -- 2024 -- Penguin Publishing Group -- b8a3ddf06b1fa20c5f285a346c655595 -- Anna’s Archive.pdf | book | unknown | reference | `BOOKS/The Small and the Mighty_ Twelve Unsung Americans Who -- Sharon McMahon -- 2024 -- Penguin Publishing Group -- b8a3ddf06b1fa20c5f285a346c655595 -- Anna’s Archive.pdf` | Reference asset/file: The Small and the Mighty_ Twelve Unsung Americans Who -- Sharon McMahon -- 2024 -- Penguin Publishing Group -- b8a3ddf06b1fa20c5f285a346c655595 -- Anna’s Archive.pdf |

## SURFACES

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 07_SURFACES | surface | unknown | candidate_for_promotion | `07_SURFACES` | UNKNOWN — needs operator label; filename suggests: 07 SURFACES |
| conversation_instruction_compiler_sample.html | surface | unknown | candidate_for_promotion | `07_SURFACES/generated/conversation_instruction_compiler_sample.html` | UNKNOWN — needs operator label; filename suggests: conversation instruction compiler sample |
| marrow_loop_status.html | surface | unknown | candidate_for_promotion | `07_SURFACES/generated/marrow_loop_status.html` | UNKNOWN — needs operator label; filename suggests: marrow loop status |
| status_ledger.html | surface | unknown | candidate_for_promotion | `07_SURFACES/generated/status_ledger.html` | UNKNOWN — needs operator label; filename suggests: status ledger |
| conversation_instruction_compiler_sample.json | surface | unknown | candidate_for_promotion | `07_SURFACES/sidecars/conversation_instruction_compiler_sample.json` | UNKNOWN — needs operator label; filename suggests: conversation instruction compiler sample |
| marrow_loop_status.json | surface | unknown | candidate_for_promotion | `07_SURFACES/sidecars/marrow_loop_status.json` | UNKNOWN — needs operator label; filename suggests: marrow loop status |

## SCRAPERS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| fetch_server_test_models.py | scraper | sandbox | reusable_prior | `01_REPOS/llama.cpp/scripts/fetch_server_test_models.py` | This script fetches all the models used in the server tests. This is useful for slow tests that use larger models, to avoid them timing out on the model downloads. It is meant to be run from the root of the repository. Example: python scrip |
| 003_survey_protocol.sql | scraper | sandbox | reference | `06_SCHEMA/003_survey_protocol.sql` | Survey / hop-pivot intake schema for LUCIDOTA. |
| 011_body_capture.sql | scraper | sandbox | reference | `06_SCHEMA/011_body_capture.sql` | Body Capture capture/evidence diff schema. |
| lucidota_body_capture.py | scraper | unknown | reusable_prior | `scripts/legacy/lucidota_body_capture.py` | Body Capture capture v0: operator-supervised HTTP/body snapshot into local CAS. This is the first non-browser capture slice. Browser screenshots/DOM via Playwright come next; this establishes the same evidence path: fetch, hash, CAS, metada |
| lucidota_dbos_survey.py | scraper | unknown | reusable_prior | `scripts/legacy/lucidota_dbos_survey.py` | DBOS wrapper for the Survey Protocol. This makes Survey a durable workflow-shaped action without changing Survey's CLI. DBOS is command brain; Survey is the tool; workflow_event remains the common event surface. |
| lucidota_survey.py | scraper | unknown | reusable_prior | `scripts/legacy/lucidota_survey.py` | LUCIDOTA Survey Protocol: first-contact URL/file triage + local CAS + pivot hints. |
| lucidota_body_capture.py | scraper | unknown | reusable_prior | `scripts/lucidota_body_capture.py` | Active/importable entrypoint for Body Capture until the Rust adapter replaces it. |
| lucidota_body_capture_evidence.py | scraper | sandbox | reusable_prior | `scripts/lucidota_body_capture_evidence.py` | Body Capture evidence bundle + text/Wayback diff. Reads latest Body Capture captures from local CAS, computes a compact text diff and exports a JSON evidence bundle. Optional Wayback comparison fetches one archived snapshot for current-vs-a |
| lucidota_body_capture_policy.py | scraper | sandbox | reusable_prior | `scripts/lucidota_body_capture_policy.py` | Body Capture watcher policy evaluator. Evaluates latest capture pairs against watcher profiles. Capture is evidence; alerts are policy-dependent. |
| lucidota_browser_body_capture.py | scraper | sandbox | reusable_prior | `scripts/lucidota_browser_body_capture.py` | Body Capture browser capture contract. Browser rendering is policy-gated fallback, not default extraction. If no browser binary is available, this reports skipped instead of fake success. |
| lucidota_security_scan.py | scraper | sandbox | reusable_prior | `scripts/lucidota_security_scan.py` | Lightweight repo security tripwire for LUCIDOTA. This scanner must be boring and non-leaky: report locations and masked matches only. Never echo live-looking credential material back to stdout. |
| lucidota_survey.py | scraper | unknown | reusable_prior | `scripts/lucidota_survey.py` | Active/importable entrypoint for the Survey adapter until the Rust port lands. |

## WORKFLOWS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 00_PROJECT_BRAIN | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN` | LUCIDOTA Project Brain — Archive Pointer |
| ACTIVE_INSTRUCTION_INDEX.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md` | Active Canonical Instruction Sources |
| 01_IDENTITY_AND_CLAIM_STATE_LAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` | <!-- |
| 02_EXECUTION_SPINE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` | <!-- |
| 03_CUSTODY_ETL_PIPELINE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` | LUCIDOTA Custody ETL Pipeline |
| 04_DEV_LIBRARY_REUSE_LAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md` | LUCIDOTA Dev Library Reuse Law |
| 05_COMPONENT_AUTHORITY_MAP.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` | <!-- |
| 06_BARE_STEEL_DOCTRINE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md` | Bare Steel Doctrine |
| 07_WORKING_REALITY_LAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/07_WORKING_REALITY_LAW.md` | Working Reality Law / Ontology Humility Contract |
| 08_BOARD_EFFECT_TOURNAMENT_LAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ACTIVE_SPEC/08_BOARD_EFFECT_TOURNAMENT_LAW.md` | Board Effect Tournament Law |
| AGENT_SELF_SOVEREIGNTY_CHARTER.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/AGENT_SELF_SOVEREIGNTY_CHARTER.md` | agentSI Operational Self-Sovereignty Charter |
| ARCHITECTURE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/ARCHITECTURE.md` | agentSI Architecture |
| JOB_BOOTHS.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/JOB_BOOTHS.json` | UNKNOWN — needs operator label; filename suggests: JOB BOOTHS |
| MVP_PLAN.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/MVP_PLAN.md` | agentSI MVP Plan |
| PERSONA_SEED_SCHEMA.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/PERSONA_SEED_SCHEMA.json` | UNKNOWN — needs operator label; filename suggests: PERSONA SEED SCHEMA |
| README.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/README.md` | agentSI — Agent Self-Sovereign Job Fair |
| STATUS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/STATUS.md` | agentSI Current Status |
| BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md` | Blueprint-First / PocketFlow Pseudolaw |
| CANONICAL_FINISHED_PRODUCT_MAP.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md` | Consolidation Notice |
| CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json` | UNKNOWN — needs operator label; filename suggests: CODEX PROMPTING GUIDE LUCIDOTA POLICY |
| CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.md` | Codex Prompting Guide - LUCIDOTA Policy |
| DURABLE_WORKFLOW_DECISION.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json` | UNKNOWN — needs operator label; filename suggests: DURABLE WORKFLOW DECISION |
| DBOS_CONTRACT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ETL_PIPELINE/DBOS_CONTRACT.md` | ETL Pipeline DBOS Contract |
| DB_MAPPING.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ETL_PIPELINE/DB_MAPPING.md` | ETL Pipeline → Existing LUCIDOTA DB Mapping |
| LIVE_INGEST_NOTES.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ETL_PIPELINE/LIVE_INGEST_NOTES.md` | ETL Pipeline — Notes From The Live Ingest |
| PLAN.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ETL_PIPELINE/PLAN.md` | ETL Pipeline Implementation Plan |
| WORKER_POLICY.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/ETL_PIPELINE/WORKER_POLICY.md` | ETL Pipeline Worker Policy |
| FILESYSTEM_TREE_INDEX_CURRENT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/FILESYSTEM_TREE_INDEX_CURRENT.md` | LUCIDOTA FULL SYSTEM FILESYSTEM TREE INDEX — CURRENT |
| ARCHITECTURE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ARCHITECTURE.md` | Architectural Design Document: INDY_READs Polycareer Workflow Wizard |
| GLOW_HUNTER_SEEDLIST.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/GLOW_HUNTER_SEEDLIST.md` | Glow Hunter Seedlist |
| README.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/README.md` | INDY_READs Polycareer Workflow Wizard |
| ROLE_MODES.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ROLE_MODES.json` | UNKNOWN — needs operator label; filename suggests: ROLE MODES |
| WORKFLOW_CONTRACT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/WORKFLOW_CONTRACT.md` | INDY_READs Workflow Contract |
| BLUEPRINT_FIRST_MODEL_SECOND.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/BLUEPRINT_FIRST_MODEL_SECOND.md` | Knowledge Card: Blueprint First, Model Second |
| CYBERCRAFTER_DRF.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/CYBERCRAFTER_DRF.md` | Knowledge Card: CyberCrafter Deterministic Reasoning Framework (DRF) |
| FLYWHEEL1412_NCNN.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/FLYWHEEL1412_NCNN.md` | Knowledge Card: flywheel1412 ncnn |
| LLM_WORKFLOW_ROUTER_DOBY_BAXTER.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/LLM_WORKFLOW_ROUTER_DOBY_BAXTER.md` | Knowledge Card: Doby Baxter LLM Workflow Router |
| README.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/README.md` | LUCIDOTA Knowledge Library |
| SYDSEC_SYD.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/SYDSEC_SYD.md` | Knowledge Card: Sydsec Syd |
| SYDSEC_SYD_STUB.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/SYDSEC_SYD_STUB.md` | Knowledge Stub: Sydsec Syd |
| UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents.html | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents.html` | UNKNOWN — needs operator label; filename suggests: UNIT   Privacy and Records Management   2026 31 RE   Response Records.pdf   All Documents |
| 1996.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/1996.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([[1996],{6176(a,b,c){c.d(b,{a:()=>g});var d=c("odsp.util_162"),e=c(4703),f=c(2822);let g=(0,d.Dp)("switchViewActionProvider",class{evaluateAction({view:a}){return |
| 2006.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/2006.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([[2006],{8657(a,b,c){c.d(b,{a:()=>ai});var d=c(3139),e=c(6393),f=c(4265),g=c(6183),h=c(2842),i=c(2987),j=c(8112),k=c("odsp.util_162");let l=k.Ov.isActivated("3c1d |
| 275.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/275.js` | UNKNOWN — needs operator label; filename suggests: 275 |
| deferred.items-view.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.items-view.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["deferred.items-view"],{483(a,b,c){c.r(b),c.d(b,{ApproveRejectAction:()=>ar.a,FileHandlerOpenAction:()=>as.a,GroupByAction:()=>aj,GroupByActionExecutor:()=>ak.G |
| deferred.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.js` | UNKNOWN — needs operator label; filename suggests: deferred |
| deferred.odsp-common.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.odsp-common.js` | UNKNOWN — needs operator label; filename suggests: deferred.odsp common |
| deferred.odsp-datasources.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.odsp-datasources.js` | UNKNOWN — needs operator label; filename suggests: deferred.odsp datasources |
| deferred.odsp-shared-react.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.odsp-shared-react.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["deferred.odsp-shared-react"],{2543(a,b,c){c.r(b),c.d(b,{SiteScriptProgressIndicator:()=>u});var d=c("react-lib"),e=c(618),f=c(619),g=c(219),h=c(39),i=c(11132), |
| deferred.office-ui-fabric-react.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.office-ui-fabric-react.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["deferred.office-ui-fabric-react"],{6922(a,b,c){c.d(b,{a:()=>d});let d=(0,c(427).a)("Dismiss20Regular","20",["m4.09 4.22.06-.07a.5.5 0 0 1 .63-.06l.07.06L10 9.2 |
| deferred.resx.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.resx.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["deferred.resx"],{11416(a){a.exports=JSON.parse('{"a":"Open in Immersive Reader"}')} |
| fp-min.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/fp-min.js` | Footprint=function(){var t=1,n=2,e=t\|n,r=8,o=16,i=r\|o,u=e\|i,a=128,f=256,s=e\|(a\|f),c="http://",l=200,m="trans.gif",p="/apc/",g=5e3,d="trans.gif",h="100k.gif",v=822.128,w=1e3,T="GET",y="POST",M=-1,I="20190214",b="x-userhostaddress",D="x- |
| fui.core-2c6f08cc.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/fui.core-2c6f08cc.js` | UNKNOWN — needs operator label; filename suggests: fui.core 2c6f08cc |
| fui.util-39ae0f8d.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/fui.util-39ae0f8d.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["fui.util"],{"fui.util_713"(a,b,c){let d;c.d(b,{co:()=>bS,Kc:()=>cf,Ec:()=>Q,bn:()=>bZ,E$:()=>al,MI:()=>bY,rc:()=>ai,Fq:()=>a3,IJ:()=>ah,r5:()=>bb,SQ:()=>p,yP:( |
| initial.resx.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/initial.resx.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["initial.resx"],{2228(a){a.exports=JSON.parse('{"ComponentName":"agent","ComponentShortName":"agent","ComponentPluralName":"agents"}')} |
| modernFrame.html | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame.html` | UNKNOWN — needs operator label; filename suggests: modernFrame |
| a.html | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/a.html` | UNKNOWN — needs operator label; filename suggests: a |
| a_002.html | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/a_002.html` | UNKNOWN — needs operator label; filename suggests: a 002 |
| initial.resx.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/initial.resx.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["initial.resx"],{2187(a){a.exports=JSON.parse('{"ComponentName":"agent","ComponentShortName":"agent","ComponentPluralName":"agents"}')} |
| modernframe.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/modernframe.js` | UNKNOWN — needs operator label; filename suggests: modernframe |
| plt.listviewdataprefetch.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/plt.listviewdataprefetch.js` | UNKNOWN — needs operator label; filename suggests: plt.listviewdataprefetch |
| odsp.knockout.lib-fa98a149.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/odsp.knockout.lib-fa98a149.js` | For license information please see odsp.knockout.lib-fa98a149.js.LICENSE.txt */ |
| odsp.react.lib-760716ce.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/odsp.react.lib-760716ce.js` | UNKNOWN — needs operator label; filename suggests: odsp.react.lib 760716ce |
| odsp.util-463f631c.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/odsp.util-463f631c.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["odsp.util"],{"odsp.util_162":function(a,b,c){let d,e,f,g,h,i,j,k,l,m;function n(a,b){let c="";for(;c.length<a;){let a=16*(b??Math.random)();a\|=0,c+=a.toString |
| officebrowserfeedbackstrings.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/officebrowserfeedbackstrings.js` | OfficeBrowserFeedback.setUiStrings({FeedbackSubtitle:"Send Feedback to Microsoft",PrivacyStatement:"Privacy Statement",Form:{CommentPlaceholder:"Please do not include any confidential or personal information in your comment",CategoryPlaceho |
| ondemand.resx.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/ondemand.resx.js` | UNKNOWN — needs operator label; filename suggests: ondemand.resx |
| plt.items-view.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.items-view.js` | UNKNOWN — needs operator label; filename suggests: plt.items view |
| plt.listviewdataprefetch.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.listviewdataprefetch.js` | var __webpack_result__;(()=>{"use strict";var a,b,c,d,e={1545(a,b,c){c.r(b),c.d(b,{__assign:()=>d.q5,__asyncDelegator:()=>d.DQ,__asyncGenerator:()=>d.$0,__asyncValues:()=>d.Gl,__await:()=>d.HE,__awaiter:()=>d.yv,__classPrivateFieldGet:()=>d |
| plt.odsp-common.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.odsp-common.js` | UNKNOWN — needs operator label; filename suggests: plt.odsp common |
| plt.office-ui-fabric-react.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.office-ui-fabric-react.js` | UNKNOWN — needs operator label; filename suggests: plt.office ui fabric react |
| plt.preact.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.preact.js` | "use strict";(self.odspNextWebpackJsonp=self.odspNextWebpackJsonp\|\|[]).push([["plt.preact"],{3413(a,b,c){c.r(b),c.d(b,{Component:()=>u,Fragment:()=>t,cloneElement:()=>K,createContext:()=>L,createElement:()=>q,createRef:()=>s,h:()=>q,hydra |
| splistwebpack.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/splistwebpack.js` | UNKNOWN — needs operator label; filename suggests: splistwebpack |
| suiteux.shell.core.a34fe4706885bcb77f89.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.core.a34fe4706885bcb77f89.js` | UNKNOWN — needs operator label; filename suggests: suiteux.shell.core.a34fe4706885bcb77f89 |
| suiteux.shell.ecsclient.cecbd5ef83e028645cf5.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.ecsclient.cecbd5ef83e028645cf5.js` | function(){var e={401:function(e,t,r){r(832),e.exports=self.fetch.bind(self)},832:function(e,t,r){"use strict";r.r(t),r.d(t,{DOMException:function(){return T},Headers:function(){return l},Request:function(){return b},Response:function(){ret |
| suiteux.shell.mast.d883ad4900e81860f22a.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.mast.d883ad4900e81860f22a.js` | UNKNOWN — needs operator label; filename suggests: suiteux.shell.mast.d883ad4900e81860f22a |
| suiteux.shell.otellogging.d1a49d28c74cd9d315b0.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.otellogging.d1a49d28c74cd9d315b0.js` | UNKNOWN — needs operator label; filename suggests: suiteux.shell.otellogging.d1a49d28c74cd9d315b0 |
| suiteux.shell.plus.4a6e59c2f4ac09534ca6.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.plus.4a6e59c2f4ac09534ca6.js` | UNKNOWN — needs operator label; filename suggests: suiteux.shell.plus.4a6e59c2f4ac09534ca6 |
| suiteux.shell.responsive.602279667ad94abd2293.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.responsive.602279667ad94abd2293.js` | var shellPerformance=window.performance,HighResolutionTimingSupported=!!shellPerformance&&"function"==typeof shellPerformance.mark;HighResolutionTimingSupported&&shellPerformance.mark("shell_responsive_start"),(self["suiteux_shell_webpackJs |
| suiteux.shell.search.02ecc588cc567de9ee02.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.search.02ecc588cc567de9ee02.js` | var shellPerformance=window.performance,HighResolutionTimingSupported=!!shellPerformance&&"function"==typeof shellPerformance.mark;HighResolutionTimingSupported&&shellPerformance.mark("shell_search_start"),(self["suiteux_shell_webpackJsonp_ |
| suiteux.shell.searchbox.db39686c69105329489b.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.searchbox.db39686c69105329489b.js` | UNKNOWN — needs operator label; filename suggests: suiteux.shell.searchbox.db39686c69105329489b |
| tslib-30bcb21f.js | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/tslib-30bcb21f.js` | For license information please see tslib-30bcb21f.js.LICENSE.txt */ |
| VRAM_LLM_SELECTION_BESPOKE_WIRING_BEGINNINGS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/VRAM_LLM_SELECTION_BESPOKE_WIRING_BEGINNINGS.md` | Knowledge Card: VRAM LLM Selection — Bespoke and Wiring Beginnings |
| index.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/index.json` | UNKNOWN — needs operator label; filename suggests: index |
| LUCIDOTA_CANONICAL_MANIFEST.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/LUCIDOTA_CANONICAL_MANIFEST.md` | Consolidation Notice |
| LUCIDOTA_GAPS_ACTION_WORKFLOWS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/LUCIDOTA_GAPS_ACTION_WORKFLOWS.md` | LUCIDOTA GAPS → ACTIONABLE WORKFLOWS |
| LUCIDOTA_OPS_MANUAL.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/LUCIDOTA_OPS_MANUAL.md` | Consolidation Notice |
| POSTGRES_AUDIT_CURRENT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/POSTGRES_AUDIT_CURRENT.md` | LUCIDOTA POSTGRES AUDIT — CURRENT |
| PROJECT_2501_ADMIN_PROMPT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/PROJECT_2501_ADMIN_PROMPT.md` | PROJECT 2501 — THE MAJOR ADMIN PROMPT |
| PROJECT_2501_CORE_CONTRACT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/PROJECT_2501_CORE_CONTRACT.md` | PROJECT 2501 CORE CONTRACT |
| READTHISFIRST_CURRENT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/READTHISFIRST_CURRENT.md` | LUCIDOTA READ THIS FIRST — CURRENT |
| GOAL_COMPLETION_AUDIT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/GOAL_COMPLETION_AUDIT.md` | LUCIDOTA Goal Completion Audit |
| GOAL_REQUIREMENT_MATRIX.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/GOAL_REQUIREMENT_MATRIX.json` | UNKNOWN — needs operator label; filename suggests: GOAL REQUIREMENT MATRIX |
| README.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/README.md` | LUCIDOTA RFC Program |
| RFC-000-MASTER-THESIS-PROGRAM.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-000-MASTER-THESIS-PROGRAM.md` | RFC-000: LUCIDOTA Master Thesis Program |
| RFC-001-SYSTEM-THESIS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-001-SYSTEM-THESIS.md` | RFC-001: LUCIDOTA System Thesis |
| RFC-010-SLOP-LAWS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-010-SLOP-LAWS.md` | RFC-010: Slop Laws / Anti-Bullshit Engineering |
| RFC-020-MAIN-SPINE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-020-MAIN-SPINE.md` | RFC-020: Main Spine / Graph-as-Memory Execution Model |
| RFC-030-FULL-ETL-PIPELINE.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-030-FULL-ETL-PIPELINE.md` | RFC-030: Full ETL Pipeline / Custody-First Ingestion |
| RFC-040-DEV-LIBRARY.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-040-DEV-LIBRARY.md` | RFC-040: Dev Library / Reusable Parts Without Authority Leakage |
| RFC-050-DIOGENES-KERNEL.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-050-DIOGENES-KERNEL.md` | RFC-050: Diogenes Kernel / Command Authority |
| RFC-060-ABSURD-WORKFLOWS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-060-ABSURD-WORKFLOWS.md` | RFC-060: ABSURD Workflows / Durable Postgres Work |
| RFC-070-KRAMPUS-KORPUS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-070-KRAMPUS-KORPUS.md` | RFC-070: Krampus Express / KORPUS Port |
| RFC-080-LOCAL-LLM-FABRIC.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-080-LOCAL-LLM-FABRIC.md` | RFC-080: Local RAM / VRAM / LLM Fabric |
| RFC-090-PERCYPHONAI.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-090-PERCYPHONAI.md` | RFC-090: PercyphonAI |
| RFC-100-ARTIFACT-TEMPLATES.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-100-ARTIFACT-TEMPLATES.md` | RFC-100: Artifact Generation Templates |
| RFC-110-INPUT-MULTIPLEXING.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-110-INPUT-MULTIPLEXING.md` | RFC-110: Input Multiplexing / Hot Lane and Slow Lane |
| RFC-120-OUTPUT-HYPERPLEXING.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-120-OUTPUT-HYPERPLEXING.md` | RFC-120: Output Hyperplexing / Sequencing of Language |
| RFC-130-INDY-READS.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-130-INDY-READS.md` | RFC-130: INDY_READs Teammate Model |
| RFC-140-CONSTANT-LEARNING.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-140-CONSTANT-LEARNING.md` | RFC-140: Constant Learning / River, Bytewax, GLiNER, LoRA, SONA, RuVector |
| RFC-150-BOARD-GAME-LAB.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-150-BOARD-GAME-LAB.md` | RFC-150: Board-Game / Simulation Lab |
| RFC-160-AUTOMATION.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-160-AUTOMATION.md` | RFC-160: Automation / Scheduler / Watchers |
| RFC-170-ACTIVE-ONTOLOGY.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-170-ACTIVE-ONTOLOGY.md` | RFC-170: Active Ontology / GO-25 / OBJECT-EVENT-EDGE |
| RFC-180-ABBA63.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-180-ABBA63.md` | RFC-180: ABBA63 / Abductive-Not-Credulous Heuristics |
| RFC-190-SELF-SOVEREIGNTY-OSINT.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-190-SELF-SOVEREIGNTY-OSINT.md` | RFC-190: Self-Sovereignty / OSINT Domain Argument |
| RFC-200-FILESYSTEM-LAW.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC-200-FILESYSTEM-LAW.md` | RFC-200: Filesystem Organization / Archive / Promotion Law |
| RFC_SUBJECT_REGISTRY.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/RFC_SUBJECT_REGISTRY.json` | UNKNOWN — needs operator label; filename suggests: RFC SUBJECT REGISTRY |
| SOURCES.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RFCS/SOURCES.md` | RFC Source Bibliography Seed |
| RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md` | Ruvector Concept Capture for LUCIDOTA ABSURD / RiverML / SONA |
| SCRIPTS_MANIFEST_V2.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/SCRIPTS_MANIFEST_V2.md` | LUCIDOTA Scripts Manifest v2 |
| STATUS_LEDGER.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/STATUS_LEDGER.md` | LUCIDOTA STATUS LEDGER |
| TICKLETRUNK.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/TICKLETRUNK.json` | UNKNOWN — needs operator label; filename suggests: TICKLETRUNK |
| TICKLETRUNK.md | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/TICKLETRUNK.md` | UNKNOWN — needs operator label; filename suggests: TICKLETRUNK |
| canonical_graph_write_allowlist.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/canonical_graph_write_allowlist.json` | UNKNOWN — needs operator label; filename suggests: canonical graph write allowlist |
| claude_code_claw_runtime_registry.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/claude_code_claw_runtime_registry.json` | UNKNOWN — needs operator label; filename suggests: claude code claw runtime registry |
| gpu_model_runtime_registry.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/gpu_model_runtime_registry.json` | UNKNOWN — needs operator label; filename suggests: gpu model runtime registry |
| instruction_authority_registry.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/instruction_authority_registry.json` | UNKNOWN — needs operator label; filename suggests: instruction authority registry |
| operator_graph_eligibility_policy.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/operator_graph_eligibility_policy.json` | UNKNOWN — needs operator label; filename suggests: operator graph eligibility policy |
| rust_port_candidacy_registry.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/rust_port_candidacy_registry.json` | UNKNOWN — needs operator label; filename suggests: rust port candidacy registry |
| spine_authority_registry.json | workflow | unknown | reusable_prior | `00_PROJECT_BRAIN/spine_authority_registry.json` | UNKNOWN — needs operator label; filename suggests: spine authority registry |

## REPOS

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 01_REPOS | repo_tool | sandbox | candidate_for_promotion | `01_REPOS` | Source Repositories |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/PocketFlow/README.md` | <div align="center"> |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/PocketFlow/cookbook/README.md` | Pocket Flow Cookbook |
| setup.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/PocketFlow/setup.py` | UNKNOWN — needs operator label; filename suggests: setup |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/README.md` | Source Repositories |
| CLAW.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/CLAW.md` | CLAW.md |
| MODBIBLE.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/MODBIBLE.md` | MODBIBLE — ClaudeCode / Claw Code Total Overhaul Bible |
| PARITY.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/PARITY.md` | PARITY GAP ANALYSIS |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/README.md` | ClaudeCode (Open Source) |
| Cargo.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/rust/Cargo.toml` | UNKNOWN — needs operator label; filename suggests: Cargo |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/claudecode/rust/README.md` | Claw Code |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/cybercrafter_drf/README.md` | Deterministic Reasoning Framework (DRF) |
| drf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/cybercrafter_drf/drf.py` | UNKNOWN — needs operator label; filename suggests: drf |
| requirements.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/cybercrafter_drf/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| PRIMER.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/doggystyle/PRIMER.md` | CKDOG1 Primer |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/doggystyle/README.md` | CKDOG1 V3 MVP |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/doggystyle/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| CMakeLists.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/CMakeLists.txt` | UNKNOWN — needs operator label; filename suggests: CMakeLists |
| CONTRIBUTING.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/CONTRIBUTING.md` | Acknowledgements |
| LICENSE.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/LICENSE.txt` | UNKNOWN — needs operator label; filename suggests: LICENSE |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/README.md` | [ncnn](https://raw.githubusercontent.com/Tencent/ncnn/master/images/256-ncnn.png) |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/benchmark/README.md` | UNKNOWN — needs operator label; filename suggests: README |
| build.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/build.sh` | android armv7 without neon |
| codeformat.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/codeformat.sh` | we run clang-format and astyle twice to get stable format output |
| package.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/package.sh` | NAME=ncnn |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/python/README.md` | ncnn |
| setup.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/flywheel1412_ncnn/setup.py` | UNKNOWN — needs operator label; filename suggests: setup |
| AGENTS.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/AGENTS.md` | Instructions for llama.cpp |
| CLAUDE.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/CLAUDE.md` | IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work. |
| CMakeLists.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/CMakeLists.txt` | UNKNOWN — needs operator label; filename suggests: CMakeLists |
| CMakePresets.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/CMakePresets.json` | UNKNOWN — needs operator label; filename suggests: CMakePresets |
| CONTRIBUTING.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/CONTRIBUTING.md` | Contributors |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/README.md` | llama.cpp |
| SECURITY.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/SECURITY.md` | Security Policy |
| build-xcframework.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/build-xcframework.sh` | Options |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/ci/README.md` | CI |
| convert_hf_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/convert_hf_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert hf to gguf |
| convert_hf_to_gguf_update.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/convert_hf_to_gguf_update.py` | UNKNOWN — needs operator label; filename suggests: convert hf to gguf update |
| convert_llama_ggml_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/convert_llama_ggml_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert llama ggml to gguf |
| convert_lora_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/convert_lora_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert lora to gguf |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/gguf-py/README.md` | gguf |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/gguf-py/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/grammars/README.md` | GBNF Guide |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| pyrightconfig.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/pyrightconfig.json` | UNKNOWN — needs operator label; filename suggests: pyrightconfig |
| requirements.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| ty.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llama.cpp/ty.toml` | UNKNOWN — needs operator label; filename suggests: ty |
| ARCHITECTURE.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llm-router/ARCHITECTURE.md` | LLM Workflow Router --- Architecture |
| LICENSE-COMMERCIAL.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llm-router/LICENSE-COMMERCIAL.txt` | UNKNOWN — needs operator label; filename suggests: LICENSE COMMERCIAL |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llm-router/README.md` | LLM Workflow Router |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llm-router/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| AGENTS.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/AGENTS.md` | Completion Checklist |
| CHANGELOG.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/CHANGELOG.md` | Changelog |
| CONTRIBUTING.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/CONTRIBUTING.md` | How to Contribute |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/README.md` | <h1> |
| README_CN.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/README_CN.md` | LLxprt Code |
| ROADMAP.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/ROADMAP.md` | LLxprt Code Roadmap |
| esbuild.config.js | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/esbuild.config.js` | @license |
| eslint.config.js | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/eslint.config.js` | @license |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/evals/README.md` | LLxprt Behavioral Evals |
| package-lock.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/package-lock.json` | UNKNOWN — needs operator label; filename suggests: package lock |
| package.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/package.json` | UNKNOWN — needs operator label; filename suggests: package |
| tsconfig.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/llxprt-code/tsconfig.json` | UNKNOWN — needs operator label; filename suggests: tsconfig |
| Cargo.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/lucidota_etl/Cargo.toml` | UNKNOWN — needs operator label; filename suggests: Cargo |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/lucidota_etl/README.md` | LUCIDOTA Rust Hot-Path MVP |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/needle/README.md` | Needle |
| launch_train.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/needle/launch_train.sh` | bin/bash |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/needle/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| requirements.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/needle/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| setup | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/needle/setup` | UNKNOWN — needs operator label; filename suggests: setup |
| AGENTS.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/AGENTS.md` | Instructions for llama.cpp |
| CLAUDE.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/CLAUDE.md` | IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work. |
| CMakeLists.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/CMakeLists.txt` | UNKNOWN — needs operator label; filename suggests: CMakeLists |
| CMakePresets.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/CMakePresets.json` | UNKNOWN — needs operator label; filename suggests: CMakePresets |
| CONTRIBUTING.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/CONTRIBUTING.md` | Contributors |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/README.md` | llama.cpp |
| SECURITY.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/SECURITY.md` | Security Policy |
| build-xcframework.sh | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/build-xcframework.sh` | Options |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/ci/README.md` | CI |
| convert_hf_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/convert_hf_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert hf to gguf |
| convert_hf_to_gguf_update.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/convert_hf_to_gguf_update.py` | UNKNOWN — needs operator label; filename suggests: convert hf to gguf update |
| convert_llama_ggml_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/convert_llama_ggml_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert llama ggml to gguf |
| convert_lora_to_gguf.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/convert_lora_to_gguf.py` | UNKNOWN — needs operator label; filename suggests: convert lora to gguf |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/gguf-py/README.md` | gguf |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/gguf-py/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/grammars/README.md` | GBNF Guide |
| pyproject.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/pyproject.toml` | UNKNOWN — needs operator label; filename suggests: pyproject |
| pyrightconfig.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/pyrightconfig.json` | UNKNOWN — needs operator label; filename suggests: pyrightconfig |
| requirements.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| ty.toml | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/prismml_llama.cpp/ty.toml` | UNKNOWN — needs operator label; filename suggests: ty |
| gemini-code-1779083240490.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779083240490.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779083240490 |
| gemini-code-1779083502852.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779083502852.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779083502852 |
| gemini-code-1779083616000.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779083616000.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779083616000 |
| gemini-code-1779083633812.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779083633812.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779083633812 |
| gemini-code-1779084568395.json | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779084568395.json` | UNKNOWN — needs operator label; filename suggests: gemini code 1779084568395 |
| gemini-code-1779084572530.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779084572530.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779084572530 |
| gemini-code-1779084576822.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779084576822.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779084576822 |
| gemini-code-1779084582363.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sharksnailmangame/gemini-code-1779084582363.py` | UNKNOWN — needs operator label; filename suggests: gemini code 1779084582363 |
| CHANGELOG.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/CHANGELOG.md` | Changelog |
| CONTRIBUTING.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/CONTRIBUTING.md` | Contributing to Syd |
| DEPLOYMENT_CHECKLIST.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/DEPLOYMENT_CHECKLIST.md` | Deployment Checklist |
| INSTALL.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/INSTALL.md` | Installation Guide |
| PUSH_TO_GITHUB.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/PUSH_TO_GITHUB.txt` | UNKNOWN — needs operator label; filename suggests: PUSH TO GITHUB |
| QUICKSTART.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/QUICKSTART.md` | Quick Start - 3 Steps |
| README.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/README.md` | Syd - Offline Penetration Testing Assistant |
| READY_TO_LAUNCH.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/READY_TO_LAUNCH.md` | Syd is Ready to Launch |
| RELEASE_READY.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/RELEASE_READY.md` | Release Preparation Complete |
| SUPPORT.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/SUPPORT.md` | Support |
| UTILITIES.md | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/UTILITIES.md` | Utility Scripts |
| bloodhound_analyzer.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/bloodhound_analyzer.py` | BloodHound Active Directory Analyzer Detects and analyzes pasted BloodHound output (JSON, ZIP, Cypher queries, paths) |
| bloodhound_fact_extractor.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/bloodhound_fact_extractor.py` | BloodHound Fact Extractor - Deterministic Pattern Extraction Mirrors the Nmap fact extractor architecture for consistent anti-hallucination validation. This module extracts hard facts from BloodHound JSON data and converts them to Q&A forma |
| chunk_and_embed_bloodhound.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/chunk_and_embed_bloodhound.py` | Chunk and Embed BloodHound Knowledge Base Converts BloodHound knowledge JSON into FAISS index with metadata |
| chunk_and_embed_volatility.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/chunk_and_embed_volatility.py` | Chunk and Embed Volatility Knowledge Base Converts Opus-generated JSON into FAISS index with metadata |
| fix_all_faiss_indexes.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/fix_all_faiss_indexes.py` | Fix ALL FAISS indexes (Nmap, BloodHound, Volatility) by regenerating with proper CPU embeddings This fixes the "Cannot copy out of meta tensor" error |
| nmap_fact_extractor.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/nmap_fact_extractor.py` | Deterministic Nmap Fact Extractor (Stage A) Parses Nmap output into structured facts - 100% accurate, no AI |
| requirements.txt | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/requirements.txt` | UNKNOWN — needs operator label; filename suggests: requirements |
| setup.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/setup.py` | Syd Setup Script Installs all dependencies and downloads model No HuggingFace account required |
| syd.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/syd.py` | UNKNOWN — needs operator label; filename suggests: syd |
| volatility_analyzer.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/volatility_analyzer.py` | Volatility 3 Output Parser with Intelligent Threat Detection Parses output from common Volatility plugins and detects ransomware, malware, C2, and credential theft |
| volatility_fact_extractor.py | repo_tool | sandbox | candidate_for_promotion | `01_REPOS/sydsec_syd/volatility_fact_extractor.py` | Deterministic Volatility 3 Fact Extractor (Stage A) Parses Volatility 3 output into structured facts - 100% accurate, no AI Mirrors Nmap/BloodHound fact extractor architecture |

## RUNTIME

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 04_RUNTIME | runtime_asset | unknown | reference | `04_RUNTIME` | Local runtime placeholder. Database files, service state, caches, sockets, and temporary execution outputs stay uncommitted unless explicitly promoted. |
| README.md | runtime_asset | unknown | reference | `04_RUNTIME/README.md` | Runtime |
| local_audit_block_0001_prompt.txt | runtime_asset | unknown | reference | `04_RUNTIME/absurd_abductive/local_audit_block_0001_prompt.txt` | UNKNOWN — needs operator label; filename suggests: local audit block 0001 prompt |
| absurd_probe_2026-05-19.txt | runtime_asset | unknown | reference | `04_RUNTIME/absurd_ingest_probe/absurd_probe_2026-05-19.txt` | UNKNOWN — needs operator label; filename suggests: absurd probe 2026 05 19 |
| absurd_prompt_vram_probe_2026-05-19.txt | runtime_asset | unknown | reference | `04_RUNTIME/absurd_ingest_probe/absurd_prompt_vram_probe_2026-05-19.txt` | UNKNOWN — needs operator label; filename suggests: absurd prompt vram probe 2026 05 19 |
| absurd_prompt_vram_probe_new_worker_2026-05-19.txt | runtime_asset | unknown | reference | `04_RUNTIME/absurd_ingest_probe/absurd_prompt_vram_probe_new_worker_2026-05-19.txt` | UNKNOWN — needs operator label; filename suggests: absurd prompt vram probe new worker 2026 05 19 |
| indy-reads.json | runtime_asset | unknown | reference | `04_RUNTIME/agentsi/agents/indy-reads.json` | UNKNOWN — needs operator label; filename suggests: indy reads |
| async_duckdb_footprint.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/bytewax_abductive_blender/async_duckdb_footprint.jsonl` | UNKNOWN — needs operator label; filename suggests: async duckdb footprint |
| simple_dev_order_4ce1a0e20f9e7a8167cd4c51e4319731694e1a1ff496367a0413d9f8ef40cde5.txt | runtime_asset | unknown | reference | `04_RUNTIME/dev_orders/original/simple_dev_order_4ce1a0e20f9e7a8167cd4c51e4319731694e1a1ff496367a0413d9f8ef40cde5.txt` | UNKNOWN — needs operator label; filename suggests: simple dev order 4ce1a0e20f9e7a8167cd4c51e4319731694e1a1ff496367a0413d9f8ef40cde5 |
| stdin_fixture_816217781dad823cf50e7d57f7bf9a326753bfeb891735a4a1020758f43bb2e5.txt | runtime_asset | unknown | reference | `04_RUNTIME/dev_orders/original/stdin_fixture_816217781dad823cf50e7d57f7bf9a326753bfeb891735a4a1020758f43bb2e5.txt` | UNKNOWN — needs operator label; filename suggests: stdin fixture 816217781dad823cf50e7d57f7bf9a326753bfeb891735a4a1020758f43bb2e5 |
| simple_dev_order.wrapped.txt | runtime_asset | unknown | reference | `04_RUNTIME/dev_orders/wrapped/simple_dev_order.wrapped.txt` | UNKNOWN — needs operator label; filename suggests: simple dev order.wrapped |
| ternary_router_heartbeat.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/fairyfuse/ternary_router_heartbeat.jsonl` | UNKNOWN — needs operator label; filename suggests: ternary router heartbeat |
| groq_worker_slice_audit.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/groq_worker_slice_audit.txt` | UNKNOWN — needs operator label; filename suggests: groq worker slice audit |
| groq_worker_slice_plan.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/groq_worker_slice_plan.txt` | UNKNOWN — needs operator label; filename suggests: groq worker slice plan |
| groq_worker_slice_redteam.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/groq_worker_slice_redteam.txt` | UNKNOWN — needs operator label; filename suggests: groq worker slice redteam |
| model_fabric_groq_20260526T063223564540Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063223564540Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063223564540Z |
| model_fabric_groq_20260526T063238218105Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063238218105Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063238218105Z |
| model_fabric_groq_20260526T063258826415Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063258826415Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063258826415Z |
| model_fabric_groq_20260526T063620627262Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063620627262Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063620627262Z |
| model_fabric_groq_20260526T063632447832Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063632447832Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063632447832Z |
| model_fabric_groq_20260526T063649503788Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T063649503788Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T063649503788Z |
| model_fabric_groq_20260526T064714019472Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T064714019472Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T064714019472Z |
| model_fabric_groq_20260526T064947858523Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T064947858523Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T064947858523Z |
| model_fabric_groq_20260526T065117542897Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T065117542897Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T065117542897Z |
| model_fabric_groq_20260526T070131550071Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T070131550071Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T070131550071Z |
| model_fabric_groq_20260526T203057424045Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T203057424045Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T203057424045Z |
| model_fabric_groq_20260526T203657290137Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T203657290137Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T203657290137Z |
| model_fabric_groq_20260526T203832372042Z.txt | runtime_asset | unknown | reference | `04_RUNTIME/goals/model_fabric_groq_20260526T203832372042Z.txt` | UNKNOWN — needs operator label; filename suggests: model fabric groq 20260526T203832372042Z |
| indy_percyphon_hunch_subtleknife_binding.json | runtime_asset | unknown | reference | `04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json` | UNKNOWN — needs operator label; filename suggests: indy percyphon hunch subtleknife binding |
| indy_reads_adapter_registry.json | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_adapter_registry.json` | UNKNOWN — needs operator label; filename suggests: indy reads adapter registry |
| 640e52a52ba4b4d3.txt | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_extracted/640e52a52ba4b4d3.txt` | UNKNOWN — needs operator label; filename suggests: 640e52a52ba4b4d3 |
| indy_reads_lora_stage.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_lora_stage.jsonl` | UNKNOWN — needs operator label; filename suggests: indy reads lora stage |
| indy_reads_persona_config.json | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_persona_config.json` | UNKNOWN — needs operator label; filename suggests: indy reads persona config |
| openai_codex_prompting_guide.md | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_sources/openai_codex_prompting_guide.md` | Codex** Prompting Guide |
| rtk_install.md | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_sources/rtk_install.md` | RTK Installation Guide - For AI Coding Assistants |
| rtk_readme.md | runtime_asset | unknown | reference | `04_RUNTIME/indy_reads_sources/rtk_readme.md` | <p align="center"> |
| dual_engine_receipts.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/inference_os/dual_engine_receipts.jsonl` | UNKNOWN — needs operator label; filename suggests: dual engine receipts |
| preemption_receipts.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/inference_os/preemption_receipts.jsonl` | UNKNOWN — needs operator label; filename suggests: preemption receipts |
| vram_llm_selection_bespoke_staging.json | runtime_asset | unknown | reference | `04_RUNTIME/inference_os/vram_llm_selection_bespoke_staging.json` | UNKNOWN — needs operator label; filename suggests: vram llm selection bespoke staging |
| ingestion_watchdog_done_count.txt | runtime_asset | unknown | reference | `04_RUNTIME/ingestion_watchdog_done_count.txt` | UNKNOWN — needs operator label; filename suggests: ingestion watchdog done count |
| state.sqlite | runtime_asset | unknown | reference | `04_RUNTIME/kernel_api_smoke/state.sqlite` | Reference asset/file: state.sqlite |
| korpus_result_stream_state.json | runtime_asset | unknown | reference | `04_RUNTIME/korpus_result_stream_state.json` | UNKNOWN — needs operator label; filename suggests: korpus result stream state |
| llxprt_hook_audit.jsonl | runtime_asset | unknown | reference | `04_RUNTIME/llxprt_hook_audit.jsonl` | UNKNOWN — needs operator label; filename suggests: llxprt hook audit |
| task_block_0001_9ff8ea4daf2c_local.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0001_9ff8ea4daf2c_local.txt` | UNKNOWN — needs operator label; filename suggests: task block 0001 9ff8ea4daf2c local |
| task_block_0001_local.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0001_local.txt` | UNKNOWN — needs operator label; filename suggests: task block 0001 local |
| task_block_0002_45a242f605e8_groq.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0002_45a242f605e8_groq.txt` | UNKNOWN — needs operator label; filename suggests: task block 0002 45a242f605e8 groq |
| task_block_0002_groq.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0002_groq.txt` | UNKNOWN — needs operator label; filename suggests: task block 0002 groq |
| task_block_0003_79bc2c7607c4_cohere.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0003_79bc2c7607c4_cohere.txt` | UNKNOWN — needs operator label; filename suggests: task block 0003 79bc2c7607c4 cohere |
| task_block_0003_cohere.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits/task_block_0003_cohere.txt` | UNKNOWN — needs operator label; filename suggests: task block 0003 cohere |
| task_block_0001_9ff8ea4daf2c_local_strict.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits_strict/task_block_0001_9ff8ea4daf2c_local_strict.txt` | UNKNOWN — needs operator label; filename suggests: task block 0001 9ff8ea4daf2c local strict |
| task_block_0002_45a242f605e8_groq_strict.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits_strict/task_block_0002_45a242f605e8_groq_strict.txt` | UNKNOWN — needs operator label; filename suggests: task block 0002 45a242f605e8 groq strict |
| task_block_0001_9ff8ea4daf2c_local_strict2.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits_strict2/task_block_0001_9ff8ea4daf2c_local_strict2.txt` | UNKNOWN — needs operator label; filename suggests: task block 0001 9ff8ea4daf2c local strict2 |
| task_block_0002_45a242f605e8_groq_strict2.txt | runtime_asset | unknown | reference | `04_RUNTIME/model_audits_strict2/task_block_0002_45a242f605e8_groq_strict2.txt` | UNKNOWN — needs operator label; filename suggests: task block 0002 45a242f605e8 groq strict2 |
| board_effect_doctrine_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/board_effect_doctrine_latest.json` | UNKNOWN — needs operator label; filename suggests: board effect doctrine latest |
| dev_journey_decision_points_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/dev_journey_decision_points_latest.json` | UNKNOWN — needs operator label; filename suggests: dev journey decision points latest |
| hunch_hypertimeline_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/hunch_hypertimeline_latest.json` | UNKNOWN — needs operator label; filename suggests: hunch hypertimeline latest |
| hunch_postgres_ingest_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/hunch_postgres_ingest_latest.json` | UNKNOWN — needs operator label; filename suggests: hunch postgres ingest latest |
| model_invocation_audit_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/model_invocation_audit_latest.json` | UNKNOWN — needs operator label; filename suggests: model invocation audit latest |
| project2501_model_workshare_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/project2501_model_workshare_latest.json` | UNKNOWN — needs operator label; filename suggests: project2501 model workshare latest |
| working_reality_latest.json | runtime_asset | unknown | reference | `04_RUNTIME/observation_center/working_reality_latest.json` | UNKNOWN — needs operator label; filename suggests: working reality latest |
| treelite_note_extracted.txt | runtime_asset | unknown | reference | `04_RUNTIME/treelite_note_extracted.txt` | UNKNOWN — needs operator label; filename suggests: treelite note extracted |

## VAULT

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| 03_VAULT | runtime_asset | unknown | reference | `03_VAULT` | Local Vault |
| README.md | runtime_asset | unknown | reference | `03_VAULT/README.md` | Local Vault |
| IMPORT_MANIFEST_2026_05_13.json | runtime_asset | unknown | reference | `03_VAULT/drive_pull/maths/IMPORT_MANIFEST_2026_05_13.json` | UNKNOWN — needs operator label; filename suggests: IMPORT MANIFEST 2026 05 13 |
| cards.min.json | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/cards.min.json` | UNKNOWN — needs operator label; filename suggests: cards.min |
| changelog.json | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/changelog.json` | UNKNOWN — needs operator label; filename suggests: changelog |
| errata.json | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/errata.json` | UNKNOWN — needs operator label; filename suggests: errata |
| faq.json | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/faq.json` | UNKNOWN — needs operator label; filename suggests: faq |
| main.84a1395bd2271d65.js | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/main.84a1395bd2271d65.js` | UNKNOWN — needs operator label; filename suggests: main.84a1395bd2271d65 |
| meta.json | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/meta.json` | UNKNOWN — needs operator label; filename suggests: meta |
| polyfills.4d95b91ae2d907c9.js | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/polyfills.4d95b91ae2d907c9.js` | "use strict";(self.webpackChunkapp=self.webpackChunkapp\|\|[]).push([[3461],{6935:()=>{!function(t){const n=t.performance;function i(L){n&&n.mark&&n.mark(L)}function o(L,T){n&&n.measure&&n.measure(L,T)}i("Zone");const c=t.__Zone_symbol_pref |
| runtime.729cd8b9244b5e65.js | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/runtime.729cd8b9244b5e65.js` | (()=>{"use strict";var e,v={},g={};function f(e){var r=g[e];if(void 0!==r)return r.exports;var a=g[e]={id:e,loaded:!1,exports:{}};return v[e].call(a.exports,a,a.exports,f),a.loaded=!0,a.exports}f.m=v,e=[],f.O=(r,a,c,b)=>{if(!a){var t=1/0;fo |
| styles.f7ae0b670f79e2b0.css | runtime_asset | unknown | reference | `03_VAULT/external/leder_cards/styles.f7ae0b670f79e2b0.css` | UNKNOWN — needs operator label; filename suggests: styles.f7ae0b670f79e2b0 |
| BRAIN.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/BRAIN.md` | LUCIDOTA Brain Snapshot |
| CLAW.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/CLAW.md` | CLAW.md |
| GO_GRAPH_STORAGE_PROMOTION_RULES.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/GO_GRAPH_STORAGE_PROMOTION_RULES.md` | DIOGENES / LUCIDOTA Step 6 — GO Graph Storage + Promotion Rules |
| GRAPH_IDENTITY_VERSIONING_RULES.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/GRAPH_IDENTITY_VERSIONING_RULES.md` | DIOGENES / LUCIDOTA Graph Identity + Versioning Rules |
| OFFICIAL_ONTOLOGY.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/OFFICIAL_ONTOLOGY.md` | OFFICIAL DIOGENES / LUCIDOTA ONTOLOGY |
| OFFICIAL_ONTOLOGY_EXTENSION.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/OFFICIAL_ONTOLOGY_EXTENSION.md` | Official GO Extension Terms — Investigation / Civic / Code |
| TOPOLOGY.md | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/TOPOLOGY.md` | LUCIDOTA Runtime Topology |
| manifest.json | runtime_asset | unknown | reference | `03_VAULT/ingested_markdown/20260514T223602Z/manifest.json` | UNKNOWN — needs operator label; filename suggests: manifest |
| cas_ingest.jsonl | runtime_asset | unknown | reference | `03_VAULT/journal/cas_ingest.jsonl` | UNKNOWN — needs operator label; filename suggests: cas ingest |
| 20260514T180525.drop_test.md | runtime_asset | unknown | reference | `03_VAULT/korpus_krampii/DIGESTED/20260514T180525.drop_test.md` | KRAMPUSCHEWING Drop Test |
| 20260514T195647.hardmath_drop_test.md | runtime_asset | unknown | reference | `03_VAULT/korpus_krampii/DIGESTED/20260514T195647.hardmath_drop_test.md` | Hard Math Drop Test |
| 20260514T215050.rechrono_smoke_20260213.md | runtime_asset | unknown | reference | `03_VAULT/korpus_krampii/DIGESTED/20260514T215050.rechrono_smoke_20260213.md` | date: 2026-02-13T09:30:00Z |
| krampus_dbstream_brain.20260519T221624Z.pkl | runtime_asset | unknown | reference | `03_VAULT/krampus_dbstream_brain.20260519T221624Z.pkl` | Reference asset/file: krampus_dbstream_brain.20260519T221624Z.pkl |
| krampus_dbstream_brain.pkl | runtime_asset | unknown | reference | `03_VAULT/krampus_dbstream_brain.pkl` | Reference asset/file: krampus_dbstream_brain.pkl |
| treelite_router_v0.tl | runtime_asset | unknown | reference | `03_VAULT/router/treelite_router_v0.tl` | Reference asset/file: treelite_router_v0.tl |

## OTHER

| name | kind | status | proof_hoard_role | path | what_it_does |
|---|---|---|---|---|---|
| TOOLS | other | unknown | unknown | `TOOLS` | TICKLETRUNK TOOLS ACCESS LAYER |
| README.md | other | unknown | unknown | `TOOLS/ALGOS/README.md` | TOOLS/ALGOS |
| README.md | other | unknown | unknown | `TOOLS/BOOKS/README.md` | TOOLS/BOOKS |
| README.md | other | unknown | unknown | `TOOLS/LORAS/README.md` | TOOLS/LORAS |
| README.md | other | unknown | unknown | `TOOLS/MODELS/README.md` | TOOLS/MODELS |
| README.md | other | unknown | unknown | `TOOLS/OTHER/README.md` | TOOLS/OTHER |
| README.md | other | unknown | unknown | `TOOLS/PLUGINS/README.md` | TOOLS/PLUGINS |
| README.md | other | unknown | unknown | `TOOLS/README.md` | TICKLETRUNK TOOLS ACCESS LAYER |
| README.md | other | unknown | unknown | `TOOLS/REPOS/README.md` | TOOLS/REPOS |
| README.md | other | unknown | unknown | `TOOLS/RUNTIME/README.md` | TOOLS/RUNTIME |
| README.md | other | unknown | unknown | `TOOLS/SCHEMAS/README.md` | TOOLS/SCHEMAS |
| README.md | other | unknown | unknown | `TOOLS/SCRAPERS/README.md` | TOOLS/SCRAPERS |
| README.md | other | unknown | unknown | `TOOLS/SCRIPTS/README.md` | TOOLS/SCRIPTS |
| README.md | other | unknown | unknown | `TOOLS/SERVICES/README.md` | TOOLS/SERVICES |
| README.md | other | unknown | unknown | `TOOLS/SKILLS/README.md` | TOOLS/SKILLS |
| README.md | other | unknown | unknown | `TOOLS/SURFACES/README.md` | TOOLS/SURFACES |
| README.md | other | unknown | unknown | `TOOLS/VAULT/README.md` | TOOLS/VAULT |
| README.md | other | unknown | unknown | `TOOLS/WORKFLOWS/README.md` | TOOLS/WORKFLOWS |
| src | other | unknown | unknown | `src` | UNKNOWN — needs operator label; filename suggests: src |
| db_core.py | other | unknown | unknown | `src/db_core.py` | UNKNOWN — needs operator label; filename suggests: db core |
| spine.py | other | unknown | unknown | `src/spine.py` | UNKNOWN — needs operator label; filename suggests: spine |
| tests | other | unknown | unknown | `tests` | UNKNOWN — needs operator label; filename suggests: tests |
| __init__.py | other | unknown | unknown | `tests/__init__.py` | UNKNOWN — needs operator label; filename suggests: init |
| 2026-05-17-note.md | other | unknown | unknown | `tests/fixtures/demo_corpus/2026-05-17-note.md` | Alice saw Evidence. Bob did not sign contract. |
| data.json | other | unknown | unknown | `tests/fixtures/demo_corpus/data.json` | UNKNOWN — needs operator label; filename suggests: data |
| simple_dev_order.txt | other | unknown | unknown | `tests/fixtures/dev_orders/simple_dev_order.txt` | UNKNOWN — needs operator label; filename suggests: simple dev order |
| valid_receipt_bundle.json | other | unknown | unknown | `tests/fixtures/golden_path/valid_receipt_bundle.json` | UNKNOWN — needs operator label; filename suggests: valid receipt bundle |
| missing_constraint_receipt.json | other | unknown | unknown | `tests/fixtures/matrix_trace/missing_constraint_receipt.json` | UNKNOWN — needs operator label; filename suggests: missing constraint receipt |
| pass_with_incomplete_matrix_receipt.json | other | unknown | unknown | `tests/fixtures/matrix_trace/pass_with_incomplete_matrix_receipt.json` | UNKNOWN — needs operator label; filename suggests: pass with incomplete matrix receipt |
| prose_only_matrix_receipt.json | other | unknown | unknown | `tests/fixtures/matrix_trace/prose_only_matrix_receipt.json` | UNKNOWN — needs operator label; filename suggests: prose only matrix receipt |
| valid_receipt.json | other | unknown | unknown | `tests/fixtures/matrix_trace/valid_receipt.json` | UNKNOWN — needs operator label; filename suggests: valid receipt |
| wrong_order_receipt.json | other | unknown | unknown | `tests/fixtures/matrix_trace/wrong_order_receipt.json` | UNKNOWN — needs operator label; filename suggests: wrong order receipt |
| generate_poison.py | other | unknown | unknown | `tests/generate_poison.py` | UNKNOWN — needs operator label; filename suggests: generate poison |
| 01_utf8_with_random_nul.txt | other | unknown | unknown | `tests/poison_drop/01_utf8_with_random_nul.txt` | UNKNOWN — needs operator label; filename suggests: 01 utf8 with random nul |
| 02_binary_copy_a.bin | other | unknown | unknown | `tests/poison_drop/02_binary_copy_a.bin` | Reference asset/file: 02_binary_copy_a.bin |
| 03_binary_copy_b.bin | other | unknown | unknown | `tests/poison_drop/03_binary_copy_b.bin` | Reference asset/file: 03_binary_copy_b.bin |
| 04_binary_copy_c.bin | other | unknown | unknown | `tests/poison_drop/04_binary_copy_c.bin` | Reference asset/file: 04_binary_copy_c.bin |
| 05_force_deadlock.py | other | unknown | unknown | `tests/poison_drop/05_force_deadlock.py` | UNKNOWN — needs operator label; filename suggests: 05 force deadlock |
| demo_product_snapshot.json | other | unknown | unknown | `tests/snapshots/demo_product_snapshot.json` | UNKNOWN — needs operator label; filename suggests: demo product snapshot |
| test_abductive_db_os_gate.py | other | unknown | unknown | `tests/test_abductive_db_os_gate.py` | UNKNOWN — needs operator label; filename suggests: test abductive db os gate |
| test_abductive_db_os_ledger.py | other | unknown | unknown | `tests/test_abductive_db_os_ledger.py` | UNKNOWN — needs operator label; filename suggests: test abductive db os ledger |
| test_abductive_next_move_engine.py | other | unknown | unknown | `tests/test_abductive_next_move_engine.py` | UNKNOWN — needs operator label; filename suggests: test abductive next move engine |
| test_absurd_abductive_ledger.py | other | unknown | unknown | `tests/test_absurd_abductive_ledger.py` | UNKNOWN — needs operator label; filename suggests: test absurd abductive ledger |
| test_absurd_chrono_worker_contract.py | other | unknown | unknown | `tests/test_absurd_chrono_worker_contract.py` | UNKNOWN — needs operator label; filename suggests: test absurd chrono worker contract |
| test_absurd_consume_kernel_authorization.py | other | unknown | unknown | `tests/test_absurd_consume_kernel_authorization.py` | UNKNOWN — needs operator label; filename suggests: test absurd consume kernel authorization |
| test_absurd_corpus_bridge.py | other | unknown | unknown | `tests/test_absurd_corpus_bridge.py` | UNKNOWN — needs operator label; filename suggests: test absurd corpus bridge |
| test_absurd_gate.py | other | unknown | unknown | `tests/test_absurd_gate.py` | UNKNOWN — needs operator label; filename suggests: test absurd gate |
| test_absurd_graph_promotion_worker_contract.py | other | unknown | unknown | `tests/test_absurd_graph_promotion_worker_contract.py` | UNKNOWN — needs operator label; filename suggests: test absurd graph promotion worker contract |
| test_absurd_intake_worker_contract.py | other | unknown | unknown | `tests/test_absurd_intake_worker_contract.py` | UNKNOWN — needs operator label; filename suggests: test absurd intake worker contract |
| test_absurd_momentary_flow.py | other | unknown | unknown | `tests/test_absurd_momentary_flow.py` | UNKNOWN — needs operator label; filename suggests: test absurd momentary flow |
| test_absurd_next_move_engine.py | other | unknown | unknown | `tests/test_absurd_next_move_engine.py` | UNKNOWN — needs operator label; filename suggests: test absurd next move engine |
| test_absurd_queue_spine_contract.py | other | unknown | unknown | `tests/test_absurd_queue_spine_contract.py` | UNKNOWN — needs operator label; filename suggests: test absurd queue spine contract |
| test_absurd_real_work_loop_bootstrap.py | other | unknown | unknown | `tests/test_absurd_real_work_loop_bootstrap.py` | UNKNOWN — needs operator label; filename suggests: test absurd real work loop bootstrap |
| test_absurd_remaining_worker_contract_alignment.py | other | unknown | unknown | `tests/test_absurd_remaining_worker_contract_alignment.py` | UNKNOWN — needs operator label; filename suggests: test absurd remaining worker contract alignment |
| test_absurd_river_worker_contract.py | other | unknown | unknown | `tests/test_absurd_river_worker_contract.py` | UNKNOWN — needs operator label; filename suggests: test absurd river worker contract |
| test_absurd_river_worker_hard_fail.py | other | unknown | unknown | `tests/test_absurd_river_worker_hard_fail.py` | UNKNOWN — needs operator label; filename suggests: test absurd river worker hard fail |
| test_activity_tree_ingest_dry_run.py | other | unknown | unknown | `tests/test_activity_tree_ingest_dry_run.py` | UNKNOWN — needs operator label; filename suggests: test activity tree ingest dry run |
| test_adhd_slow_lane_divergence.py | other | unknown | unknown | `tests/test_adhd_slow_lane_divergence.py` | UNKNOWN — needs operator label; filename suggests: test adhd slow lane divergence |
| test_anthropic_proxy_local.py | other | unknown | unknown | `tests/test_anthropic_proxy_local.py` | UNKNOWN — needs operator label; filename suggests: test anthropic proxy local |
| test_bare_steel_doctrine.py | other | unknown | unknown | `tests/test_bare_steel_doctrine.py` | UNKNOWN — needs operator label; filename suggests: test bare steel doctrine |
| test_bitloops_airlock_audit.py | other | unknown | unknown | `tests/test_bitloops_airlock_audit.py` | UNKNOWN — needs operator label; filename suggests: test bitloops airlock audit |
| test_bitloops_automation_loop.py | other | unknown | unknown | `tests/test_bitloops_automation_loop.py` | UNKNOWN — needs operator label; filename suggests: test bitloops automation loop |
| test_bitloops_full_reingest_manifest.py | other | unknown | unknown | `tests/test_bitloops_full_reingest_manifest.py` | UNKNOWN — needs operator label; filename suggests: test bitloops full reingest manifest |
| test_blueprint_first_pseudolaw.py | other | unknown | unknown | `tests/test_blueprint_first_pseudolaw.py` | UNKNOWN — needs operator label; filename suggests: test blueprint first pseudolaw |
| test_board_effect_doctrine.py | other | unknown | unknown | `tests/test_board_effect_doctrine.py` | UNKNOWN — needs operator label; filename suggests: test board effect doctrine |
| test_boring_beast_runtime.py | other | unknown | unknown | `tests/test_boring_beast_runtime.py` | UNKNOWN — needs operator label; filename suggests: test boring beast runtime |
| test_bytewax_treelite_matrix.py | other | unknown | unknown | `tests/test_bytewax_treelite_matrix.py` | UNKNOWN — needs operator label; filename suggests: test bytewax treelite matrix |
| test_canonical_graph_write_scanner.py | other | unknown | unknown | `tests/test_canonical_graph_write_scanner.py` | UNKNOWN — needs operator label; filename suggests: test canonical graph write scanner |
| test_case_dashboard_data.py | other | unknown | unknown | `tests/test_case_dashboard_data.py` | UNKNOWN — needs operator label; filename suggests: test case dashboard data |
| test_case_packet_renderer.py | other | unknown | unknown | `tests/test_case_packet_renderer.py` | UNKNOWN — needs operator label; filename suggests: test case packet renderer |
| test_case_workspace_isolation.py | other | unknown | unknown | `tests/test_case_workspace_isolation.py` | UNKNOWN — needs operator label; filename suggests: test case workspace isolation |
| test_chrono_conservation.py | other | unknown | unknown | `tests/test_chrono_conservation.py` | UNKNOWN — needs operator label; filename suggests: test chrono conservation |
| test_chrono_source_trust_validator.py | other | unknown | unknown | `tests/test_chrono_source_trust_validator.py` | UNKNOWN — needs operator label; filename suggests: test chrono source trust validator |
| test_claim_clusterer.py | other | unknown | unknown | `tests/test_claim_clusterer.py` | UNKNOWN — needs operator label; filename suggests: test claim clusterer |
| test_claim_support_score.py | other | unknown | unknown | `tests/test_claim_support_score.py` | UNKNOWN — needs operator label; filename suggests: test claim support score |
| test_content_store.py | other | unknown | unknown | `tests/test_content_store.py` | UNKNOWN — needs operator label; filename suggests: test content store |
| test_contradiction_report.py | other | unknown | unknown | `tests/test_contradiction_report.py` | UNKNOWN — needs operator label; filename suggests: test contradiction report |
| test_contradiction_resolution.py | other | unknown | unknown | `tests/test_contradiction_resolution.py` | UNKNOWN — needs operator label; filename suggests: test contradiction resolution |
| test_decision_gated_graph_promotion.py | other | unknown | unknown | `tests/test_decision_gated_graph_promotion.py` | UNKNOWN — needs operator label; filename suggests: test decision gated graph promotion |
| test_decision_gated_memory.py | other | unknown | unknown | `tests/test_decision_gated_memory.py` | UNKNOWN — needs operator label; filename suggests: test decision gated memory |
| test_demo_product_snapshot.py | other | unknown | unknown | `tests/test_demo_product_snapshot.py` | UNKNOWN — needs operator label; filename suggests: test demo product snapshot |
| test_dev_journey_decision_points.py | other | unknown | unknown | `tests/test_dev_journey_decision_points.py` | UNKNOWN — needs operator label; filename suggests: test dev journey decision points |
| test_dev_library_scan_wrapper.py | other | unknown | unknown | `tests/test_dev_library_scan_wrapper.py` | UNKNOWN — needs operator label; filename suggests: test dev library scan wrapper |
| test_dev_order_gate.py | other | unknown | unknown | `tests/test_dev_order_gate.py` | UNKNOWN — needs operator label; filename suggests: test dev order gate |
| test_dev_order_matrix_wrapper.py | other | unknown | unknown | `tests/test_dev_order_matrix_wrapper.py` | UNKNOWN — needs operator label; filename suggests: test dev order matrix wrapper |
| test_diogenes_memory_gate.py | other | unknown | unknown | `tests/test_diogenes_memory_gate.py` | UNKNOWN — needs operator label; filename suggests: test diogenes memory gate |
| test_entity_candidate_registry.py | other | unknown | unknown | `tests/test_entity_candidate_registry.py` | UNKNOWN — needs operator label; filename suggests: test entity candidate registry |
| test_event_candidate_registry.py | other | unknown | unknown | `tests/test_event_candidate_registry.py` | UNKNOWN — needs operator label; filename suggests: test event candidate registry |
| test_export_bundle.py | other | unknown | unknown | `tests/test_export_bundle.py` | UNKNOWN — needs operator label; filename suggests: test export bundle |
| test_fast_slow_lane_gate.py | other | unknown | unknown | `tests/test_fast_slow_lane_gate.py` | UNKNOWN — needs operator label; filename suggests: test fast slow lane gate |
| test_full_system_soak_audit.py | other | unknown | unknown | `tests/test_full_system_soak_audit.py` | UNKNOWN — needs operator label; filename suggests: test full system soak audit |
| test_goal_agent_packet.py | other | unknown | unknown | `tests/test_goal_agent_packet.py` | UNKNOWN — needs operator label; filename suggests: test goal agent packet |
| test_goal_chain_telemetry.py | other | unknown | unknown | `tests/test_goal_chain_telemetry.py` | UNKNOWN — needs operator label; filename suggests: test goal chain telemetry |
| test_goal_dev_control.py | other | unknown | unknown | `tests/test_goal_dev_control.py` | UNKNOWN — needs operator label; filename suggests: test goal dev control |
| test_goal_handoff.py | other | unknown | unknown | `tests/test_goal_handoff.py` | UNKNOWN — needs operator label; filename suggests: test goal handoff |
| test_goal_model_fabric_control.py | other | unknown | unknown | `tests/test_goal_model_fabric_control.py` | UNKNOWN — needs operator label; filename suggests: test goal model fabric control |
| test_goal_model_fabric_orchestrate.py | other | unknown | unknown | `tests/test_goal_model_fabric_orchestrate.py` | UNKNOWN — needs operator label; filename suggests: test goal model fabric orchestrate |
| test_goal_scenario_batch.py | other | unknown | unknown | `tests/test_goal_scenario_batch.py` | UNKNOWN — needs operator label; filename suggests: test goal scenario batch |
| test_goal_scenario_compare.py | other | unknown | unknown | `tests/test_goal_scenario_compare.py` | UNKNOWN — needs operator label; filename suggests: test goal scenario compare |
| test_goal_scenario_holdout.py | other | unknown | unknown | `tests/test_goal_scenario_holdout.py` | UNKNOWN — needs operator label; filename suggests: test goal scenario holdout |
| test_goal_swarm_brief.py | other | unknown | unknown | `tests/test_goal_swarm_brief.py` | UNKNOWN — needs operator label; filename suggests: test goal swarm brief |
| test_goal_swarm_dispatch.py | other | unknown | unknown | `tests/test_goal_swarm_dispatch.py` | UNKNOWN — needs operator label; filename suggests: test goal swarm dispatch |
| test_golden_path_dry_run.py | other | unknown | unknown | `tests/test_golden_path_dry_run.py` | UNKNOWN — needs operator label; filename suggests: test golden path dry run |
| test_golden_path_regression_gate.py | other | unknown | unknown | `tests/test_golden_path_regression_gate.py` | UNKNOWN — needs operator label; filename suggests: test golden path regression gate |
| test_graph_materialization_command_policy.py | other | unknown | unknown | `tests/test_graph_materialization_command_policy.py` | UNKNOWN — needs operator label; filename suggests: test graph materialization command policy |
| test_graph_promotion_gate_safety.py | other | unknown | unknown | `tests/test_graph_promotion_gate_safety.py` | UNKNOWN — needs operator label; filename suggests: test graph promotion gate safety |
| test_graph_symbol_compare.py | other | unknown | unknown | `tests/test_graph_symbol_compare.py` | UNKNOWN — needs operator label; filename suggests: test graph symbol compare |
| test_graph_symbol_condensation.py | other | unknown | unknown | `tests/test_graph_symbol_condensation.py` | UNKNOWN — needs operator label; filename suggests: test graph symbol condensation |
| test_graph_symbol_dispatch.py | other | unknown | unknown | `tests/test_graph_symbol_dispatch.py` | UNKNOWN — needs operator label; filename suggests: test graph symbol dispatch |
| test_groq_goal_delegate.py | other | unknown | unknown | `tests/test_groq_goal_delegate.py` | UNKNOWN — needs operator label; filename suggests: test groq goal delegate |
| test_hunch_hypertimeline_audit.py | other | unknown | unknown | `tests/test_hunch_hypertimeline_audit.py` | UNKNOWN — needs operator label; filename suggests: test hunch hypertimeline audit |
| test_hunch_postgres_ingest.py | other | unknown | unknown | `tests/test_hunch_postgres_ingest.py` | UNKNOWN — needs operator label; filename suggests: test hunch postgres ingest |
| test_hunch_subtleknife_binding.py | other | unknown | unknown | `tests/test_hunch_subtleknife_binding.py` | UNKNOWN — needs operator label; filename suggests: test hunch subtleknife binding |
| test_import_export_bundle.py | other | unknown | unknown | `tests/test_import_export_bundle.py` | UNKNOWN — needs operator label; filename suggests: test import export bundle |
| test_import_policy.py | other | unknown | unknown | `tests/test_import_policy.py` | UNKNOWN — needs operator label; filename suggests: test import policy |
| test_indy_book_learning_pipeline.py | other | unknown | unknown | `tests/test_indy_book_learning_pipeline.py` | UNKNOWN — needs operator label; filename suggests: test indy book learning pipeline |
| test_indy_reads_safe_books_batch.py | other | unknown | unknown | `tests/test_indy_reads_safe_books_batch.py` | UNKNOWN — needs operator label; filename suggests: test indy reads safe books batch |
| test_instruction_authority_registry.py | other | unknown | unknown | `tests/test_instruction_authority_registry.py` | UNKNOWN — needs operator label; filename suggests: test instruction authority registry |
| test_instruction_conflict_scanner.py | other | unknown | unknown | `tests/test_instruction_conflict_scanner.py` | UNKNOWN — needs operator label; filename suggests: test instruction conflict scanner |
| test_kernel_authority_spine.py | other | unknown | unknown | `tests/test_kernel_authority_spine.py` | UNKNOWN — needs operator label; filename suggests: test kernel authority spine |
| test_kernel_authorized_pipeline_submission.py | other | unknown | unknown | `tests/test_kernel_authorized_pipeline_submission.py` | UNKNOWN — needs operator label; filename suggests: test kernel authorized pipeline submission |
| test_kernel_control_packet.py | other | unknown | unknown | `tests/test_kernel_control_packet.py` | UNKNOWN — needs operator label; filename suggests: test kernel control packet |
| test_knowledge_library_check.py | other | unknown | unknown | `tests/test_knowledge_library_check.py` | UNKNOWN — needs operator label; filename suggests: test knowledge library check |
| test_krampuschewing_graph_stage.py | other | unknown | unknown | `tests/test_krampuschewing_graph_stage.py` | UNKNOWN — needs operator label; filename suggests: test krampuschewing graph stage |
| test_language_router.py | other | unknown | unknown | `tests/test_language_router.py` | UNKNOWN — needs operator label; filename suggests: test language router |
| test_llxprt_project2501.py | other | unknown | unknown | `tests/test_llxprt_project2501.py` | UNKNOWN — needs operator label; filename suggests: test llxprt project2501 |
| test_local_bonsai_ternary_wiring.py | other | unknown | unknown | `tests/test_local_bonsai_ternary_wiring.py` | UNKNOWN — needs operator label; filename suggests: test local bonsai ternary wiring |
| test_local_deterministic_audit.py | other | unknown | unknown | `tests/test_local_deterministic_audit.py` | UNKNOWN — needs operator label; filename suggests: test local deterministic audit |
| test_local_model_chat_cli.py | other | unknown | unknown | `tests/test_local_model_chat_cli.py` | UNKNOWN — needs operator label; filename suggests: test local model chat cli |
| test_lucidota_acceptance.py | other | unknown | unknown | `tests/test_lucidota_acceptance.py` | UNKNOWN — needs operator label; filename suggests: test lucidota acceptance |
| test_lucidota_ci_gate.py | other | unknown | unknown | `tests/test_lucidota_ci_gate.py` | UNKNOWN — needs operator label; filename suggests: test lucidota ci gate |
| test_lucidota_cli.py | other | unknown | unknown | `tests/test_lucidota_cli.py` | UNKNOWN — needs operator label; filename suggests: test lucidota cli |
| test_lucidota_goal_audit.py | other | unknown | unknown | `tests/test_lucidota_goal_audit.py` | UNKNOWN — needs operator label; filename suggests: test lucidota goal audit |
| test_lucidota_ouroboros_loop.py | other | unknown | unknown | `tests/test_lucidota_ouroboros_loop.py` | UNKNOWN — needs operator label; filename suggests: test lucidota ouroboros loop |
| test_lucidota_pipeline_api.py | other | unknown | unknown | `tests/test_lucidota_pipeline_api.py` | UNKNOWN — needs operator label; filename suggests: test lucidota pipeline api |
| test_lucidota_pipeline_synthesis.py | other | unknown | unknown | `tests/test_lucidota_pipeline_synthesis.py` | UNKNOWN — needs operator label; filename suggests: test lucidota pipeline synthesis |
| test_lucidota_progress.py | other | unknown | unknown | `tests/test_lucidota_progress.py` | UNKNOWN — needs operator label; filename suggests: test lucidota progress |
| test_lucidota_release_gate.py | other | unknown | unknown | `tests/test_lucidota_release_gate.py` | UNKNOWN — needs operator label; filename suggests: test lucidota release gate |
| test_lucidota_runtime_smoke.py | other | unknown | unknown | `tests/test_lucidota_runtime_smoke.py` | UNKNOWN — needs operator label; filename suggests: test lucidota runtime smoke |
| test_lucidota_swarm_router.py | other | unknown | unknown | `tests/test_lucidota_swarm_router.py` | UNKNOWN — needs operator label; filename suggests: test lucidota swarm router |
| test_lucidota_synthesis_pass.py | other | unknown | unknown | `tests/test_lucidota_synthesis_pass.py` | UNKNOWN — needs operator label; filename suggests: test lucidota synthesis pass |
| test_lucidota_usecase_proof.py | other | unknown | unknown | `tests/test_lucidota_usecase_proof.py` | UNKNOWN — needs operator label; filename suggests: test lucidota usecase proof |
| test_matrix_trace_checker.py | other | unknown | unknown | `tests/test_matrix_trace_checker.py` | UNKNOWN — needs operator label; filename suggests: test matrix trace checker |
| test_mega_gate_v2_repairs.py | other | unknown | unknown | `tests/test_mega_gate_v2_repairs.py` | UNKNOWN — needs operator label; filename suggests: test mega gate v2 repairs |
| test_missing_evidence_to_work_orders.py | other | unknown | unknown | `tests/test_missing_evidence_to_work_orders.py` | UNKNOWN — needs operator label; filename suggests: test missing evidence to work orders |
| test_model_audit_absurd_adapter.py | other | unknown | unknown | `tests/test_model_audit_absurd_adapter.py` | UNKNOWN — needs operator label; filename suggests: test model audit absurd adapter |
| test_model_generation_event_bridge.py | other | unknown | unknown | `tests/test_model_generation_event_bridge.py` | UNKNOWN — needs operator label; filename suggests: test model generation event bridge |
| test_model_invocation_audit.py | other | unknown | unknown | `tests/test_model_invocation_audit.py` | UNKNOWN — needs operator label; filename suggests: test model invocation audit |
| test_model_invocation_trace.py | other | unknown | unknown | `tests/test_model_invocation_trace.py` | UNKNOWN — needs operator label; filename suggests: test model invocation trace |
| test_model_output_contract_audit.py | other | unknown | unknown | `tests/test_model_output_contract_audit.py` | UNKNOWN — needs operator label; filename suggests: test model output contract audit |
| test_model_runner_cli.py | other | unknown | unknown | `tests/test_model_runner_cli.py` | UNKNOWN — needs operator label; filename suggests: test model runner cli |
| test_model_turbine_ontology.py | other | unknown | unknown | `tests/test_model_turbine_ontology.py` | UNKNOWN — needs operator label; filename suggests: test model turbine ontology |
| test_mutation_safety_oracle.py | other | unknown | unknown | `tests/test_mutation_safety_oracle.py` | UNKNOWN — needs operator label; filename suggests: test mutation safety oracle |
| test_ncnn_edge_runtime_probe.py | other | unknown | unknown | `tests/test_ncnn_edge_runtime_probe.py` | UNKNOWN — needs operator label; filename suggests: test ncnn edge runtime probe |
| test_no_delete_guard.py | other | unknown | unknown | `tests/test_no_delete_guard.py` | UNKNOWN — needs operator label; filename suggests: test no delete guard |
| test_ocr_backlog_jobs.py | other | unknown | unknown | `tests/test_ocr_backlog_jobs.py` | UNKNOWN — needs operator label; filename suggests: test ocr backlog jobs |
| test_ontology_fidelity_extraction_outputs.py | other | unknown | unknown | `tests/test_ontology_fidelity_extraction_outputs.py` | Regression suite: extraction outputs fail on softened ontology and pass exact labels. |
| test_ontology_staging_schema_contract.py | other | unknown | unknown | `tests/test_ontology_staging_schema_contract.py` | UNKNOWN — needs operator label; filename suggests: test ontology staging schema contract |
| test_openai_codex_prompt_guide_ingest.py | other | unknown | unknown | `tests/test_openai_codex_prompt_guide_ingest.py` | UNKNOWN — needs operator label; filename suggests: test openai codex prompt guide ingest |
| test_operator_command_to_case.py | other | unknown | unknown | `tests/test_operator_command_to_case.py` | UNKNOWN — needs operator label; filename suggests: test operator command to case |
| test_operator_decisions.py | other | unknown | unknown | `tests/test_operator_decisions.py` | UNKNOWN — needs operator label; filename suggests: test operator decisions |
| test_operator_graph_eligibility_policy.py | other | unknown | unknown | `tests/test_operator_graph_eligibility_policy.py` | UNKNOWN — needs operator label; filename suggests: test operator graph eligibility policy |
| test_operator_ontology_fidelity.py | other | unknown | unknown | `tests/test_operator_ontology_fidelity.py` | Operator ontology fidelity fixture/extraction guard. This is a fixture_check when no extraction output is supplied. When an extraction output path is supplied with --extraction-output or OPERATOR_ONTOLOGY_EXTRACTION_OUTPUT, it also checks t |
| test_percyphon_kernel_bridge.py | other | unknown | unknown | `tests/test_percyphon_kernel_bridge.py` | UNKNOWN — needs operator label; filename suggests: test percyphon kernel bridge |
| test_pipeline_absurd_submission.py | other | unknown | unknown | `tests/test_pipeline_absurd_submission.py` | UNKNOWN — needs operator label; filename suggests: test pipeline absurd submission |
| test_pipeline_content_store_integration.py | other | unknown | unknown | `tests/test_pipeline_content_store_integration.py` | UNKNOWN — needs operator label; filename suggests: test pipeline content store integration |
| test_pipeline_contracts.py | other | unknown | unknown | `tests/test_pipeline_contracts.py` | UNKNOWN — needs operator label; filename suggests: test pipeline contracts |
| test_pipeline_resume.py | other | unknown | unknown | `tests/test_pipeline_resume.py` | UNKNOWN — needs operator label; filename suggests: test pipeline resume |
| test_pipeline_run_store.py | other | unknown | unknown | `tests/test_pipeline_run_store.py` | UNKNOWN — needs operator label; filename suggests: test pipeline run store |
| test_pipeline_worker_execution.py | other | unknown | unknown | `tests/test_pipeline_worker_execution.py` | UNKNOWN — needs operator label; filename suggests: test pipeline worker execution |
| test_post_gate_target_selector.py | other | unknown | unknown | `tests/test_post_gate_target_selector.py` | UNKNOWN — needs operator label; filename suggests: test post gate target selector |
| test_product_intake.py | other | unknown | unknown | `tests/test_product_intake.py` | UNKNOWN — needs operator label; filename suggests: test product intake |
| test_product_model_lane.py | other | unknown | unknown | `tests/test_product_model_lane.py` | UNKNOWN — needs operator label; filename suggests: test product model lane |
| test_product_parse_pipeline.py | other | unknown | unknown | `tests/test_product_parse_pipeline.py` | UNKNOWN — needs operator label; filename suggests: test product parse pipeline |
| test_product_timeline.py | other | unknown | unknown | `tests/test_product_timeline.py` | UNKNOWN — needs operator label; filename suggests: test product timeline |
| test_project2501_admin_prompt.py | other | unknown | unknown | `tests/test_project2501_admin_prompt.py` | UNKNOWN — needs operator label; filename suggests: test project2501 admin prompt |
| test_project2501_board_worker.py | other | unknown | unknown | `tests/test_project2501_board_worker.py` | UNKNOWN — needs operator label; filename suggests: test project2501 board worker |
| test_project2501_bytewax_board_stream.py | other | unknown | unknown | `tests/test_project2501_bytewax_board_stream.py` | UNKNOWN — needs operator label; filename suggests: test project2501 bytewax board stream |
| test_project2501_bytewax_board_stream_service.py | other | unknown | unknown | `tests/test_project2501_bytewax_board_stream_service.py` | UNKNOWN — needs operator label; filename suggests: test project2501 bytewax board stream service |
| test_project2501_core_contract.py | other | unknown | unknown | `tests/test_project2501_core_contract.py` | UNKNOWN — needs operator label; filename suggests: test project2501 core contract |
| test_project2501_prompt_policy.py | other | unknown | unknown | `tests/test_project2501_prompt_policy.py` | UNKNOWN — needs operator label; filename suggests: test project2501 prompt policy |
| test_project2501_script_audit_worker.py | other | unknown | unknown | `tests/test_project2501_script_audit_worker.py` | UNKNOWN — needs operator label; filename suggests: test project2501 script audit worker |
| test_project2501_watch_server.py | other | unknown | unknown | `tests/test_project2501_watch_server.py` | UNKNOWN — needs operator label; filename suggests: test project2501 watch server |
| test_project_brain_doc_authority.py | other | unknown | unknown | `tests/test_project_brain_doc_authority.py` | UNKNOWN — needs operator label; filename suggests: test project brain doc authority |
| test_proof_kernel.py | other | unknown | unknown | `tests/test_proof_kernel.py` | UNKNOWN — needs operator label; filename suggests: test proof kernel |
| test_quality_work_order_compiler.py | other | unknown | unknown | `tests/test_quality_work_order_compiler.py` | UNKNOWN — needs operator label; filename suggests: test quality work order compiler |
| test_quarantine_review_workflow.py | other | unknown | unknown | `tests/test_quarantine_review_workflow.py` | UNKNOWN — needs operator label; filename suggests: test quarantine review workflow |
| test_quarantine_streaming_guards.py | other | unknown | unknown | `tests/test_quarantine_streaming_guards.py` | UNKNOWN — needs operator label; filename suggests: test quarantine streaming guards |
| test_rete_bandit_gate_cli.py | other | unknown | unknown | `tests/test_rete_bandit_gate_cli.py` | UNKNOWN — needs operator label; filename suggests: test rete bandit gate cli |
| test_rfc_claim_discipline.py | other | unknown | unknown | `tests/test_rfc_claim_discipline.py` | UNKNOWN — needs operator label; filename suggests: test rfc claim discipline |
| test_rfc_program_check.py | other | unknown | unknown | `tests/test_rfc_program_check.py` | UNKNOWN — needs operator label; filename suggests: test rfc program check |
| test_river_bytewax_root_scripts.py | other | unknown | unknown | `tests/test_river_bytewax_root_scripts.py` | UNKNOWN — needs operator label; filename suggests: test river bytewax root scripts |
| test_ruthless_remediation_contracts.py | other | unknown | unknown | `tests/test_ruthless_remediation_contracts.py` | UNKNOWN — needs operator label; filename suggests: test ruthless remediation contracts |
| test_same_lineage_validator.py | other | unknown | unknown | `tests/test_same_lineage_validator.py` | UNKNOWN — needs operator label; filename suggests: test same lineage validator |
| test_script_bucket_manifest.py | other | unknown | unknown | `tests/test_script_bucket_manifest.py` | UNKNOWN — needs operator label; filename suggests: test script bucket manifest |
| test_script_survival_audit.py | other | unknown | unknown | `tests/test_script_survival_audit.py` | UNKNOWN — needs operator label; filename suggests: test script survival audit |
| test_signal_aggregator_script.py | other | unknown | unknown | `tests/test_signal_aggregator_script.py` | UNKNOWN — needs operator label; filename suggests: test signal aggregator script |
| test_slop_audit_law.py | other | unknown | unknown | `tests/test_slop_audit_law.py` | UNKNOWN — needs operator label; filename suggests: test slop audit law |
| test_source_bundle.py | other | unknown | unknown | `tests/test_source_bundle.py` | UNKNOWN — needs operator label; filename suggests: test source bundle |
| test_source_linked_claims.py | other | unknown | unknown | `tests/test_source_linked_claims.py` | UNKNOWN — needs operator label; filename suggests: test source linked claims |
| test_source_quote_extractor.py | other | unknown | unknown | `tests/test_source_quote_extractor.py` | UNKNOWN — needs operator label; filename suggests: test source quote extractor |
| test_spine_authority_checker.py | other | unknown | unknown | `tests/test_spine_authority_checker.py` | UNKNOWN — needs operator label; filename suggests: test spine authority checker |
| test_status_ledger_evidence_gate.py | other | unknown | unknown | `tests/test_status_ledger_evidence_gate.py` | UNKNOWN — needs operator label; filename suggests: test status ledger evidence gate |
| test_strict_model_stack_admission.py | other | unknown | unknown | `tests/test_strict_model_stack_admission.py` | UNKNOWN — needs operator label; filename suggests: test strict model stack admission |
| test_subsystem_quality_audit.py | other | unknown | unknown | `tests/test_subsystem_quality_audit.py` | UNKNOWN — needs operator label; filename suggests: test subsystem quality audit |
| test_surface_demem_boundaries.py | other | unknown | unknown | `tests/test_surface_demem_boundaries.py` | Surface instruction acceptance tests against DeMem boundaries. |
| test_surface_instruction_compiler.py | other | unknown | unknown | `tests/test_surface_instruction_compiler.py` | UNKNOWN — needs operator label; filename suggests: test surface instruction compiler |
| test_swarm_usage_ledger.py | other | unknown | unknown | `tests/test_swarm_usage_ledger.py` | UNKNOWN — needs operator label; filename suggests: test swarm usage ledger |
| test_system_graph_safety_audit.py | other | unknown | unknown | `tests/test_system_graph_safety_audit.py` | UNKNOWN — needs operator label; filename suggests: test system graph safety audit |
| test_template_contract.py | other | unknown | unknown | `tests/test_template_contract.py` | UNKNOWN — needs operator label; filename suggests: test template contract |
| test_villager_status.py | other | unknown | unknown | `tests/test_villager_status.py` | UNKNOWN — needs operator label; filename suggests: test villager status |
| test_work_order_importer_cli.py | other | unknown | unknown | `tests/test_work_order_importer_cli.py` | UNKNOWN — needs operator label; filename suggests: test work order importer cli |
| test_working_reality_law.py | other | unknown | unknown | `tests/test_working_reality_law.py` | UNKNOWN — needs operator label; filename suggests: test working reality law |
| test_workshare_allocator.py | other | unknown | unknown | `tests/test_workshare_allocator.py` | UNKNOWN — needs operator label; filename suggests: test workshare allocator |

## NEEDS OPERATOR LABEL

| name | category | path |
|---|---|---|
| ALGOS | ALGOS | `ALGOS` |
| scripts | SCRIPTS | `01_REPOS/doggystyle/scripts` |
| benchmark_mark2.py | SCRIPTS | `01_REPOS/doggystyle/scripts/benchmark_mark2.py` |
| diogenes_grpc_smoke.py | SCRIPTS | `01_REPOS/doggystyle/scripts/diogenes_grpc_smoke.py` |
| scripts | SCRIPTS | `01_REPOS/llama.cpp/scripts` |
| compare-llama-bench.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/compare-llama-bench.py` |
| compare-logprobs.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/compare-logprobs.py` |
| gen-unicode-data.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/gen-unicode-data.py` |
| gcn-cdna-vgpr-check.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/hip/gcn-cdna-vgpr-check.py` |
| jinja-tester.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/jinja/jinja-tester.py` |
| requirements.txt | SCRIPTS | `01_REPOS/llama.cpp/scripts/jinja/requirements.txt` |
| server-bench.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/server-bench.py` |
| server-test-model.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/server-test-model.py` |
| ggml-hexagon-profile.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py` |
| requirements.txt | SCRIPTS | `01_REPOS/llama.cpp/scripts/snapdragon/qdc/requirements.txt` |
| sync_vendor.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/sync_vendor.py` |
| verify-checksum-models.py | SCRIPTS | `01_REPOS/llama.cpp/scripts/verify-checksum-models.py` |
| absurd_job.schema.json | SCRIPTS | `06_SCHEMA/absurd_job.schema.json` |
| bytewax_signal_packet.schema.json | SCRIPTS | `06_SCHEMA/bytewax_signal_packet.schema.json` |
| chrono_claim.schema.json | SCRIPTS | `06_SCHEMA/chrono_claim.schema.json` |
| contradiction_record.schema.json | SCRIPTS | `06_SCHEMA/contradiction_record.schema.json` |
| conversation_command.schema.json | SCRIPTS | `06_SCHEMA/conversation_command.schema.json` |
| dev_order_matrix_policy.v1.json | SCRIPTS | `06_SCHEMA/dev_order_matrix_policy.v1.json` |
| document_parse_result.schema.json | SCRIPTS | `06_SCHEMA/document_parse_result.schema.json` |
| entity_candidate_registry.schema.json | SCRIPTS | `06_SCHEMA/entity_candidate_registry.schema.json` |
| event_candidate_registry.schema.json | SCRIPTS | `06_SCHEMA/event_candidate_registry.schema.json` |
| graph_promotion_packet.schema.json | SCRIPTS | `06_SCHEMA/graph_promotion_packet.schema.json` |
| import_policy.schema.json | SCRIPTS | `06_SCHEMA/import_policy.schema.json` |
| krampus_custody.schema.json | SCRIPTS | `06_SCHEMA/krampus_custody.schema.json` |
| memory_candidate.schema.json | SCRIPTS | `06_SCHEMA/memory_candidate.schema.json` |
| model_invocation_receipt.schema.json | SCRIPTS | `06_SCHEMA/model_invocation_receipt.schema.json` |
| ontology_staging.schema.json | SCRIPTS | `06_SCHEMA/ontology_staging.schema.json` |
| operator_decision_receipt.schema.json | SCRIPTS | `06_SCHEMA/operator_decision_receipt.schema.json` |
| pipeline_stage.schema.json | SCRIPTS | `06_SCHEMA/pipeline_stage.schema.json` |
| proof_object.schema.json | SCRIPTS | `06_SCHEMA/proof_object.schema.json` |
| source_bundle.schema.json | SCRIPTS | `06_SCHEMA/source_bundle.schema.json` |
| status_ledger_entry.schema.json | SCRIPTS | `06_SCHEMA/status_ledger_entry.schema.json` |
| work_order.schema.json | SCRIPTS | `06_SCHEMA/work_order.schema.json` |
| scripts | SCRIPTS | `scripts` |
| dbos_fault_injector_20260527T034022927312Z.json | SCRIPTS | `scripts/05_OUTPUTS/chaos/dbos_fault_injector_20260527T034022927312Z.json` |
| dbos_river_wrapper_worker-once_20260526T205102405690Z.json | SCRIPTS | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205102405690Z.json` |
| dbos_river_wrapper_worker-once_20260526T205321153628Z.json | SCRIPTS | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205321153628Z.json` |
| dbos_river_wrapper_worker-once_20260526T205401745296Z.json | SCRIPTS | `scripts/05_OUTPUTS/dbos/dbos_river_wrapper_worker-once_20260526T205401745296Z.json` |
| CORPSE_MANIFEST.jsonl | SCRIPTS | `scripts/CORPSE_MANIFEST.jsonl` |
| KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl | SCRIPTS | `scripts/KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl` |
| SCRIPT_AUDIT_MANIFEST.jsonl | SCRIPTS | `scripts/SCRIPT_AUDIT_MANIFEST.jsonl` |
| absurd_health_check.py | SCRIPTS | `scripts/absurd_health_check.py` |
| bytewax_absurd_stream_audit.py | SCRIPTS | `scripts/bytewax_absurd_stream_audit.py` |
| case_dashboard_data.py | SCRIPTS | `scripts/case_dashboard_data.py` |
| case_packet_compiler.py | SCRIPTS | `scripts/case_packet_compiler.py` |
| case_packet_renderer.py | SCRIPTS | `scripts/case_packet_renderer.py` |
| case_pipeline_dispatch.py | SCRIPTS | `scripts/case_pipeline_dispatch.py` |
| case_workspace.py | SCRIPTS | `scripts/case_workspace.py` |
| cep_builder.py | SCRIPTS | `scripts/cep_builder.py` |
| cep_to_kernel_route.py | SCRIPTS | `scripts/cep_to_kernel_route.py` |
| check_all_lucidota_services.py | SCRIPTS | `scripts/check_all_lucidota_services.py` |
| chrono_from_inventory.py | SCRIPTS | `scripts/chrono_from_inventory.py` |
| chrono_projection_claim_verifier.py | SCRIPTS | `scripts/chrono_projection_claim_verifier.py` |
| chunk_to_staging.py | SCRIPTS | `scripts/chunk_to_staging.py` |
| ckdog_kernel_events.py | SCRIPTS | `scripts/ckdog_kernel_events.py` |
| ckdog_kernel_route_plan.py | SCRIPTS | `scripts/ckdog_kernel_route_plan.py` |
| claim_clusterer.py | SCRIPTS | `scripts/claim_clusterer.py` |
| claim_support_score.py | SCRIPTS | `scripts/claim_support_score.py` |
| claim_table_compiler.py | SCRIPTS | `scripts/claim_table_compiler.py` |
| content_store.py | SCRIPTS | `scripts/content_store.py` |
| contradiction_report.py | SCRIPTS | `scripts/contradiction_report.py` |
| control_packet_ledger.py | SCRIPTS | `scripts/control_packet_ledger.py` |
| dev_order_gate.py | SCRIPTS | `scripts/dev_order_gate.py` |
| dev_order_matrix_wrapper.py | SCRIPTS | `scripts/dev_order_matrix_wrapper.py` |
| document_parse_router.py | SCRIPTS | `scripts/document_parse_router.py` |
| entity_candidate_registry.py | SCRIPTS | `scripts/entity_candidate_registry.py` |
| event_candidate_registry.py | SCRIPTS | `scripts/event_candidate_registry.py` |
| export_bundle.py | SCRIPTS | `scripts/export_bundle.py` |
| generate_lucidota_architecture_manifest.py | SCRIPTS | `scripts/generate_lucidota_architecture_manifest.py` |
| golden_path_regression_gate.py | SCRIPTS | `scripts/golden_path_regression_gate.py` |
| gpu_runtime_budget.py | SCRIPTS | `scripts/gpu_runtime_budget.py` |
| graph_candidate.py | SCRIPTS | `scripts/graph_candidate.py` |
| graph_promotion.py | SCRIPTS | `scripts/graph_promotion.py` |
| graph_promotion_policy_check.py | SCRIPTS | `scripts/graph_promotion_policy_check.py` |
| graph_store_adapter.py | SCRIPTS | `scripts/graph_store_adapter.py` |
| import_export_bundle.py | SCRIPTS | `scripts/import_export_bundle.py` |
| import_policy.py | SCRIPTS | `scripts/import_policy.py` |
| indy_reads_absurd_brief.py | SCRIPTS | `scripts/indy_reads_absurd_brief.py` |
| instruction_conflict_scanner.py | SCRIPTS | `scripts/instruction_conflict_scanner.py` |
| kernel_packet_cli.py | SCRIPTS | `scripts/kernel_packet_cli.py` |
| knowledge_library_check.py | SCRIPTS | `scripts/knowledge_library_check.py` |
| korpus_krampii.py | SCRIPTS | `scripts/korpus_krampii.py` |
| krampus_extension_policy.py | SCRIPTS | `scripts/krampus_extension_policy.py` |
| krampus_inventory.py | SCRIPTS | `scripts/krampus_inventory.py` |
| krampuschewing_absurd_adapter.py | SCRIPTS | `scripts/krampuschewing_absurd_adapter.py` |
| DBOS_LEGACY_ARCHIVE_MANIFEST.json | SCRIPTS | `scripts/legacy/DBOS_LEGACY_ARCHIVE_MANIFEST.json` |
| boring_beast_event_chain_verify.py | SCRIPTS | `scripts/legacy/boring_beast_event_chain_verify.py` |
| catchme_sensitivity_map.py | SCRIPTS | `scripts/legacy/catchme_sensitivity_map.py` |
| dbos_fault_injector.py | SCRIPTS | `scripts/legacy/dbos_fault_injector.py` |
| korpus_krampii.py | SCRIPTS | `scripts/legacy/korpus_krampii.py` |
| lucidota_dbos_smoke.py | SCRIPTS | `scripts/legacy/lucidota_dbos_smoke.py` |
| lucidota_workflow_registry.py | SCRIPTS | `scripts/legacy/lucidota_workflow_registry.py` |
| production_readiness_eval.py | SCRIPTS | `scripts/legacy/production_readiness_eval.py` |
| simplemem_promotion_bridge.py | SCRIPTS | `scripts/legacy/simplemem_promotion_bridge.py` |
| system_state_desync_detector.py | SCRIPTS | `scripts/legacy/system_state_desync_detector.py` |
| system_telemetry_exporter.py | SCRIPTS | `scripts/legacy/system_telemetry_exporter.py` |
| unified_absurd_ingest_worker.py | SCRIPTS | `scripts/legacy/unified_absurd_ingest_worker.py` |
| worker_command_registry.py | SCRIPTS | `scripts/legacy/worker_command_registry.py` |
| lucidota_acceptance.py | SCRIPTS | `scripts/lucidota_acceptance.py` |
| lucidota_chaos_suite.py | SCRIPTS | `scripts/lucidota_chaos_suite.py` |
| lucidota_ci_gate.py | SCRIPTS | `scripts/lucidota_ci_gate.py` |
| lucidota_cli.py | SCRIPTS | `scripts/lucidota_cli.py` |
| lucidota_deploy_dry_run.py | SCRIPTS | `scripts/lucidota_deploy_dry_run.py` |
| lucidota_pipeline.py | SCRIPTS | `scripts/lucidota_pipeline.py` |
| lucidota_production_signoff.py | SCRIPTS | `scripts/lucidota_production_signoff.py` |
| lucidota_release_gate.py | SCRIPTS | `scripts/lucidota_release_gate.py` |
| lucidota_release_manifest.py | SCRIPTS | `scripts/lucidota_release_manifest.py` |
| lucidota_surface_emit_command.py | SCRIPTS | `scripts/lucidota_surface_emit_command.py` |
| matrix_trace_checker.py | SCRIPTS | `scripts/matrix_trace_checker.py` |
| media_metadata.py | SCRIPTS | `scripts/media_metadata.py` |
| mega_gate_fault_injector.py | SCRIPTS | `scripts/mega_gate_fault_injector.py` |
| memory_candidates.py | SCRIPTS | `scripts/memory_candidates.py` |
| missing_evidence_checklist.py | SCRIPTS | `scripts/missing_evidence_checklist.py` |
| model_inventory.py | SCRIPTS | `scripts/model_inventory.py` |
| model_runner_config.py | SCRIPTS | `scripts/model_runner_config.py` |
| model_runner_stub.py | SCRIPTS | `scripts/model_runner_stub.py` |
| mutation_safety_oracle.py | SCRIPTS | `scripts/mutation_safety_oracle.py` |
| next_action_compiler.py | SCRIPTS | `scripts/next_action_compiler.py` |
| ocr_backlog.py | SCRIPTS | `scripts/ocr_backlog.py` |
| ocr_routing.py | SCRIPTS | `scripts/ocr_routing.py` |
| ontology_staging.py | SCRIPTS | `scripts/ontology_staging.py` |
| operator_command_router.py | SCRIPTS | `scripts/operator_command_router.py` |
| operator_decisions.py | SCRIPTS | `scripts/operator_decisions.py` |
| patch_runner.py | SCRIPTS | `scripts/patch_runner.py` |
| path_redaction.py | SCRIPTS | `scripts/path_redaction.py` |
| phase05_allowlisted_ingest.py | SCRIPTS | `scripts/phase05_allowlisted_ingest.py` |
| pipeline_contracts.py | SCRIPTS | `scripts/pipeline_contracts.py` |
| pipeline_run_store.py | SCRIPTS | `scripts/pipeline_run_store.py` |
| product_intake.py | SCRIPTS | `scripts/product_intake.py` |
| product_parse_pipeline.py | SCRIPTS | `scripts/product_parse_pipeline.py` |
| product_proof_harness.py | SCRIPTS | `scripts/product_proof_harness.py` |
| quarantine_review.py | SCRIPTS | `scripts/quarantine_review.py` |
| receipt_exporter.py | SCRIPTS | `scripts/receipt_exporter.py` |
| receipt_writer.py | SCRIPTS | `scripts/receipt_writer.py` |
| recovery_matrix.py | SCRIPTS | `scripts/recovery_matrix.py` |
| report_retention_index.py | SCRIPTS | `scripts/report_retention_index.py` |
| run_dev_order_methodology_checks.py | SCRIPTS | `scripts/run_dev_order_methodology_checks.py` |
| run_golden_path_hardening_checks.py | SCRIPTS | `scripts/run_golden_path_hardening_checks.py` |
| run_instruction_hygiene.py | SCRIPTS | `scripts/run_instruction_hygiene.py` |
| source_bundle.py | SCRIPTS | `scripts/source_bundle.py` |
| source_quote_extractor.py | SCRIPTS | `scripts/source_quote_extractor.py` |
| spine_authority_checker.py | SCRIPTS | `scripts/spine_authority_checker.py` |
| spine_common.py | SCRIPTS | `scripts/spine_common.py` |
| spine_job_adapter.py | SCRIPTS | `scripts/spine_job_adapter.py` |
| status_ledger_evidence_gate.py | SCRIPTS | `scripts/status_ledger_evidence_gate.py` |
| status_ledger_fault_injector.py | SCRIPTS | `scripts/status_ledger_fault_injector.py` |
| text_chunker.py | SCRIPTS | `scripts/text_chunker.py` |
| tickletrunk_fault_injector.py | SCRIPTS | `scripts/tickletrunk_fault_injector.py` |
| timeline_compiler.py | SCRIPTS | `scripts/timeline_compiler.py` |
| work_order_importer.py | SCRIPTS | `scripts/work_order_importer.py` |
| models | MODELS | `01_REPOS/llama.cpp/models` |
| Apriel-1.6-15b-Thinker-fixed.jinja | MODELS | `01_REPOS/llama.cpp/models/templates/Apriel-1.6-15b-Thinker-fixed.jinja` |
| models | MODELS | `03_VAULT/models` |
| model_source.json | MODELS | `03_VAULT/models/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/model_source.json` |
| gliner_config.json | MODELS | `03_VAULT/models/gliner/urchade_gliner_small-v2.1/gliner_config.json` |
| model_source.json | MODELS | `03_VAULT/models/prism-ml/Ternary-Bonsai-4B-gguf/model_source.json` |
| model_source.json | MODELS | `03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/model_source.json` |
| CMakeLists.txt | LORAS | `01_REPOS/llama.cpp/tools/export-lora/CMakeLists.txt` |
| export-lora.cpp | LORAS | `01_REPOS/llama.cpp/tools/export-lora/export-lora.cpp` |
| lora_cartridges | LORAS | `04_RUNTIME/lora_cartridges` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-big-boy-did-it-and-ran-away__75b098800d33/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__a-death-in-malta---an-assassination-and-a-family-s-quest-for__7c801e56c9e5/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__blood-in-the-machine_-the-origins-of-the-rebellion-against__be935ea0def1/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__one-day-everyone-will-have-always-been-against-this__f3b9eb68ba41/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__out-of-darkness-_-essays-on-corporate-power-and-civic__54a366e87cc1/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads__the-small-and-the-mighty_-twelve-unsung-americans-who__de970a47136f/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_0ee1dd6a9303e1c9/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_14247afedb37c8f3/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_1785bdd20d80af31/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_2338bd2a53d46e8d/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_329caffc4b9663a9/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_36f35faf6c761cdc/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_38a81aea7ee464d7/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4774a702320f3978/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_4eb25a2b79fc10d3/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_5b224636ef576a85/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_602f7eb947ebb36e/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6080da5006afc9b4/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_689bb86fcbd0a797/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_6e01552d7f12f520/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_791c3b11585e512c/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_88912819f143b52f/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_b3b4a111b82f97c1/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_c0c93800dddcb166/validation.jsonl` |
| manifest.json | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/manifest.json` |
| train.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/train.jsonl` |
| validation.jsonl | LORAS | `04_RUNTIME/lora_cartridges/indy_reads_book_e1ecc91ee15ff224/validation.jsonl` |
| 06_SCHEMA | SCHEMAS | `06_SCHEMA` |
| abductive_db_os_objects.v1.json | SCHEMAS | `06_SCHEMA/abductive_db_os/abductive_db_os_objects.v1.json` |
| absurd_abductive_objects.v1.json | SCHEMAS | `06_SCHEMA/absurd_abductive/absurd_abductive_objects.v1.json` |
| ontology_packs | SCHEMAS | `BOOKS/ontology_packs` |
| registry.json | SCHEMAS | `BOOKS/ontology_packs/sio8/registry.json` |
| skills | SKILLS | `/home/mfspx/.codex/skills` |
| LICENSE.txt | SKILLS | `/home/mfspx/.codex/skills/.system/imagegen/LICENSE.txt` |
| openai.yaml | SKILLS | `/home/mfspx/.codex/skills/.system/imagegen/agents/openai.yaml` |
| LICENSE.txt | SKILLS | `/home/mfspx/.codex/skills/.system/openai-docs/LICENSE.txt` |
| openai.yaml | SKILLS | `/home/mfspx/.codex/skills/.system/openai-docs/agents/openai.yaml` |
| openai.yaml | SKILLS | `/home/mfspx/.codex/skills/.system/plugin-creator/agents/openai.yaml` |
| openai.yaml | SKILLS | `/home/mfspx/.codex/skills/.system/skill-creator/agents/openai.yaml` |
| license.txt | SKILLS | `/home/mfspx/.codex/skills/.system/skill-creator/license.txt` |
| LICENSE.txt | SKILLS | `/home/mfspx/.codex/skills/.system/skill-installer/LICENSE.txt` |
| openai.yaml | SKILLS | `/home/mfspx/.codex/skills/.system/skill-installer/agents/openai.yaml` |
| plugins | PLUGINS | `/home/mfspx/.claw/plugins` |
| installed.json | PLUGINS | `/home/mfspx/.claw/plugins/installed.json` |
| plugin.json | PLUGINS | `/home/mfspx/.claw/plugins/installed/example-bundled-bundled/.claw-plugin/plugin.json` |
| plugin.json | PLUGINS | `/home/mfspx/.claw/plugins/installed/sample-hooks-bundled/.claw-plugin/plugin.json` |
| plugins | PLUGINS | `/home/mfspx/.codex/plugins` |
| services | SERVICES | `services` |
| __init__.py | SERVICES | `services/__init__.py` |
| __init__.py | SERVICES | `services/fairyfuse/__init__.py` |
| vendor_manifest.json | SERVICES | `services/ternary_lab/vendor_manifest.json` |
| BOOKS | BOOKS | `BOOKS` |
| watch_state.json | BOOKS | `BOOKS/.indy_reads/watch_state.json` |
| 07_SURFACES | SURFACES | `07_SURFACES` |
| conversation_instruction_compiler_sample.html | SURFACES | `07_SURFACES/generated/conversation_instruction_compiler_sample.html` |
| marrow_loop_status.html | SURFACES | `07_SURFACES/generated/marrow_loop_status.html` |
| status_ledger.html | SURFACES | `07_SURFACES/generated/status_ledger.html` |
| conversation_instruction_compiler_sample.json | SURFACES | `07_SURFACES/sidecars/conversation_instruction_compiler_sample.json` |
| marrow_loop_status.json | SURFACES | `07_SURFACES/sidecars/marrow_loop_status.json` |
| JOB_BOOTHS.json | WORKFLOWS | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/JOB_BOOTHS.json` |
| PERSONA_SEED_SCHEMA.json | WORKFLOWS | `00_PROJECT_BRAIN/AGENTSI_SELF_SOVEREIGN_JOB_FAIR/PERSONA_SEED_SCHEMA.json` |
| CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json | WORKFLOWS | `00_PROJECT_BRAIN/CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json` |
| DURABLE_WORKFLOW_DECISION.json | WORKFLOWS | `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json` |
| ROLE_MODES.json | WORKFLOWS | `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ROLE_MODES.json` |
| UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents.html | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents.html` |
| 275.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/275.js` |
| deferred.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.js` |
| deferred.odsp-common.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.odsp-common.js` |
| deferred.odsp-datasources.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/deferred.odsp-datasources.js` |
| fui.core-2c6f08cc.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/fui.core-2c6f08cc.js` |
| modernFrame.html | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame.html` |
| a.html | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/a.html` |
| a_002.html | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/a_002.html` |
| modernframe.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/modernframe.js` |
| plt.listviewdataprefetch.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/modernFrame_data/plt.listviewdataprefetch.js` |
| odsp.react.lib-760716ce.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/odsp.react.lib-760716ce.js` |
| ondemand.resx.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/ondemand.resx.js` |
| plt.items-view.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.items-view.js` |
| plt.odsp-common.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.odsp-common.js` |
| plt.office-ui-fabric-react.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/plt.office-ui-fabric-react.js` |
| splistwebpack.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/splistwebpack.js` |
| suiteux.shell.core.a34fe4706885bcb77f89.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.core.a34fe4706885bcb77f89.js` |
| suiteux.shell.mast.d883ad4900e81860f22a.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.mast.d883ad4900e81860f22a.js` |
| suiteux.shell.otellogging.d1a49d28c74cd9d315b0.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.otellogging.d1a49d28c74cd9d315b0.js` |
| suiteux.shell.plus.4a6e59c2f4ac09534ca6.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.plus.4a6e59c2f4ac09534ca6.js` |
| suiteux.shell.searchbox.db39686c69105329489b.js | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/UNIT - Privacy and Records Management - 2026-31-RE - Response Records.pdf - All Documents_files/suiteux.shell.searchbox.db39686c69105329489b.js` |
| index.json | WORKFLOWS | `00_PROJECT_BRAIN/KNOWLEDGE_LIBRARY/index.json` |
| GOAL_REQUIREMENT_MATRIX.json | WORKFLOWS | `00_PROJECT_BRAIN/RFCS/GOAL_REQUIREMENT_MATRIX.json` |
| RFC_SUBJECT_REGISTRY.json | WORKFLOWS | `00_PROJECT_BRAIN/RFCS/RFC_SUBJECT_REGISTRY.json` |
| TICKLETRUNK.json | WORKFLOWS | `00_PROJECT_BRAIN/TICKLETRUNK.json` |
| TICKLETRUNK.md | WORKFLOWS | `00_PROJECT_BRAIN/TICKLETRUNK.md` |
| canonical_graph_write_allowlist.json | WORKFLOWS | `00_PROJECT_BRAIN/canonical_graph_write_allowlist.json` |
| claude_code_claw_runtime_registry.json | WORKFLOWS | `00_PROJECT_BRAIN/claude_code_claw_runtime_registry.json` |
| gpu_model_runtime_registry.json | WORKFLOWS | `00_PROJECT_BRAIN/gpu_model_runtime_registry.json` |
| instruction_authority_registry.json | WORKFLOWS | `00_PROJECT_BRAIN/instruction_authority_registry.json` |
| operator_graph_eligibility_policy.json | WORKFLOWS | `00_PROJECT_BRAIN/operator_graph_eligibility_policy.json` |
| rust_port_candidacy_registry.json | WORKFLOWS | `00_PROJECT_BRAIN/rust_port_candidacy_registry.json` |
| spine_authority_registry.json | WORKFLOWS | `00_PROJECT_BRAIN/spine_authority_registry.json` |
| setup.py | REPOS | `01_REPOS/PocketFlow/setup.py` |
| Cargo.toml | REPOS | `01_REPOS/claudecode/rust/Cargo.toml` |
| drf.py | REPOS | `01_REPOS/cybercrafter_drf/drf.py` |
| requirements.txt | REPOS | `01_REPOS/cybercrafter_drf/requirements.txt` |
| pyproject.toml | REPOS | `01_REPOS/doggystyle/pyproject.toml` |
| CMakeLists.txt | REPOS | `01_REPOS/flywheel1412_ncnn/CMakeLists.txt` |
| LICENSE.txt | REPOS | `01_REPOS/flywheel1412_ncnn/LICENSE.txt` |
| README.md | REPOS | `01_REPOS/flywheel1412_ncnn/benchmark/README.md` |
| pyproject.toml | REPOS | `01_REPOS/flywheel1412_ncnn/pyproject.toml` |
| setup.py | REPOS | `01_REPOS/flywheel1412_ncnn/setup.py` |
| CMakeLists.txt | REPOS | `01_REPOS/llama.cpp/CMakeLists.txt` |
| CMakePresets.json | REPOS | `01_REPOS/llama.cpp/CMakePresets.json` |
| convert_hf_to_gguf.py | REPOS | `01_REPOS/llama.cpp/convert_hf_to_gguf.py` |
| convert_hf_to_gguf_update.py | REPOS | `01_REPOS/llama.cpp/convert_hf_to_gguf_update.py` |
| convert_llama_ggml_to_gguf.py | REPOS | `01_REPOS/llama.cpp/convert_llama_ggml_to_gguf.py` |
| convert_lora_to_gguf.py | REPOS | `01_REPOS/llama.cpp/convert_lora_to_gguf.py` |
| pyproject.toml | REPOS | `01_REPOS/llama.cpp/gguf-py/pyproject.toml` |
| pyproject.toml | REPOS | `01_REPOS/llama.cpp/pyproject.toml` |
| pyrightconfig.json | REPOS | `01_REPOS/llama.cpp/pyrightconfig.json` |
| requirements.txt | REPOS | `01_REPOS/llama.cpp/requirements.txt` |
| ty.toml | REPOS | `01_REPOS/llama.cpp/ty.toml` |
| LICENSE-COMMERCIAL.txt | REPOS | `01_REPOS/llm-router/LICENSE-COMMERCIAL.txt` |
| pyproject.toml | REPOS | `01_REPOS/llm-router/pyproject.toml` |
| package-lock.json | REPOS | `01_REPOS/llxprt-code/package-lock.json` |
| package.json | REPOS | `01_REPOS/llxprt-code/package.json` |
| tsconfig.json | REPOS | `01_REPOS/llxprt-code/tsconfig.json` |
| Cargo.toml | REPOS | `01_REPOS/lucidota_etl/Cargo.toml` |
| pyproject.toml | REPOS | `01_REPOS/needle/pyproject.toml` |
| requirements.txt | REPOS | `01_REPOS/needle/requirements.txt` |
| setup | REPOS | `01_REPOS/needle/setup` |
| CMakeLists.txt | REPOS | `01_REPOS/prismml_llama.cpp/CMakeLists.txt` |
| CMakePresets.json | REPOS | `01_REPOS/prismml_llama.cpp/CMakePresets.json` |
| convert_hf_to_gguf.py | REPOS | `01_REPOS/prismml_llama.cpp/convert_hf_to_gguf.py` |
| convert_hf_to_gguf_update.py | REPOS | `01_REPOS/prismml_llama.cpp/convert_hf_to_gguf_update.py` |
| convert_llama_ggml_to_gguf.py | REPOS | `01_REPOS/prismml_llama.cpp/convert_llama_ggml_to_gguf.py` |
| convert_lora_to_gguf.py | REPOS | `01_REPOS/prismml_llama.cpp/convert_lora_to_gguf.py` |
| pyproject.toml | REPOS | `01_REPOS/prismml_llama.cpp/gguf-py/pyproject.toml` |
| pyproject.toml | REPOS | `01_REPOS/prismml_llama.cpp/pyproject.toml` |
| pyrightconfig.json | REPOS | `01_REPOS/prismml_llama.cpp/pyrightconfig.json` |
| requirements.txt | REPOS | `01_REPOS/prismml_llama.cpp/requirements.txt` |
| ty.toml | REPOS | `01_REPOS/prismml_llama.cpp/ty.toml` |
| gemini-code-1779083240490.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779083240490.py` |
| gemini-code-1779083502852.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779083502852.py` |
| gemini-code-1779083616000.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779083616000.py` |
| gemini-code-1779083633812.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779083633812.py` |
| gemini-code-1779084568395.json | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779084568395.json` |
| gemini-code-1779084572530.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779084572530.py` |
| gemini-code-1779084576822.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779084576822.py` |
| gemini-code-1779084582363.py | REPOS | `01_REPOS/sharksnailmangame/gemini-code-1779084582363.py` |
| PUSH_TO_GITHUB.txt | REPOS | `01_REPOS/sydsec_syd/PUSH_TO_GITHUB.txt` |
| requirements.txt | REPOS | `01_REPOS/sydsec_syd/requirements.txt` |
| syd.py | REPOS | `01_REPOS/sydsec_syd/syd.py` |
| local_audit_block_0001_prompt.txt | RUNTIME | `04_RUNTIME/absurd_abductive/local_audit_block_0001_prompt.txt` |
| absurd_probe_2026-05-19.txt | RUNTIME | `04_RUNTIME/absurd_ingest_probe/absurd_probe_2026-05-19.txt` |
| absurd_prompt_vram_probe_2026-05-19.txt | RUNTIME | `04_RUNTIME/absurd_ingest_probe/absurd_prompt_vram_probe_2026-05-19.txt` |
| absurd_prompt_vram_probe_new_worker_2026-05-19.txt | RUNTIME | `04_RUNTIME/absurd_ingest_probe/absurd_prompt_vram_probe_new_worker_2026-05-19.txt` |
| indy-reads.json | RUNTIME | `04_RUNTIME/agentsi/agents/indy-reads.json` |
| async_duckdb_footprint.jsonl | RUNTIME | `04_RUNTIME/bytewax_abductive_blender/async_duckdb_footprint.jsonl` |
| simple_dev_order_4ce1a0e20f9e7a8167cd4c51e4319731694e1a1ff496367a0413d9f8ef40cde5.txt | RUNTIME | `04_RUNTIME/dev_orders/original/simple_dev_order_4ce1a0e20f9e7a8167cd4c51e4319731694e1a1ff496367a0413d9f8ef40cde5.txt` |
| stdin_fixture_816217781dad823cf50e7d57f7bf9a326753bfeb891735a4a1020758f43bb2e5.txt | RUNTIME | `04_RUNTIME/dev_orders/original/stdin_fixture_816217781dad823cf50e7d57f7bf9a326753bfeb891735a4a1020758f43bb2e5.txt` |
| simple_dev_order.wrapped.txt | RUNTIME | `04_RUNTIME/dev_orders/wrapped/simple_dev_order.wrapped.txt` |
| ternary_router_heartbeat.jsonl | RUNTIME | `04_RUNTIME/fairyfuse/ternary_router_heartbeat.jsonl` |
| groq_worker_slice_audit.txt | RUNTIME | `04_RUNTIME/goals/groq_worker_slice_audit.txt` |
| groq_worker_slice_plan.txt | RUNTIME | `04_RUNTIME/goals/groq_worker_slice_plan.txt` |
| groq_worker_slice_redteam.txt | RUNTIME | `04_RUNTIME/goals/groq_worker_slice_redteam.txt` |
| model_fabric_groq_20260526T063223564540Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063223564540Z.txt` |
| model_fabric_groq_20260526T063238218105Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063238218105Z.txt` |
| model_fabric_groq_20260526T063258826415Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063258826415Z.txt` |
| model_fabric_groq_20260526T063620627262Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063620627262Z.txt` |
| model_fabric_groq_20260526T063632447832Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063632447832Z.txt` |
| model_fabric_groq_20260526T063649503788Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T063649503788Z.txt` |
| model_fabric_groq_20260526T064714019472Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T064714019472Z.txt` |
| model_fabric_groq_20260526T064947858523Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T064947858523Z.txt` |
| model_fabric_groq_20260526T065117542897Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T065117542897Z.txt` |
| model_fabric_groq_20260526T070131550071Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T070131550071Z.txt` |
| model_fabric_groq_20260526T203057424045Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T203057424045Z.txt` |
| model_fabric_groq_20260526T203657290137Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T203657290137Z.txt` |
| model_fabric_groq_20260526T203832372042Z.txt | RUNTIME | `04_RUNTIME/goals/model_fabric_groq_20260526T203832372042Z.txt` |
| indy_percyphon_hunch_subtleknife_binding.json | RUNTIME | `04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json` |
| indy_reads_adapter_registry.json | RUNTIME | `04_RUNTIME/indy_reads_adapter_registry.json` |
| 640e52a52ba4b4d3.txt | RUNTIME | `04_RUNTIME/indy_reads_extracted/640e52a52ba4b4d3.txt` |
| indy_reads_lora_stage.jsonl | RUNTIME | `04_RUNTIME/indy_reads_lora_stage.jsonl` |
| indy_reads_persona_config.json | RUNTIME | `04_RUNTIME/indy_reads_persona_config.json` |
| dual_engine_receipts.jsonl | RUNTIME | `04_RUNTIME/inference_os/dual_engine_receipts.jsonl` |
| preemption_receipts.jsonl | RUNTIME | `04_RUNTIME/inference_os/preemption_receipts.jsonl` |
| vram_llm_selection_bespoke_staging.json | RUNTIME | `04_RUNTIME/inference_os/vram_llm_selection_bespoke_staging.json` |
| ingestion_watchdog_done_count.txt | RUNTIME | `04_RUNTIME/ingestion_watchdog_done_count.txt` |
| korpus_result_stream_state.json | RUNTIME | `04_RUNTIME/korpus_result_stream_state.json` |
| llxprt_hook_audit.jsonl | RUNTIME | `04_RUNTIME/llxprt_hook_audit.jsonl` |
| task_block_0001_9ff8ea4daf2c_local.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0001_9ff8ea4daf2c_local.txt` |
| task_block_0001_local.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0001_local.txt` |
| task_block_0002_45a242f605e8_groq.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0002_45a242f605e8_groq.txt` |
| task_block_0002_groq.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0002_groq.txt` |
| task_block_0003_79bc2c7607c4_cohere.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0003_79bc2c7607c4_cohere.txt` |
| task_block_0003_cohere.txt | RUNTIME | `04_RUNTIME/model_audits/task_block_0003_cohere.txt` |
| task_block_0001_9ff8ea4daf2c_local_strict.txt | RUNTIME | `04_RUNTIME/model_audits_strict/task_block_0001_9ff8ea4daf2c_local_strict.txt` |
| task_block_0002_45a242f605e8_groq_strict.txt | RUNTIME | `04_RUNTIME/model_audits_strict/task_block_0002_45a242f605e8_groq_strict.txt` |
| task_block_0001_9ff8ea4daf2c_local_strict2.txt | RUNTIME | `04_RUNTIME/model_audits_strict2/task_block_0001_9ff8ea4daf2c_local_strict2.txt` |
| task_block_0002_45a242f605e8_groq_strict2.txt | RUNTIME | `04_RUNTIME/model_audits_strict2/task_block_0002_45a242f605e8_groq_strict2.txt` |
| board_effect_doctrine_latest.json | RUNTIME | `04_RUNTIME/observation_center/board_effect_doctrine_latest.json` |
| dev_journey_decision_points_latest.json | RUNTIME | `04_RUNTIME/observation_center/dev_journey_decision_points_latest.json` |
| hunch_hypertimeline_latest.json | RUNTIME | `04_RUNTIME/observation_center/hunch_hypertimeline_latest.json` |
| hunch_postgres_ingest_latest.json | RUNTIME | `04_RUNTIME/observation_center/hunch_postgres_ingest_latest.json` |
| model_invocation_audit_latest.json | RUNTIME | `04_RUNTIME/observation_center/model_invocation_audit_latest.json` |
| project2501_model_workshare_latest.json | RUNTIME | `04_RUNTIME/observation_center/project2501_model_workshare_latest.json` |
| working_reality_latest.json | RUNTIME | `04_RUNTIME/observation_center/working_reality_latest.json` |
| treelite_note_extracted.txt | RUNTIME | `04_RUNTIME/treelite_note_extracted.txt` |
| IMPORT_MANIFEST_2026_05_13.json | VAULT | `03_VAULT/drive_pull/maths/IMPORT_MANIFEST_2026_05_13.json` |
| cards.min.json | VAULT | `03_VAULT/external/leder_cards/cards.min.json` |
| changelog.json | VAULT | `03_VAULT/external/leder_cards/changelog.json` |
| errata.json | VAULT | `03_VAULT/external/leder_cards/errata.json` |
| faq.json | VAULT | `03_VAULT/external/leder_cards/faq.json` |
| main.84a1395bd2271d65.js | VAULT | `03_VAULT/external/leder_cards/main.84a1395bd2271d65.js` |
| meta.json | VAULT | `03_VAULT/external/leder_cards/meta.json` |
| styles.f7ae0b670f79e2b0.css | VAULT | `03_VAULT/external/leder_cards/styles.f7ae0b670f79e2b0.css` |
| manifest.json | VAULT | `03_VAULT/ingested_markdown/20260514T223602Z/manifest.json` |
| cas_ingest.jsonl | VAULT | `03_VAULT/journal/cas_ingest.jsonl` |
| src | OTHER | `src` |
| db_core.py | OTHER | `src/db_core.py` |
| spine.py | OTHER | `src/spine.py` |
| tests | OTHER | `tests` |
| __init__.py | OTHER | `tests/__init__.py` |
| data.json | OTHER | `tests/fixtures/demo_corpus/data.json` |
| simple_dev_order.txt | OTHER | `tests/fixtures/dev_orders/simple_dev_order.txt` |
| valid_receipt_bundle.json | OTHER | `tests/fixtures/golden_path/valid_receipt_bundle.json` |
| missing_constraint_receipt.json | OTHER | `tests/fixtures/matrix_trace/missing_constraint_receipt.json` |
| pass_with_incomplete_matrix_receipt.json | OTHER | `tests/fixtures/matrix_trace/pass_with_incomplete_matrix_receipt.json` |
| prose_only_matrix_receipt.json | OTHER | `tests/fixtures/matrix_trace/prose_only_matrix_receipt.json` |
| valid_receipt.json | OTHER | `tests/fixtures/matrix_trace/valid_receipt.json` |
| wrong_order_receipt.json | OTHER | `tests/fixtures/matrix_trace/wrong_order_receipt.json` |
| generate_poison.py | OTHER | `tests/generate_poison.py` |
| 01_utf8_with_random_nul.txt | OTHER | `tests/poison_drop/01_utf8_with_random_nul.txt` |
| 05_force_deadlock.py | OTHER | `tests/poison_drop/05_force_deadlock.py` |
| demo_product_snapshot.json | OTHER | `tests/snapshots/demo_product_snapshot.json` |
| test_abductive_db_os_gate.py | OTHER | `tests/test_abductive_db_os_gate.py` |
| test_abductive_db_os_ledger.py | OTHER | `tests/test_abductive_db_os_ledger.py` |
| test_abductive_next_move_engine.py | OTHER | `tests/test_abductive_next_move_engine.py` |
| test_absurd_abductive_ledger.py | OTHER | `tests/test_absurd_abductive_ledger.py` |
| test_absurd_chrono_worker_contract.py | OTHER | `tests/test_absurd_chrono_worker_contract.py` |
| test_absurd_consume_kernel_authorization.py | OTHER | `tests/test_absurd_consume_kernel_authorization.py` |
| test_absurd_corpus_bridge.py | OTHER | `tests/test_absurd_corpus_bridge.py` |
| test_absurd_gate.py | OTHER | `tests/test_absurd_gate.py` |
| test_absurd_graph_promotion_worker_contract.py | OTHER | `tests/test_absurd_graph_promotion_worker_contract.py` |
| test_absurd_intake_worker_contract.py | OTHER | `tests/test_absurd_intake_worker_contract.py` |
| test_absurd_momentary_flow.py | OTHER | `tests/test_absurd_momentary_flow.py` |
| test_absurd_next_move_engine.py | OTHER | `tests/test_absurd_next_move_engine.py` |
| test_absurd_queue_spine_contract.py | OTHER | `tests/test_absurd_queue_spine_contract.py` |
| test_absurd_real_work_loop_bootstrap.py | OTHER | `tests/test_absurd_real_work_loop_bootstrap.py` |
| test_absurd_remaining_worker_contract_alignment.py | OTHER | `tests/test_absurd_remaining_worker_contract_alignment.py` |
| test_absurd_river_worker_contract.py | OTHER | `tests/test_absurd_river_worker_contract.py` |
| test_absurd_river_worker_hard_fail.py | OTHER | `tests/test_absurd_river_worker_hard_fail.py` |
| test_activity_tree_ingest_dry_run.py | OTHER | `tests/test_activity_tree_ingest_dry_run.py` |
| test_adhd_slow_lane_divergence.py | OTHER | `tests/test_adhd_slow_lane_divergence.py` |
| test_anthropic_proxy_local.py | OTHER | `tests/test_anthropic_proxy_local.py` |
| test_bare_steel_doctrine.py | OTHER | `tests/test_bare_steel_doctrine.py` |
| test_bitloops_airlock_audit.py | OTHER | `tests/test_bitloops_airlock_audit.py` |
| test_bitloops_automation_loop.py | OTHER | `tests/test_bitloops_automation_loop.py` |
| test_bitloops_full_reingest_manifest.py | OTHER | `tests/test_bitloops_full_reingest_manifest.py` |
| test_blueprint_first_pseudolaw.py | OTHER | `tests/test_blueprint_first_pseudolaw.py` |
| test_board_effect_doctrine.py | OTHER | `tests/test_board_effect_doctrine.py` |
| test_boring_beast_runtime.py | OTHER | `tests/test_boring_beast_runtime.py` |
| test_bytewax_treelite_matrix.py | OTHER | `tests/test_bytewax_treelite_matrix.py` |
| test_canonical_graph_write_scanner.py | OTHER | `tests/test_canonical_graph_write_scanner.py` |
| test_case_dashboard_data.py | OTHER | `tests/test_case_dashboard_data.py` |
| test_case_packet_renderer.py | OTHER | `tests/test_case_packet_renderer.py` |
| test_case_workspace_isolation.py | OTHER | `tests/test_case_workspace_isolation.py` |
| test_chrono_conservation.py | OTHER | `tests/test_chrono_conservation.py` |
| test_chrono_source_trust_validator.py | OTHER | `tests/test_chrono_source_trust_validator.py` |
| test_claim_clusterer.py | OTHER | `tests/test_claim_clusterer.py` |
| test_claim_support_score.py | OTHER | `tests/test_claim_support_score.py` |
| test_content_store.py | OTHER | `tests/test_content_store.py` |
| test_contradiction_report.py | OTHER | `tests/test_contradiction_report.py` |
| test_contradiction_resolution.py | OTHER | `tests/test_contradiction_resolution.py` |
| test_decision_gated_graph_promotion.py | OTHER | `tests/test_decision_gated_graph_promotion.py` |
| test_decision_gated_memory.py | OTHER | `tests/test_decision_gated_memory.py` |
| test_demo_product_snapshot.py | OTHER | `tests/test_demo_product_snapshot.py` |
| test_dev_journey_decision_points.py | OTHER | `tests/test_dev_journey_decision_points.py` |
| test_dev_library_scan_wrapper.py | OTHER | `tests/test_dev_library_scan_wrapper.py` |
| test_dev_order_gate.py | OTHER | `tests/test_dev_order_gate.py` |
| test_dev_order_matrix_wrapper.py | OTHER | `tests/test_dev_order_matrix_wrapper.py` |
| test_diogenes_memory_gate.py | OTHER | `tests/test_diogenes_memory_gate.py` |
| test_entity_candidate_registry.py | OTHER | `tests/test_entity_candidate_registry.py` |
| test_event_candidate_registry.py | OTHER | `tests/test_event_candidate_registry.py` |
| test_export_bundle.py | OTHER | `tests/test_export_bundle.py` |
| test_fast_slow_lane_gate.py | OTHER | `tests/test_fast_slow_lane_gate.py` |
| test_full_system_soak_audit.py | OTHER | `tests/test_full_system_soak_audit.py` |
| test_goal_agent_packet.py | OTHER | `tests/test_goal_agent_packet.py` |
| test_goal_chain_telemetry.py | OTHER | `tests/test_goal_chain_telemetry.py` |
| test_goal_dev_control.py | OTHER | `tests/test_goal_dev_control.py` |
