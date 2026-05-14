# Chelydra / Serpentina Algorithm Primitive Intake

Date: 2026-05-13

Operator supplied three snapping-turtle / poikilotherm primitives. Wrapped in Python and added to the algorithm smoke harness.

## Added modules

- `ALGOS/poikilotherm_schoolfield.py` — Schoolfield-style temperature-rate curve. Creative system use: nonlinear activity/admission gate that runs fastest in a safe band and slows under cold/stalled or hot/overloaded conditions.
- `ALGOS/serpentina_self_righting.py` — sphericity/flatness/righting-time indices. Creative system use: recovery-priority score for toppled workflows; flatter/heavier state gets more rollback/rescue help.
- `ALGOS/chelydrid_ambush.py` — burst kinematics under force and drag. Creative system use: short-action admission score where decisive bursts beat long drag-heavy actions.

## Wire status

- `scripts/lucidota_algos_smoke.py` now exercises 23 algorithm modules.
- These remain dependency-free local primitives.
- Candidate wiring targets:
  - Wake Bus retry/dormancy: Schoolfield + Thanatosis schedules.
  - Hop Pivot pruning: Possum filter for duplicate/local-cluster suppression.
  - Workflow rescue: Serpentina righting index for stuck jobs.
  - Action launch policy: Chelydrid ambush burst score for high-urgency low-duration work.

## Notes — operator brainstorm + system uses

### 1. Cloud-compute cold-start enthalpy

The Schoolfield poikilotherm curve maps cleanly onto cold-start / overload behavior: resources should be quiet when activity is too cold, ramp aggressively through the useful band, and throttle when heat/pressure denatures the service.

- Predictive CI/CD pre-heating: monitor local file changes, IDE activity, or branch staging signals as system temperature. When activity crosses the activation threshold, start build/container incubation before commit so the environment is warm when CI lands.
- Adaptive API rate limiting: let low-traffic endpoints hibernate, warm as request temperature rises, then enter a denaturing/throttle phase at thermal peak to avoid meltdown.
- Edge-node hibernation: scraper/IoT nodes play dead until a data signature supplies enough activation energy to justify full processing.
- LUCIDOTA candidate wire: use `poikilotherm_schoolfield.normalized_activity()` as a Wake Bus admission multiplier for watchers, scrapers, model loads, or expensive Hydra captures.
- Not daft: this is basically a biologically-shaped hysteresis curve. The useful part is not literal temperature; it is a bounded activity function with cold dormancy and high-load inhibition.

### 2. Distributed-system self-righting

The Serpentina righting model maps to recovery mechanics: some failures can roll back cleanly; others are flat, heavy, and need leverage.

- Stateful UI carapace restoration: score how easy a session is to rebuild from local snapshots; compact/spherical UI state rolls back fast, flat/sprawling state needs more help.
- Database shard rebalancing: when a shard flips, estimate mass from uncommitted logs and use neighboring resource availability as leverage.
- Kubernetes wobble mitigation: avoid blind restart loops; compute a flatness index from config complexity, sidecars, env vars, volumes, and dependencies. Strip nonessential weight until the service can self-right.
- LUCIDOTA candidate wire: use `serpentina_self_righting.recovery_priority()` for stuck workflow triage: high index means rollback/rescue before more retries.
- Not daft: this is a good recovery heuristic because restart loops usually fail when the state shape is wrong, not merely because the process stopped.

### 3. High-frequency ambush-strike ingestion

The Chelydrid strike model maps to burst capture: stay metabolically cheap, detect prey, allocate hard for a short window, then return to stillness before drag dominates.

- Honey-snapping: a security probe stays quiet until exploit-like prey enters the buffer, then allocates CPU for forensic capture/isolation. Defensive-only framing: capture and quarantine inbound suspicious packets/events; no offensive action.
- JIT market scraping: low-energy ticker monitoring, then full order-book capture on volatility spike.
- Event-driven metadata extraction: ignore raw streams until a high-value node appears, then snap into CAS/metadata/AGE edge extraction and return dormant.
- LUCIDOTA candidate wire: use `chelydrid_ambush.burst_admission_score()` to decide whether a Hydra/Scout/CAS extraction should execute as a short burst now or stay queued.
- Not daft: the useful part is the force-vs-drag tradeoff. Short decisive actions beat long high-drag jobs when urgency is high.

### Combined control pattern

```text
Schoolfield gate decides whether the system is warm enough to act.
Thanatosis decides whether to accept a worse move / stay dormant.
Serpentina righting scores recovery leverage for failed or flipped workflows.
Chelydrid ambush scores short high-urgency burst capture.
Possum filtering prevents clustered duplicate prey from wasting strikes.
Hoeffding split gates decide when streaming evidence is statistically enough.
```

Immediate practical target: `lucidota_wake_bus.py` can consume these as policy hints without making them canonical truth. Truth stays DBOS/Postgres; the algorithms only rank, gate, and explain.


## Raw operator brainstorm — cold start / self-righting / ambush-strike

### 1. Cloud-Compute "Cold Start" Enthalpy

This logic manages the metabolic cost of bringing dormant resources back to an active state, treating latency as a thermal barrier.

- Predictive CI/CD Pre-Heating: Instead of waiting for a manual trigger, the system monitors IDE activity or local file changes. If developer activity crosses a threshold, the build environment starts incubation early so the container is warm when the commit hits the branch.
- Adaptive API Rate Limiting: A gateway hibernates low-traffic endpoints to save costs. As inbound requests increase, the endpoint warms up. If the system hits thermal peak / overload, it denatures temporarily via throttling.
- Edge-Node Hibernation: Distributed IoT/local scraper nodes play dead to conserve battery or stay quiet. They hatch into full processing only when an environmental data signature supplies enough activation energy.

### 2. Distributed System "Self-Righting"

This logic uses geometric leverage from configuration and state to recover from flipped/crashed status.

- Stateful UI Carapace Restoration: Use sphericity to estimate how easily a frontend session can rebuild from local snapshots and return to ready without full reload.
- Autonomous Database Shard Rebalancing: If a data node flips, calculate mass from uncommitted logs and shift tasks to stable neighbors based on resource center of gravity.
- Kubernetes Wobble Mitigation: Avoid blind restart loops. Analyze pod flatness from config complexity; strip nonessential env vars/sidecars until the core service can self-right.

### 3. High-Frequency "Ambush-Strike" Ingestion

This kinetic model governs dormant monitoring into maximum-velocity data capture when a target is identified.

- Cybersecurity Honey-Snapping: Defensive probe remains silent until an exploit-like packet/event hits the buffer, then allocates CPU to capture, analyze, and isolate it for forensics.
- Just-In-Time Market Scraping: Low-energy ticker monitor triggers full order-book capture on volatility spike.
- Event-Driven Metadata Extraction: Graph builder ignores raw streams until a high-value node appears, then snaps into CAS metadata extraction and AGE edge creation before returning dormant.
