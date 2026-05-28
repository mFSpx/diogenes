# RFC-010: Slop Laws / Anti-Bullshit Engineering

Status: DRAFT  
Subject ID: `slop_laws`  
Normative role: defines how LUCIDOTA rejects fake-working code, fake certainty, and hidden authority.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA must treat slop as an epistemic safety failure, not as an aesthetic complaint. Slop is any artifact that makes the system less honest about what it knows, what it can prove, what it changed, or what authority it has.

This thesis is true because LUCIDOTA's domain is not ordinary software output. It is live intelligence work: ingesting imperfect evidence, coding while operating, learning from diffs, producing reports/letters, and preserving uncertainty. In that domain, the worst bug is not a crash. The worst bug is unlabeled certainty that quietly enters the graph, a report, a workflow, or the operator's reasoning loop.

## 2. Sources

### Local sources

- `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` defines the one law: be honest about the relationship between code and truth.
- `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md` requires visible workflow paths, bounded model use, and receipts over claims.
- `scripts/slop_audit_law.py` is the executable simplicity audit hook.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` defines authority/mutation classes and graph-promotion requirements.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md` defines reuse/promotion law so indexed tools do not become production authority and production work does not reinvent existing wheels.
- `00_PROJECT_BRAIN/RFCS/RFC-000-MASTER-THESIS-PROGRAM.md` defines RFC evidence and falsifier requirements.

### External sources

- RFC 2119 gives the RFC program a disciplined vocabulary for `MUST`, `SHOULD`, and `MAY`: <https://datatracker.ietf.org/doc/rfc2119/>.
- W3C PROV-O is a provenance standard for describing how data came to be and what activity/agent/entity relationships produced it: <https://www.w3.org/TR/prov-o/>.
- SEPIO supports treating claims, evidence, and provenance as integrated first-class concepts: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- `Blueprint First, Model Second` supports the local rule that workflow control belongs in inspectable code/schema/templates/queues, not hidden model improvisation: <https://arxiv.org/abs/2508.02721>.
- PostgreSQL `SKIP LOCKED` documentation supports the system's queue-spine choice for concurrent work consumers while warning that skipped locked rows are queue-like, not a general consistent view: <https://www.postgresql.org/docs/current/sql-select.html>.
- NIST's McCabe/cyclomatic-complexity material and Sonar's Cognitive Complexity both support the real engineering point that understandability is more than line count, but size/branching are valid warning signals: <https://www.nist.gov/publications/structured-testing-software-testing-methodology-using-cyclomatic-complexity-metric> and <https://www.sonarsource.com/resources/cognitive-complexity/>.
- Google Testing Blog's small/medium/large test framing supports tiered tests rather than pretending huge indiscriminate test counts equal confidence: <https://testing.googleblog.com/2010/12/test-sizes.html>.

## 3. What Counts as Slop

A LUCIDOTA artifact is slop if any of these are true:

- It claims a capability that is not wired.
- It hides workflow control in prose or prompt behavior.
- It treats model output as truth or command authority.
- It writes or implies graph truth without evidence refs and authority class.
- It invents a new tool without checking the Dev Library.
- It passes weak tests that only prove importability or vibes.
- It creates outputs that cannot be traced to source, hash, command, or receipt.
- It mixes lab/reference material into the active spine without a boundary.
- It uses cute naming as a substitute for a `what / why / how / authority` contract.
- It adds security, proof, logging, or audit machinery that freezes live work, creates redundant gigabytes, or slows low-risk/reversible actions without proportional risk reduction.
- It reinvents an existing well-supported wheel without a sovereignty reason, an objective improvement, a hard system need, or explicit operator intent to build a better version.
- It gives a made-up local nickname the same status as a real established concept.
- It refuses lawful/sensitive intelligence work through generic moralizing friction instead of naming the actual blocked risk and the safe allowed path.
- It answers current/time-sensitive questions from stale memory instead of checking current sources.
- It ignores context-window pressure until the working memory silently degrades.

Slop labels:

- `VIBE`: sounds coherent but lacks proof.
- `SLOP`: vague, bloated, or fake-clear.
- `FANTASY`: claimed runtime/capability is not actually wired.
- `REINVENTION`: ignores existing reusable parts.
- `UNTRACED`: no source/hash/receipt/test.
- `DANGEROUS`: hidden irreversible or external effect.
- `AUTHORITY_LEAK`: candidate/inference/model output promoted beyond its class.
- `SECURITY_THEATER`: protection/logging/audit slows or bloats the system without buying real safety.
- `OPERATIONAL_FRICTION`: process, refusal, testing, or hygiene wastes operator time without improving truth/risk.
- `TERMINOLOGY_FOG`: invented name presented as if it were a real domain concept.
- `TEMPORAL_SLOP`: stale answer to a current-status question.
- `CONTEXT_SLOP`: no warning/summary when context pressure threatens continuity.

## 3.1 Proportionality Law

The right rule is not “maximum gates everywhere.” The right rule is proportional proof at the right boundary.

- Hot-lane work needs small fast checks, hashes, and receipts.
- Slow-lane work can afford deeper audits.
- Graph materialization, external effects, destructive actions, and privacy-sensitive exports need strong gates.
- Reversible experiments, local drafts, and lab simulations need low friction and clear labels.

If a safety mechanism makes the system sluggish, frozen, noisy, or unusable, it has become slop. A gigabyte of redundant logs is not more truth than a small receipt that proves the invariant.

## 3.2 Reuse / Wheel Law

Reinvention is slop by default. LUCIDOTA should build its own thing only when at least one condition is true:

1. dependency would make the system less sovereign,
2. local implementation can be objectively better for this system,
3. the system needs a specific behavior the existing tool does not provide,
4. the operator explicitly wants to build a better version.

Otherwise: use the known tool, library, protocol, standard, or existing Dev Library artifact. If a custom version is still chosen, the receipt should say why.

## 3.3 Terminology Truth Law

An assistant MUST distinguish:

- real established concept,
- local project name,
- metaphor/design handle,
- newly invented shorthand,
- hypothesis about similarity.

If the operator describes multiplexing, call it multiplexing. If the operator describes something novel, say it is a local design pattern or pending name, not a fake established concept. Presenting “Horizontal Mumbo Jumbo Generator” beside a real term as if both are equally true is slop.

## 3.4 Code Size / Complexity Reality Check

There is no universal scientific law that says every function must be under 100 lines. Real maintainability depends on cohesion, branching, coupling, naming, tests, data shape, and cognitive complexity.

For LUCIDOTA, these are good operational thresholds:

- **~100 LOC**: reasonable upper target for most individual functions/tools; smaller is usually better.
- **~500 LOC**: large for a single script/tool; review for splitting, data-driven design, or reuse.
- **~2000 LOC**: serious justification required unless it is legacy, generated, vendored, or a consciously bounded monolith under active reduction.

Line count is a smoke alarm, not the fire itself. A 60-line knot can be worse than a 300-line flat table-driven script. But when code grows past these thresholds, the burden of proof shifts to the code.

## 3.5 Test Signal Law

More tests are not automatically more truth. A once-daily hygiene suite is useful. A giant pile of low-signal tests used to say “look, work” is operational slop.

Tests should be tiered:

- focused invariants for changed code,
- smoke checks for active runtime paths,
- integration checks for contracts/boundaries,
- daily/full hygiene where expensive breadth belongs.

The confidence claim must name which tier ran. “14,000 working tests” is not impressive if they are redundant, flaky, slow, or unrelated to the change.

## 3.6 Temporal and Context Law

If the user asks for current status, latest/best-now information, prices, versions, laws, standards, schedules, leaders, or recommendations that change over time, stale memory is slop. Search or verify current sources.

If context pressure is high enough that continuity may degrade, the assistant should warn and emit/update a compact handoff summary before losing the thread. Silent context loss is slop.

## 3.7 Deterministic Workflow Supremacy Law

Operator law, verbatim: “Never have an LLM do what smart deterministic workflows and hardy design could do 100% correct and 20,000 times faster.”

For LUCIDOTA, using a model where a parser, schema check, hash comparison, queue worker, table-driven router, deterministic algorithm, or explicit workflow can do the job exactly is slop from the top. This is not model-zero doctrine. LLMs must still be used when they are the right tool: bounded interpretation, extraction from genuinely messy language, synthesis, drafting, adversarial ideation, ambiguity handling, and code generation. They are not permitted to replace exact computation, provenance checks, routing gates, status accounting, receipts, or graph-promotion invariants.

Default routing order:

1. deterministic code/workflow/schema/hash,
2. existing reusable algorithm or Dev Library artifact,
3. local model when language/model judgment is actually needed,
4. optional cloud model when policy, privacy, cost, latency, and receipts justify it.

Underusing models where language judgment is genuinely required is also slop: brittle regex-only behavior, fake determinism, and dropping useful messy signals are not sovereignty. Any exception in either direction must name the reason deterministic handling is insufficient or model handling is unnecessary.

## 4. Whole-System Interaction

Slop law is not a separate lint style. It is the immune system for every organ:

- **Main Spine:** blocks hidden mutation and authority leaks.
- **ETL/Krampus/KORPUS:** prevents raw evidence from becoming interpreted truth during intake.
- **Diogenes Kernel:** forces command envelopes and permission checks for privileged effects.
- **ABSURD:** turns workflow into rows/events/dead letters instead of prompt memory.
- **Dev Library:** prevents reinvention by requiring search/reuse before new tools.
- **Local LLM Fabric:** keeps model calls bounded, receipted, and draft/candidate-scoped.
- **PercyphonAI:** remains deterministic scaffolding, not entity truth.
- **Indy_READs:** remains teammate/synthesis identity, not disposable autonomous authority.
- **Ontology/Graph:** requires OBJECT/EVENT/EDGE candidates plus evidence/authority before promotion.
- **Artifact Templates:** require source-anchored output rather than freestyle prose claiming finality.

The instruction-hygiene / language-membrane standard is systemwide, not confined to GOALS. LUCIDOTA should converge toward one authoritative spine, many thin interfaces, ontology-shaped packets, deterministic-first execution, and receipt-fed self-teaching/self-audit/self-red-team loops. A ~100 LOC target remains the normal pressure for new helpers, but larger code is not automatically slop when a written justification shows that the size protects cohesion, feature completeness, performance, safety, or reuse.

## 5. Benefit to the Whole System

Anti-slop rules make LUCIDOTA faster because they remove fake shortcuts. A tool that lies about being done creates rework. A model output that silently becomes truth creates epistemic debt. A clever script with no receipt creates a mystery. By forcing evidence, authority, and receipts, the system can remain abductive without becoming credulous.

This benefits the operator's hypersystemic workflow because it preserves speed while making state inspectable. It also protects speed from fake safety: gates that do not buy real risk reduction are not sacred; they are candidates for removal or relocation to the slow lane.

```text
wild idea allowed
uncertain hypothesis allowed
model draft allowed
operator mind-change allowed
unlabeled truth claim not allowed
```

## 6. Correctness Argument

The anti-slop law is correct if LUCIDOTA's purpose is self-sovereign OSINT and exocortex work. In that domain, outputs affect belief, investigation direction, reports, and possible real-world action. Therefore, the system must optimize for honest state transitions over pleasing prose.

The law is also correct mechanically because it maps to existing executable hooks:

- code size/complexity can be audited by `scripts/slop_audit_law.py`,
- graph mutation can be scanned by `scripts/canonical_graph_write_scanner.py`,
- authority can be checked by `scripts/spine_authority_checker.py`,
- workflow completion can be proven by receipts under `05_OUTPUTS/`,
- RFC completeness can now be checked by `scripts/rfc_program_check.py`.

The law is not anti-creativity. It explicitly allows weirdness, hypotheses, simulations, and live-coding experiments. It only requires that they declare their relationship to truth and authority.

## 7. Falsifiers

This RFC is wrong if:

- strict slop gates prevent useful work more often than they prevent false claims,
- the system cannot distinguish candidate/inference/truth states in practice,
- receipt generation, security, testing, or audit becomes performative paperwork rather than real proof,
- the operator's live coding loop becomes too slow to use,
- tests become another weak ritual instead of proving invariants.
- agents invent names instead of naming real concepts or saying “local metaphor,”
- stale answers are given to current-status questions.

If any falsifier appears, the correction is not to abandon anti-slop law. The correction is to improve the gate so it catches real dishonesty without blocking useful uncertain work.

## 8. Filesystem / Runtime Consequences

- Minimum doctrine stays in five docs only.
- Deep arguments live in `00_PROJECT_BRAIN/RFCS/`.
- Dev Library search uses `scripts/dev_library_scan.py` for new operator-facing workflows.
- ROOT-414 and Ahoy stay externalized as reference/lab material.
- Any new production component must state `what / why / how / authority / failure / receipt`.
- Weak tests should be replaced with invariant tests or removed from production confidence claims.

## 9. Current Verification

Current executable evidence:

```bash
python3 scripts/rfc_program_check.py
python3 scripts/dev_library_scan.py --check
python3 scripts/lucidota_status_ledger.py --check
python3 -m pytest tests/test_rfc_program_check.py tests/test_dev_library_scan_wrapper.py tests/test_instruction_authority_registry.py tests/test_slop_audit_law.py -q
```

The latest run before this RFC recorded PASS for the RFC program, Dev Library manifest, status ledger, and 8 focused tests.
