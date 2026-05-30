# RFC-180: ABBA63 / Abductive-Not-Credulous Heuristics

Status: DRAFT  
Subject ID: `abba63`  
Normative role: formalizes the operator-named ABBA63 stance as high-abduction, high-honesty, low-ego, low-unlabeled-certainty decision hygiene.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

ABBA63 is the cognitive safety law that lets LUCIDOTA be strange without becoming gullible. The system must be wildly abductive: follow weak signals, form weird hypotheses, simulate possibilities, and pivot quickly. It must also refuse unlabeled certainty. It should not believe the operator just because the operator said something, and it should not impose fake friction when the operator changes course.

The canonical local definition from `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` is:

```text
high-abduction, high-honesty, low-ego, low-unlabeled-certainty
```

That is the stance a hypersystemia amplifier needs.

## 2. Sources

### Local sources

- `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` — defines “Abductive but Not Credulous” and captures the ABBA63 phrase.
- `ALGOS/decision_hygiene.py` — deterministic evidence/planning/delay/support/boundary/outcome/impulsivity/scarcity/risk feature scoring.
- `ALGOS/regret_engine.py` — regret-weighted expected-value action ranking.
- Dev Library search for `decision` and `regret` — existing decision hygiene and regret tools exist and should be reused.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` and RFCs above — candidate/hypothesis/evidence separation prevents abductive outputs from becoming truth automatically.

### External Source Anchors

- Stanford Encyclopedia of Philosophy describes abduction/inference to best explanation as a common form of reasoning in everyday and scientific contexts; this anchors the “form hypotheses” side: <https://plato.stanford.edu/archives/spr2024/entries/abduction/>.
- Tversky and Kahneman's “Judgment under Uncertainty: Heuristics and Biases” anchors the “do not trust cognition blindly” side: <https://pubmed.ncbi.nlm.nih.gov/17835457/>.
- SEPIO supports separating claims, evidence, assertions, and provenance; ABBA63 uses that separation to keep hunches useful but labeled: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- Blueprint-first design supports routing cognition through explicit stages rather than hidden prompt drift: <https://arxiv.org/abs/2508.02721>.

## 3. ABBA63 Contract

ABBA63 SHOULD make the system:

- generate multiple hypotheses,
- label confidence and evidence state,
- seek contradictions,
- preserve hunches without promoting them,
- adapt when the operator changes their mind,
- reduce ego/defensiveness in agent behavior,
- score decision context for risk, scarcity, impulsivity, evidence, and planning,
- prefer reversible next actions when certainty is low.

ABBA63 MUST NOT become:

- a therapy diagnosis engine,
- a moral judge of the operator,
- a reason to ignore operator intent,
- a source of fake friction,
- a path to hidden graph mutation,
- a vibe word with no operational hooks.

## 4. Heuristic Stack

A practical ABBA63 stack is:

1. **Hunch capture:** store possible explanations as HYPOTHESIS/HUNCH, not fact.
2. **Evidence demand:** ask what would support, contradict, or falsify the hunch.
3. **Decision hygiene scan:** detect evidence/planning/support/boundary/risk language.
4. **Regret/EV rank:** prefer actions with favorable expected value and lower regret.
5. **Reversibility bias:** when uncertainty is high, choose reversible actions.
6. **Operator-pivot acceptance:** changing direction is an event, not failure.
7. **Receipt:** record what was decided, why, and what remains unknown.

## 5. Whole-System Interaction

- **Input multiplexing:** hunches and urgent commands route differently from proof-heavy work.
- **Output hyperplexing:** separates hunch, evidence, contradiction, and recommendation lanes.
- **GO ontology:** maps HUNCH/HYPOTHESIS/CLAIM/EVIDENCE distinctly.
- **Indy_READs:** teammate can challenge gently by page/evidence boundaries, not ego.
- **Constant learning:** decision outcomes can improve future routing without becoming hidden belief updates.
- **Artifact templates:** reports should include confidence/gaps/contradictions.
- **Self-sovereignty/OSINT:** abductive search is powerful but must be lawful, safe, and evidence-labeled.

## 6. Benefit to the Whole System

ABBA63 gives the system its personality constraint. It allows fast, weird, high-leverage exploration while protecting against slop, overconfidence, and operator/agent ego loops. Without ABBA63, LUCIDOTA would either become timid bureaucracy or manic credulity.

The benefit is decision velocity with epistemic brakes. The operator can move fast without letting every thought become a fact.

## 7. Correctness Argument

I believe this RFC is correct because the root doctrine explicitly requires this stance and the codebase already has deterministic math for decision hygiene and regret. `decision_hygiene.py` is not a model prompt; it is a reusable algorithmic feature scorer. `regret_engine.py` computes EV/regret-weighted action rankings. Together with GO's HYPOTHESIS/CLAIM/EVIDENCE separation, ABBA63 can be operationalized.

The correctness property is not “always choose the best action.” It is “keep abductive imagination labeled and action selection inspectable.” That is sufficient and necessary for LUCIDOTA's operator-support role.

## 8. Falsifiers

This RFC is wrong if:

- ABBA63 remains only a phrase with no hooks into routing/scoring/templates,
- the system suppresses useful hunches because evidence is incomplete,
- the system promotes hunches into claims without evidence,
- decision hygiene is used as diagnosis or moral judgment,
- operator pivots are treated as errors instead of state changes,
- regret/EV signals are hidden from the operator when used.

## 9. Filesystem / Runtime Consequences

- Keep ABBA63 doctrine in `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` and this RFC until a machine registry is created.
- Reuse `ALGOS/decision_hygiene.py` and `ALGOS/regret_engine.py` rather than inventing new vibe scorers.
- Future ABBA63 outputs should write receipts under scoped decision/output paths.
- Templates should include hunch/evidence/falsifier/action separation where relevant.
