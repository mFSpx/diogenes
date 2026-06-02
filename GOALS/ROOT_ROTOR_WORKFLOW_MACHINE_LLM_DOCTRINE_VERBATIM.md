# Root-Rotor Workflow Machine / Deterministic-First LLM Doctrine

Captured: 2026-06-02
Authority: operator directive during Operation Root-Rotor.

```text
Everything is workflows — except the things workflows operate on.

That is the clean split.

OBJECTS are nouns.
WORKFLOWS are verbs.
BOXES are addresses/policies.
EVENTS are changes.
RECEIPTS are proof.
EDGES are relationships.
STATE is the latest reducible view.
LEDGER is history.
LLMs are optional tools, not system organs.

The real LUCI law:

If it can be solved by:
- SQL
- regex
- hash
- parser
- schema validation
- graph traversal
- deterministic classifier
- Treelite/XGBoost
- filesystem stat
- OCR/extraction
- embedding search
- threshold/rule
- queue retry
- checksum/diff
- cgroup/systemd telemetry
- known adapter

then it does not get an LLM.

LLM only enters when the system hits one of these:

LLM_ALLOWED_REASON:
- ambiguous human language
- messy summarization
- entity/claim extraction needing judgment
- conflict explanation
- hypothesis generation
- prompt/dialogue response
- code/design review
- natural-language transformation
- low-confidence router fallback
- human-facing synthesis

And even then:

LLM is boxed.
LLM gets typed input.
LLM returns typed output.
LLM cannot mutate canon directly.
LLM output is a claim/proposal until validated.
LLM run must produce a receipt.

So the database should enforce this:

workflow_registry
- workflow_id
- verb
- input_object_types
- output_object_types
- deterministic_first BOOLEAN
- llm_allowed BOOLEAN
- llm_required BOOLEAN DEFAULT false
- allowed_models
- validator_workflow_id
- receipt_type
- promotion_policy

The slop-killer rule:

llm_required should be rare as fuck.
llm_allowed is not permission to be lazy.
deterministic_first must be the default.

Better core doctrine:

LUCI is a workflow machine over typed objects.
The DB is the nervous system.
The ledger is memory.
The graph is meaning.
The queue is muscle.
The models are specialist organs.
The LLM is a mouth/reader/guesser — useful, expensive, contained.

Actual build sentence:

Every incoming thing becomes an object, every object enters a workflow, every workflow emits events and receipts, every event may update graph/state, every promotion to canon requires validation, and every LLM call must justify why deterministic machinery was insufficient.

That is the honest architecture. Not “AI app.” Not “agent swarm.” Not “chatbot with tools.”

It is:

deterministic workflow OS with optional LLM judgment adapters

Or more brutally:

Postgres/Absurd workflow engine first.
Graph/canon/receipt machine second.
LLMs last.

The prompt/spec line you want:

Build LUCI as a typed workflow machine. Treat almost every operation as a workflow over objects, boxes, events, edges, receipts, and state. Default to deterministic execution. An LLM is never the first tool unless the input is natural-language dialogue or ambiguity itself is the object. If SQL, parser, hash, regex, graph traversal, schema validation, Treelite, River, system telemetry, adapter logic, or existing workflow state can answer, no LLM call is permitted. LLM outputs are non-canonical proposals requiring validation receipts before promotion. Ingest, extract, classify, route, promote, quarantine, train, recap, diff, audit, and repair are workflow families, not folders. Boxes are addresses/policies. Objects are nouns. Workflows are verbs. Receipts prove changes. Ledger remembers. Graph relates. Runtime executes. Canon only changes through validated events.
```

## Implementation target

Encode this doctrine as queryable canon and workflow registry policy. Keep LLM calls boxed. Require deterministic-first workflow metadata. Treat LLM output as proposal until a validator workflow and receipt promote it.
