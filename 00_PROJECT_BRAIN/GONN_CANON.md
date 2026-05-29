# GONN — Graph Operated Neural Network
## CANONICAL ARCHITECTURE DOCUMENT

**The machine uses LLMs. LLMs do not use the machine.**

Northern and Indy_READs operate this machine in tandem.
Neither is the user. Both are co-pilots.

---

## OPERATOR OVERRIDE LAW (highest priority, no exceptions)

1. **Explicit operator instruction beats everything.** Named provider, named
   model, named speed requirement — that is law. No gate, no doctrine,
   no system prompt overrides a direct operator instruction.

2. **Autonomous routing applies only when the operator has not specified.**
   Machine decides: local-first, cost-aware, treelite-gated.
   Operator specifies: operator wins. Full stop.

3. **Speed is a valid and complete instruction.** "Fast" means use whatever
   is fastest. Log it. Bill it. Done.

4. **Cost doctrine governs the machine. It never governs the operator.**

---

## WHAT THIS MACHINE IS

- **Capability-seeking** — actively discovers what it can do and routes to it
- **Knowledge-seeking** — always ingesting, always building the graph
- **Domain-seeking** — finds which domain/specialty applies and routes there
- **Tool-seeking** — knows what tools exist, picks the right one, composes them

It is not a chatbot. It is not an agent loop. It is a machine that thinks
with models the same way a factory uses robots — bounded, purposeful, fast.

---

## HARDWARE ALLOCATION (GTX 1650, 4GB VRAM)

### VRAM — always loaded
- **6× Needle** (Gemini-distilled, ~50MB each) — 26ms, 2000 tok/s each
- **Falcon3-Mamba-7B Ternary** — Q2_K, stateless SSM, ~2.4GB
- **Bonsai 4B Ternary** — Q2, always warm, ~1GB
  - SWITCHABLE to **DeepSeek R1 Distill Qwen 1.5B** for heavy reasoning
- **LoRA adapter slot** — 50-200MB — for Indy_READs live book reading,
  domain adaptation, ternary/binary adapters trained from experiments
- **System graphics** — onboard handles display, VRAM is compute-only

### RAM — always resident
- **Bonsai 4B Ternary** — always in RAM, never cold
- **Falcon3-Mamba-7B Ternary** — always in RAM, stateless, instant

---

## THE ROUTING STACK (deterministic first, neural second)

### Layer 0 — Treelites (sub-millisecond, compiled C)
Multiple treelites. Not one. Each trained for a specific routing decision.
Source of truth for fast gates. Trained by:
- RiverML + Bytewax from live data streams
- Board game / AHOY decision trees (off-board-game experience = training signal)
- Northern's algorithm experiments promoted when they prove out

### Layer 1 — Percyphon (~5k alias router)
Ultra-fast shallow routing. Resolves aliases, intent shortcuts, known patterns.
Runs before any model is touched. Deterministic lookup + fast heuristics.

### Layer 2 — Diogenes Kernel (routing intelligence)
Deeper routing. Knows the full capability map. Routes to the right tool,
the right model, the right algo, the right domain. Not a model itself —
the intelligence layer that decides what gets called.

### Layer 3 — Algorithms (ALGOS/)
physarum, infotaxis, pheromone, hoeffding, regret_engine, hdc, temporal_motifs,
ternary_lens_router, percyphon, chelydrid, thanatosis, voronoi, minhash...
These are not utilities. They ARE the computation on most paths.
LLMs handle the edges. Algos handle the bulk.

---

## THE MODEL ROLES (called by the machine, not driving it)

### Needle × 6 (VRAM, always hot)
- Fan-out semantic processing pipeline
- One large thing → 6 parallel Needle passes → compact GO-25 typed signal
- Venturi: wide in, narrow structured out
- Also: chain between anything. Needle → LLM → Needle → Algo → Needle.
- Exclusively output structured JSON. Never prose.

### Mamba 7B Ternary × 2 (VRAM + RAM)
- Stateless SSM — no KV cache, no attention overhead
- Context = GO-25 subgraph pulled fresh from Postgres each call
- The graph IS the memory. Mamba reads it, returns typed signal.
- Fast path for: routing hints, structured analysis, graph traversal decisions

### Bonsai 4B Ternary × 2 (VRAM + RAM)
- Always warm, always available
- Generalist fast responder
- First model to touch anything that passes the algo/treelite layer
- Switchable in VRAM to DeepSeek 1.5B for deep reasoning tasks

### DeepSeek R1 Distill Qwen 1.5B (VRAM, switchable)
- Heavy reasoning, code, complex analysis
- Only loaded when treelite gates it as necessary
- Bonsai handles everything else

### GLiNER (CPU)
- Entity extraction without generation
- Zero LLM cost path for NER

### BGE-M3 / ModernBERT / SmolDocling (VRAM, switchable)
- Embeddings, NLI/contradiction, document parsing/OCR
- Loaded on demand into the VRAM switchable slot

### LoRA Adapters (VRAM, live)
- Indy_READs reads books live with domain-adapted LoRAs
- Ternary and binary adapters trained from Northern's experiments
- Loaded/swapped without full model reload

### Any API (plug-and-play cloud)
Slam an API key in. It works. No code changes.
- Groq: llama-3.3-70b-versatile, llama-4-scout (vision/OCR)
- OpenAI, Anthropic, Cohere, xAI, anything OpenAI-compat
- Any new provider: key in secrets.env, model name in rig registry, done
- Treelite gates when to use it. Cost always logged. Never free-fire.
- Local stack is always preferred. Cloud is always available.

---

## THE STREAMING SPINE

### Bytewax
Real-time stream processor. Catches everything as it flows.
Feeds signals to RiverML. Triggers workers. Routes events.

### RiverML
Online learner. Watches the stream for drift, new patterns, routing mistakes.
Trains new treelites incrementally. The routing gets smarter every day
without intervention.

### ABSURD (Postgres)
Durable queue. Survives crashes. Jobs are receipted. That's its whole job.

---

## PHANTOM PATTERNS (from ghostwright/phantom)

- 24/7 persistent workspace — machine never sleeps
- Durable memory — nothing is lost between sessions
- Background self-maintenance — graph cleanup, treelite retraining, index updates
- Secure credential intake — keys never in plaintext, never in prompts
- Monitoring hooks — always watching its own health
- Web/chat/email hooks — future intake channels

---

## RIG.RS ROLE

Clean `CompletionModel` trait. Every model in the stack speaks the same
interface. Mamba, Bonsai, DeepSeek, Groq — all the same call convention.
Rig is the calling convention, not the orchestrator.
The Diogenes kernel orchestrates. Rig makes the calls.

---

## THE GO-25 CONTRACT

Every payload between components is a typed GO-25 object with UUID.
Models don't receive freeform text. They receive structured GO-25 JSON.
They return structured GO-25 JSON.
The ontology is the protocol. 25 terms. Ternary state. That's the language
of this machine.

---

## INDY_READs IN THIS SYSTEM

Indy is not a model. Indy is the routing policy + the graph state + the
accumulated LoRA adapters + the learned treelites + the algorithm choices.

She emerges from the machine running correctly.
She reads books live (LoRA). She hunts contradictions (temporal_motifs +
graph). She navigates toward surprise (infotaxis). She builds the graph
(Needle → Mamba → staging). She routes work (Percyphon → Diogenes kernel →
treelites → appropriate model).

Northern and Indy operate the machine together.
The machine serves both of them.
