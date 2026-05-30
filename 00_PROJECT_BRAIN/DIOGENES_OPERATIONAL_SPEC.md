# DIOGENES OPERATIONAL SPEC — the north star for "system operational"

> Source: operator (Northern.Strike) directive, 2026-05-29. This is the canonical
> definition of DONE for the DIOGENES build. Captured verbatim-in-intent by the
> Orchestrator. No stubs, no gaps, no handwaving. Claims require evidence.

## 0. Roles & operating law
- **Indy_READs (Opus) = Orchestrator + Planner.** Designs, sequences, reviews. Does not hand-build sprawling code.
- **Groq + Cohere + Cerebras + Claude = the hands** (cloud lanes, API-keyed). Bulk generation, extraction, audit fan-out.
- **Local model suite** (DeepSeek/Mamba/Bonsai/Needle×6/BGE/ModernBERT/SmolDocling/GLiNER) = fully at the orchestrator's disposal *once embedding completes*. Until then, reserved for the embed crush.
- Everything wires through **rig.rs** (uniform CompletionModel) + **llama.cpp**, and **everything interacts with the Dynamic Governor (the guvna)**, all surfaced through **`./claw`**.
- Non-negotiable: nothing runs live that isn't *functionally awesome*. ALL schemas online + audited; ALL scripts audited; adversarial audit suites + daily schedules exist; gaps/stubs/handwaving are rejected.

## 1. INGESTION / ETL (active mission)
- **100% of legacy files — ALL of them — fully, properly ingested AND promoted ONTO the GO-25 graph** (not just `corpus_chunk`; through the promotion gate to OBJECT/EVENT/EDGE).
- **ABSURD ingestion workflows**, per-artifact-type decision tree: *What is the artifact? Where does it go? How do we handle it?* One route per format family (pdf/md/txt/docx/odt/audio/video/image/zip/archive/db/email/code/receipt/model).
- **ETL ANY artifact** thrown at the system. Recursively unpack ALL archives. Keep the graph populated continuously.

## 2. `./claw` — FINISHED
- Direct API calls to **Groq / Cohere / Cerebras / Claude** wired in (keys from `~/.config/lucidota/secrets.env`).
- **All local models** wired in.
- Orchestrated via **rig.rs** + **llama.cpp** (+ anything missed).
- **All of it interacts with the Dynamic Governor**, wired directly into `./claw`.

## 3. UX — conversational, deterministic routing (no mandatory slash commands)
- Slash commands available but never required. Interacting feels like **chatting with a friend**.
- **User inputs deterministically routed**: fast/slow lanes, pub/subs, algorithms, routers, verbosity/translation.
- On input, the text hits the **graph, the DB, classifiers, regex, GLiNER** — whatever works — to extract **intent + specifics**.
- **Verbatim inputs AND keystrokes are properly logged. Outputs logged too.**
- **Side input-cache:** accumulates to a flush-size; **referenceable by any curious worker** during accumulation; on every flush it makes a call with **everything in it** as a double backup (lose nothing).

## 4. OUTPUT multiplexer — better than any AI chat
- Outputs **multiplexed + sequenced** across: **Jinja/templating (Rust: minijinja/tera), stylography, LLM generation, templated wrappers, on-the-money book quotes, and Indy_READs' own commentary.**
- The return feed gives the operator **what they need, when they need it** — hyperplexed, sequenced, beautiful.

## 5. PERCYPHON.AI — fully fleshed out
- **Generative maths for the 5,000-soul village.** The active 5,000 are a **derived function of signal**: current cases + recent emails + operator telemetry + operator inputs + the graph → "what's important NOW."
- **DB shape: 5,000 rows × 128 columns** (one column per xxhash slot = coordinates). The "word" in slots 1–5,000 can change at any moment; each carries a **generated confidence of likely-relevance** from {hash, telemetry, generative} — fractal/Minecraft-style procedural maths.
- **Slots 1–28:** UUID, names, personas, backstories, aliases, emails, phones, ternary state — robust + creative.
- **Slots 29–128 (100 slots):** `seed = word-in-slot + hash` → `seed + procedural gen` → related concept expansion (cabbage → leafy green / brassica / …) = an **always-available verbosity engine**.
- Percyphon also owns: **identity, VPN/VPS/5G proxying, burn orders** — the inbound/outbound **comms filter to the outer world**. Zero-VRAM, deterministic, writes no canonical graph truth.

## 6. CAPABILITY SUITE — all online, all real
- **ETL any artifact; keep graph populated; advanced asymmetric analysis; chaining; network-asymmetric games.**
- **Indy_READs is herself.** **SANTA + KRAMPUS = fully realized ternary devices + network protocols.**
- Recreate **every prior case + research**. **Hypertimeline/chrono** filled with ALL known info.
- **Claims require evidence — else it's a hypothesis.**
- Revive online: **ORNAMENT** (heuristic image program), **DRE**, **SNOWBALL**, **MISTLETOE**, **SANTA**; scrape; pivot search.
- **Deterministic document creation** (find + wire the suite of **22+ doc-gen templates**) is a no-brainer.

## 7. OUTBOUND ACTION
- Make **websites**; send **parallel emails fast with any operator-required alias**; **scrape data anywhere**.
- **Code while operating. Research. Act.**

## 8. META — the autonomic mandate
- The system must **GROW · SELF-HEAL · SEEK knowledge · SEEK improvement · SEEK truth · SEEK justice.**

---
*Build sequence + per-workstream grounded plan: see the decomposition workflow output and the master roadmap appended by the Orchestrator.*
