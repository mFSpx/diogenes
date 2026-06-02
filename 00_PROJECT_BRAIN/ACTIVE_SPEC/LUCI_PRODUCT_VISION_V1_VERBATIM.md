# LUCI Product Vision V1 — Verbatim Operator-Approved Summary

Saved verbatim on 2026-06-01.

1. **Make Claw actually run like a client product**
   - You type into it.
   - It routes your call.
   - It uses fast/slow lanes.
   - It writes receipts/state to Postgres.
   - It does not look like leftover “clawd” bullshit at the user surface.
   - Ideally `luci` becomes the clean operator entrypoint.

2. **Bring the model fabric online truthfully**
   - 6x Needle routers.
   - DeepSeek R1 Distill Qwen lane.
   - Mamba RAM lane.
   - Bonsai RAM lane.
   - BGE embedding lane in VRAM when safe.
   - Groq/Vibes as external lanes, not confused with local resident models.
   - LocateAnything-3B as a **machine vision / visual grounding** candidate for local image understanding — object boxes, GUI grounding, document layout, OCR localization. Not image generation.

3. **Keep PromptFlow out of the hot path**
   - PromptFlow is for visual prototyping / operator experiments.
   - It should never hard-gate live requests.
   - Live LUCI should stay fast, async, DB-first.

4. **Fix ingestion for real**
   - Archives actually unpacked or streamed correctly.
   - Emails/PDFs/docs/images routed through the right extractors.
   - Garbage quarantined before embedding.
   - Readable material embedded.
   - Audit reports produced.
   - No poisoning the vector space with MIME/base64/binary sludge.

5. **Make Indy_READs actually feel alive**
   - Watches books/material.
   - Produces research/LoRA/intake artifacts.
   - Acts like your assistant/research desk, not just a dormant script.

6. **Wire the “math owns the LLMs” architecture**
   - LLMs are sidepieces / bounded processors.
   - DB, ontology, algorithms, routers, queues, telemetry, and deterministic work trees control the system.
   - Everything important is async and receipt-backed.

7. **Operate adversarially**
   - Run it.
   - Break it.
   - Inspect logs.
   - Fix the real blocker.
   - Re-run.
   - Don’t merely say “implemented.”

So the short version:

> You want me to turn Claw into LUCI: a live, local, sovereign intelligence/coding/investigation system where ingestion, routing, local models, external APIs, telemetry, ontology, and database-backed async execution all actually operate together — with receipts — instead of existing as scattered promising parts.
