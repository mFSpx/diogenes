# LUCI Convergence Mode Prompt

Saved on 2026-06-01 for reuse by Codex/agents working on LUCIDOTA.

ENTER LUCI CONVERGENCE MODE.

The goal is no longer “add more components.” The goal is to make LUCIDOTA become one coherent runnable product: LUCI.

LUCI is the single operator surface. Claw/clawd remnants are implementation history only. The user should run `luci`, type a command, and see a real routed operation happen.

Hard objective:
Turn the scattered system into one database-operated neural workstation where Postgres, ontology, queues, deterministic algorithms, routers, model lanes, telemetry, ingestion, receipts, and async workflows operate together as one machine.

Do not produce architecture prose unless it directly controls implementation.

Rules:
1. Read the durable project brain, GOALS handoff, active LUCI product vision, model registry, ontology files, current handoff, and recent receipts.
2. Identify the current runnable entrypoints.
3. Make `luci` the canonical user-facing entrypoint.
4. Make one client-like smoke path work end to end:
   user input -> ontology/routing packet -> fast/slow lane decision -> model/algorithm/workflow call -> Postgres/receipt write -> visible response.
5. Preserve PromptFlow as sidecar only. It must not block live LUCI.
6. Verify model fabric truthfully: Needle swarm, DeepSeek lane, Mamba/Bonsai lanes, BGE lane when safe, Groq/Vibes external lanes, LocateAnything as optional local machine-vision grounding.
7. Verify ingestion truthfully: readable material only to embeddings; MIME/base64/binary sludge quarantined; receipts written.
8. Verify Indy_READs as an active research/assistant surface, not a dormant script.
9. All important work must be async-capable and receipt-backed.
10. No “implemented” claims without commands, test output, receipt path, and remaining gap list.

Execution loop:
RUN -> BREAK -> INSPECT -> PATCH -> RE-RUN -> RECEIPT -> HANDOFF.

Definition of done for this pass:
- `./luci` or equivalent works.
- One real operator command completes through routing.
- One ingestion command completes or queues correctly.
- One model/algorithm lane is invoked or truthfully skipped with reason.
- One receipt is written.
- Tests for the touched surfaces pass.
- GOALS/CURRENT_HANDOFF.md and GOALS/GOAL_LOG.md are updated with exact evidence.

Do not expand the empire until the front door works.
Make LUCI one thing now.

## Active Decision Delta

Read `00_PROJECT_BRAIN/ACTIVE_SPEC/LUCI_ROUTING_FABRIC_DECISIONS_20260601.md` before changing routing/model admission. It controls the strict-local/adaptive-external model fabric, router-of-routers behavior, DB/graph control plane, and ABSURD workflow convergence target.
