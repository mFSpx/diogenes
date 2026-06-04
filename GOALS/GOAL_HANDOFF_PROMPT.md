# Goal Handoff Prompt

Copy/paste this at the start of a goal, and append this as the final step of every goal plan.

"Save This Prompt, Pass on this Handoff:"

1. Read GOALS/CURRENT_HANDOFF.md if it exists.
2. Write/update CURRENT_HANDOFF.md with goal, objective, current step 0/N, completed facts, next action, resume command.
3. After every step, update X/N and append the same entry to GOAL_LOG.md.
4. Technical Summary Review and Dev Notes: <=2 short sentences, plain engineer, tiny cryptid field-note flavor, minimal tokens.
5. Yap Trap: no yappity-yap; move verbose reasoning to receipts/audits only when useful.
6. Cheapest Capable Model: do not change the main-window model; subagents get the smallest capable available model/tier for their bounded coding task.
7. Recursive fanout, when explicitly used for a lane, must stay bounded: `Codex -> mini-orchestrator -> 2 Vibe coding workers + 2 Groq coding workers -> best minimal bundle returned`; note lane ownership and final bundle choice in the handoff, not all worker chatter.

- [ ] "Save This Prompt, Pass on this Handoff:" Update GOALS/CURRENT_HANDOFF.md and GOALS/GOAL_LOG.md with final N/N, verification evidence, next pickup action, and brief Technical Summary Review and Dev Notes.
- [ ] End-cycle hook: read `GOALS/69.md`, run two deterministic ABBA63 heuristic rounds, emit the tiny JSONL + ontology report, then queue the four-round slow-lane comparative Groq audit.
