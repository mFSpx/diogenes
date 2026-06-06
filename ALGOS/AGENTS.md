# ALGOS AGENT STARTUP LAW

1. Read root `CLAUDE.md` and `AGENTS.md` first.
2. **Do not write truth.** Algorithms gate and route; they do not persist.
3. **Do not make network calls.** Pure local execution.
4. **Do not import from scripts/.** Algorithms are standalone.
5. **Seed all randomness.** Every random function must have a deterministic seed parameter.
6. **Prefer pure functions.** No side effects. Input → computation → output.
7. When evolving new algorithms, write to `evolved/` with generation metadata.
