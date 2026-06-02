# Root-Rotor Fault Recovery Telemetry Flow / Turtle-Bandit Mapping

Captured: 2026-06-02
Authority: operator-provided architecture mapping candidate. Status: proposed local design pattern until validated against live schemas and receipts.

```text
[Fault State] -> [K restarted] -> [WHY?] -> [IS THIS WORTH IT?] -> [CHANGE IT UP?] -> [HOW?] -> [IS THAT CORRECT?]
(Process Dead) -> Rete Engine -> Snapping Turtle Flip Gate -> Treelite Prune -> Bandit Route -> Status Gate

[WHY?] = Rete Forward Chaining. It matches working-memory facts against production rules to isolate the breaking condition token.

[IS THIS WORTH IT?] = Snapping Turtle Flip Gate. It calculates resource cost and risk for hard recovery. It gates high-energy recovery before routing.

[CHANGE IT UP?] = Treelite Pruning. It prunes dead decision trees and modifies the runtime decision landscape after the flip decision.

[HOW?] = Bandit Routing. It selects the best operational arm/path after recovery is admitted.

[IS THAT CORRECT?] = Status Contract Validation. It verifies that the fix holds through a database/status gate.

Core correction: Multi-armed bandit chooses among competing paths. It does not decide whether recovery is worth the resource risk. The snapping turtle flip gate decides whether the high-energy recovery action is worth attempting.
```

## Canon handling

Do not install the proposed PL/pgSQL as real schema until live table names, existing Rete/Treelite/Bandit primitives, queue semantics, and status mutation authority are verified. Treat this as an external/operator reference seed and map it to manual nodes through the same sidecar and validator path.
