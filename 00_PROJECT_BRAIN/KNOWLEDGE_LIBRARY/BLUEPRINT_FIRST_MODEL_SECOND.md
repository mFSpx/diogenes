# Knowledge Card: Blueprint First, Model Second

- ID: `blueprint_first_model_second`
- Authority class: `pseudolaw`
- Source: https://arxiv.org/abs/2508.02721
- Local law artifact: `00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md`
- Companion reference: `01_REPOS/PocketFlow/pocketflow/__init__.py`

## Learned pattern

The paper's useful lesson for LUCIDOTA is architectural: put the workflow path in code/source blueprints and call models only as bounded tools at explicit nodes. This is now reflected in the pseudolaw file and enforced practically by `scripts/slop_audit_law.py` / `./claw slop-audit`.

## Use in future work

- Design queue workers and agent runtimes as readable state machines or scripts.
- Keep LLM calls behind schemas, typed packets, and receipts.
- Prefer small PocketFlow-style composable primitives over large hidden controllers.
