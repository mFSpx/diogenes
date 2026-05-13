# LUCIDOTA Project Brain

This directory is the living memory layer for the LUCIDOTA/DIOGENES build.

Core files:

- `THEPLAN.md`: current full build plan.
- `TODO.md`: active task ledger.
- `DECISIONS.md`: decisions, sources, and status.
- `GLOSSARY.md`: project vocabulary and canonical meanings.
- `SOURCES.md`: repos, docs, tools, models, and reference links.
- `OPEN_QUESTIONS.md`: unknowns that matter, without blocking immediate work.
- `STATUS.md`: current build state and verification notes.
- `VIBESCONTROL.md`: repo-resident assistant cognition/wiki/audit discipline. It is project source, not build code yet.
- `GOALS.md`: long-term goal and operating meaning.

Operating rule:

Keep these files short enough to use, but complete enough that INDY_READS can recover project context from disk.

The project brain lives in the repo deliberately, but it must not interfere with builds. Future product integration should consume it through explicit adapters, not hidden side effects.
