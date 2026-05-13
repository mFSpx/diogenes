# Anti-Slop Audit Doctrine

Purpose: keep DIOGENES from becoming vague, fragile, over-invented, or fake-working.

## Mandatory Checks

- Is this grounded in a source, test, inspected code path, or working command?
- Did we reuse mature FOSS before writing custom code?
- Is this solving the current loop, or just sounding architectural?
- Are assumptions labeled as assumptions?
- Are originals immutable and derived records traceable?
- Are external writes gated?
- Is encryption/security debt explicit?
- Can the output be cited, reproduced, or audited?
- Did we avoid hard-baking a temporary dev shortcut into release behavior?

## Failure Labels

- `VIBE`: sounds good, not verified.
- `SLOP`: vague or bloated artifact.
- `FANTASY`: invented capability not actually wired.
- `REINVENTION`: custom code where mature FOSS likely exists.
- `UNTRACED`: no source/test/hash/citation.
- `DANGEROUS`: hidden external effect, secret leak, or irreversible action.

## Correction Rule

When a failure label applies, either fix it immediately or record the smallest concrete next action that would make it real.

## Current Audit Result

- `FANTASY` avoided: Clawd-to-kernel bridge is not claimed as production yet; it is a verified baby-smoke command using generated Rust tonic/prost client code against the CKDOG1 Python gRPC server.
- `UNTRACED` avoided: `/home/mfspx/LUCIDOTA/check_diogenes.sh` verifies Python tests, CKDOG1 gRPC smoke, Rust workspace tests, release build, and Clawd smoke invocation.
- `DANGEROUS` contained: current smoke uses local temp homes and performs no external writes.
- `REINVENTION` avoided for bridge client: generated tonic/prost bindings are now used instead of a hand-rolled protocol.
