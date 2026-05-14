# Body Capture Visual Channel Design

Updated: 2026-05-13

## Canon

Adapters first. Browser capture is fallback, not default extraction.

## Visual Law

All visual diffs are evidence. Not all visual diffs are alerts.

## Capture Channels

1. HTTP body capture: server response bytes.
2. Rendered DOM snapshot: post-JavaScript browser DOM.
3. Screenshot image: visual evidence.
4. Optional trace/network log: evidence context only, refs/hashes in DB.

## Stabilization Rules Before Screenshot

- wait for network idle where available
- disable animations and transitions
- hide scrollbars
- force deterministic viewport
- avoid cursor/focus artifacts
- time-box browser execution
- always close browser context/process

## Hashing Rules

- `sha256`: raw artifact bytes in CAS
- `content_hash`: canonical visible text hash
- `structure_hash`: volatile-attribute-free skeleton hash
- `visual_hash`: perceptual/structural visual hash slot

## Alert Policy

Watcher profiles evaluate content, structure, and visual changes independently.
Visual-only changes default to `record_only` unless the profile is layout-sensitive.
