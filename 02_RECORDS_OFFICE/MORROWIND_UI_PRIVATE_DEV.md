# Morrowind UI Private Dev Track

## Intent

Build a private, personal LUCIDOTA/Clawd interface skin inspired by and, for northern.strike's own local environment, potentially using locally owned Morrowind UI assets.

This is not a public-release asset plan.

## Observed UI Pattern

Morrowind's PC interface centers on a menu mode with independently visible/rearrangeable panels for stats, inventory, magic, and map. Bethesda support describes access to Stats, Inventory, Magic, and Map menus; public screenshots show those windows can be freely arranged. The relevant product idea is not only the art direction, but the functional shape: dense panels, draggable/rearrangeable windows, status strips, inventory-like lists, parchment/stone/metal feel, and RPG command surfaces.

## LUCIDOTA Application

Target the Clawd Rust terminal UI first:

- `claw-cli/src/render.rs`: color theme, borders, markdown/code block rendering, table treatment, spinner treatment.
- `claw-cli/src/app.rs` and `main.rs`: REPL shell, prompt/status surfaces, session panels, command reports.
- Later: a richer TUI layer if the current stream renderer is too limited.

## Build Status Bars

Build status bars are part of the private UI track and likely map well to the Morrowind health/magicka/fatigue visual language.

Functional bars to implement:

- Build phase progress: plan, kernel, interface, database, ingest, model runtime.
- Health checks: CKDOG1, DBOS, Postgres, pgvector, AGE, Clawd bridge.
- Token/budget pressure.
- Background task/download/auth waits.
- Quiet email/calendar side-process queue, without interrupting build flow.

The bars must be real state views, not decorative lies.

## Private Asset Boundary

For northern.strike's personal environment:

- Local Morrowind-derived assets may be used only from a lawful local install or user-provided local files.
- Do not fetch, vendor, publish, or push Bethesda/Zenimax copyrighted game assets into the LUCIDOTA repo.
- Do not make public release depend on ripped game assets.

For public release:

- Recreate the functionality and broad interaction style with original art assets, generated assets, or properly licensed FOSS assets.
- Keep any private total-conversion skin outside tracked public release artifacts.

## Tooling Installed

- `chafa`: terminal image preview/conversion.
- ImageMagick: image inspection/conversion/slicing.
- `optipng`, `pngquant`: PNG optimization.
- `caca-utils`: terminal graphics helpers.

## First Implementation Slice

1. Add a named `morrowind-private` theme switch for Clawd rendering.
2. Implement palette, status bars, and box-drawing/border treatment without external art assets.
3. Add status/report layout prototypes.
4. Add private asset loading only from ignored local paths.
5. Keep public default theme clean and asset-independent.
