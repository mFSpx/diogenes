# OpenMW UI Source Map

## Purpose

Use OpenMW to separate three concerns for the private Morrowind-style Clawd/LUCIDOTA UI track:

- mechanics: panel behavior, layout structure, HUD/status bars, scroll/book/window patterns.
- FOSS reference assets: OpenMW-owned resources that may be usable with license compliance.
- private local assets: Bethesda/Morrowind assets from a lawful local install, ignored and never pushed.

This file is an implementation map, not a license grant. Before copying any asset into a distributable build, verify the specific file license and preserve required notices.

## Installed Reference Surface

Local packages installed:

- `openmw 0.48.0-1ubuntu5`
- `openmw-cs 0.48.0-1ubuntu5`
- `openmw-data 0.48.0-1ubuntu5`
- `openmw-launcher 0.48.0-1ubuntu5`

Main resource root:

- `/usr/share/games/openmw/resources`

OpenMW's useful lesson for LUCIDOTA is architectural: engine/interface code can be free and owned, while original game content remains user-supplied content.

## Mechanics To Study

The mechanics source is the MyGUI layout/skin layer under:

- `/usr/share/games/openmw/resources/vfs/mygui`

Primary files for the Clawd UI conversion:

- `openmw_hud.layout`: HUD composition and bar placement.
- `openmw_hud_energybar.skin.xml`: health/magicka/fatigue style bar behavior.
- `openmw_progress.skin.xml`: progress/status bar skins.
- `openmw_hud_box.skin.xml`: HUD frame treatment.
- `openmw_windows.skin.xml`: reusable window frame skins.
- `openmw_box.skin.xml`: box/frame primitives.
- `openmw_button.skin.xml`: button state treatment.
- `openmw_text.skin.xml`: text widgets.
- `openmw_scroll.layout` and `openmw_scroll.skin.xml`: parchment/scroll document surface.
- `openmw_inventory_window.layout`: inventory panel density and item surface.
- `openmw_stats_window.layout`: stats panel layout.
- `openmw_magicselection_dialog.layout`: magic/action selection panel.
- `openmw_map_window.layout`: map panel framing.
- `openmw_journal.layout` and `openmw_journal.skin.xml`: journal/document behavior.
- `openmw_dialogue_window.layout` and `openmw_dialogue_window.skin.xml`: dialogue and choice surface.
- `openmw_mainmenu.layout` and `openmw_mainmenu.skin.xml`: main menu composition.
- `skins.xml`, `core.skin`, `core.xml`: skin registry and base widget definitions.

Immediate implementation translation:

- terminal panels become box-drawing windows with stable dimensions.
- status bars map to real build/service/token/background state, not decoration.
- inventory panels map to tools, models, flows, algos, LoRAs, and source cartridges.
- journal/scroll panels map to plans, investigation notes, and provenance records.
- dialogue panels map to command suggestions, confirmations, and user-prompted choices.

## Asset Boundary

Allowed in private local environment:

- references to a user-owned Morrowind install.
- locally extracted textures/fonts/audio under ignored `private_assets/`.
- one-off private theme development that never enters a public artifact.

Allowed in repo after review:

- original code implementing Morrowind-like mechanics.
- layout logic inspired by RPG panel workflows.
- OpenMW-derived code or data only when license-compatible and notices are preserved.
- original or generated replacement art for eventual public release.

Not allowed in repo/public release:

- Bethesda/Zenimax game art.
- Bethesda/Zenimax fonts.
- Bethesda/Zenimax audio.
- packed game data from Morrowind, Tribunal, or Bloodmoon.
- screenshots or extracted assets used as shipped product material.

## FOSS Candidate Assets

Candidate OpenMW files to inspect before use:

- `/usr/share/games/openmw/resources/vfs/textures/omw_menu_scroll_*.dds`
- `/usr/share/games/openmw/resources/vfs/fonts/DejaVuLGCSansMono.ttf`
- `/usr/share/games/openmw/resources/vfs/fonts/DemonicLetters.ttf`
- `/usr/share/games/openmw/resources/vfs/fonts/MysticCards.ttf`
- `/usr/share/games/openmw/resources/vfs/fonts/*License.txt`
- `/usr/share/games/openmw/resources/openmw.png`

Do not assume the whole package has one asset policy. Treat each copied file as requiring attribution and license review.

## Public Release Strategy

Private dev can move fast with the ignored local asset lane. Public release later gets a sanitization pass:

1. keep mechanics and layout code.
2. replace private assets with original/generated/FOSS assets.
3. preserve OpenMW/GPL notices if any OpenMW material survives.
4. document exactly what ships and why it is redistributable.

## Current Decision

OpenMW is the clean mechanics reference. Morrowind is the private aesthetic target. LUCIDOTA owns the interface code.
