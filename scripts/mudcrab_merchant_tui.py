#!/usr/bin/env python3
"""Mudcrab Merchant: small local TUI launcher for the Clawd desktop.

No policy ceremony here: this is a convenience switchboard for the operator's
local environment. It launches Claw, Firefox, maps, and file explorers using the
best command available on this machine.
"""
from __future__ import annotations

import argparse
import curses
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


@dataclass(frozen=True)
class Launcher:
    key: str
    label: str
    description: str
    wait: bool
    command: list[str]


def which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def firefox_command(url: str | None = None) -> list[str]:
    if firefox := which("firefox"):
        return [firefox] + ([url] if url else [])
    if flatpak := which("flatpak"):
        return [flatpak, "run", "org.mozilla.firefox"] + ([url] if url else [])
    if snap := which("snap"):
        return [snap, "run", "firefox"] + ([url] if url else [])
    if opener := which("xdg-open"):
        return [opener, url or "about:blank"]
    return ["false"]


def maps_command() -> list[str]:
    for candidate in ("qgis", "marble", "gnome-maps"):
        if path := which(candidate):
            return [path]
    return firefox_command("https://www.openstreetmap.org/#map=12/49.2827/-123.1207")


def gui_files_command() -> list[str]:
    for candidate in ("nautilus", "nemo", "dolphin", "thunar", "pcmanfm"):
        if path := which(candidate):
            return [path, str(ROOT)]
    if opener := which("xdg-open"):
        return [opener, str(ROOT)]
    return ["ls", "-la", str(ROOT)]


def terminal_files_command() -> tuple[list[str], bool]:
    for candidate in ("yazi", "ranger", "lf", "mc"):
        if path := which(candidate):
            return [path, str(ROOT)], True
    return ["find", str(ROOT), "-maxdepth", "2", "-type", "f"], True


def launchers() -> list[Launcher]:
    terminal_files, terminal_wait = terminal_files_command()
    return [
        Launcher(
            "claw",
            "Claw / Clawd shell",
            "Local graph-first Claw with full local capability.",
            True,
            [str(ROOT / "claw")],
        ),
        Launcher("firefox", "Firefox", "Open Firefox on the Lucidota cockpit if available.", False, firefox_command(ROOT.as_uri())),
        Launcher("maps", "Maps", "Open native maps if installed, else OpenStreetMap in Firefox.", False, maps_command()),
        Launcher("files", "File explorer", "Open this repo in the desktop file explorer.", False, gui_files_command()),
        Launcher("terminal-files", "Terminal file explorer", "Open yazi/ranger/lf/mc if installed; fallback to find.", terminal_wait, terminal_files),
        Launcher(
            "big-board",
            "Lucidota Big Board",
            "Print the local status board JSON.",
            True,
            [str(PY), str(ROOT / "scripts" / "lucidota_big_board.py"), "--json"],
        ),
    ]


def run_launcher(item: Launcher, *, dry_run: bool = False) -> dict[str, object]:
    result = {"key": item.key, "label": item.label, "command": item.command, "wait": item.wait}
    if dry_run:
        return {"ok": True, "dry_run": True, **result}
    env = os.environ.copy()
    env.setdefault("LUCIDOTA_HOME", str(ROOT))
    env.setdefault("CLAW_LOCAL_GRAPH_RUNTIME", "1")
    env.setdefault("CLAW_PERMISSION_MODE", "danger-full-access")
    if item.wait:
        rc = subprocess.call(item.command, cwd=ROOT, env=env)
        return {"ok": rc == 0, "returncode": rc, **result}
    proc = subprocess.Popen(item.command, cwd=ROOT, env=env, start_new_session=True)
    return {"ok": True, "pid": proc.pid, **result}


def draw_menu(stdscr) -> int:
    items = launchers()
    selected = 0
    curses.curs_set(0)
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        title = "Mudcrab Merchant — Clawd Desktop Switchboard"
        stdscr.addstr(0, 0, title[: max(0, w - 1)], curses.A_BOLD)
        stdscr.addstr(1, 0, "↑/↓ or j/k move · Enter launches · q quits"[: max(0, w - 1)])
        for i, item in enumerate(items):
            attr = curses.A_REVERSE if i == selected else curses.A_NORMAL
            stdscr.addstr(3 + i * 2, 0, f"{i + 1}. {item.label}"[: max(0, w - 1)], attr)
            if 4 + i * 2 < h:
                stdscr.addstr(4 + i * 2, 3, item.description[: max(0, w - 4)])
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("q"), ord("Q"), 27):
            return 0
        if key in (curses.KEY_UP, ord("k"), ord("K")):
            selected = (selected - 1) % len(items)
        elif key in (curses.KEY_DOWN, ord("j"), ord("J")):
            selected = (selected + 1) % len(items)
        elif key in (curses.KEY_ENTER, 10, 13):
            curses.endwin()
            print(json.dumps(run_launcher(items[selected]), indent=2, sort_keys=True))
            if items[selected].wait:
                input("\nPress Enter to return to Mudcrab Merchant...")
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            stdscr.keypad(True)


def main() -> int:
    ap = argparse.ArgumentParser(prog="mudcrab")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--launch", choices=[item.key for item in launchers()])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    items = launchers()
    if args.list:
        print(json.dumps([item.__dict__ for item in items], indent=2, sort_keys=True))
        return 0
    if args.launch:
        item = next(item for item in items if item.key == args.launch)
        result = run_launcher(item, dry_run=args.dry_run)
        print(json.dumps(result, indent=2, sort_keys=True) if args.json else result)
        return 0 if result.get("ok") else 1
    if not sys.stdin.isatty():
        print(json.dumps([item.__dict__ for item in items], indent=2, sort_keys=True))
        return 0
    return curses.wrapper(draw_menu)


if __name__ == "__main__":
    raise SystemExit(main())
