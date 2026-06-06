#!/usr/bin/env python3
"""ROOT-414 Gauntlet Game: interactive terminal UI.

Run:
  scripts/lucidota_414_game.py

Controls are prompt/menu based so it works in plain terminals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
GAME = ROOT / "scripts" / "lucidota_414_gauntlet_game.py"
BENCH = ROOT / "00_PROJECT_BRAIN" / "414_PRIMITIVE_CRIES" / "benchmarks"
CASES = BENCH / "cases"
SUBS = BENCH / "submissions"
JUDGMENTS = BENCH / "judgments"
SAVE = BENCH / "game_state.json"
GRADING_SCHEMA = ROOT / "BOOKS" / "ROOT414_GAME_GRADING_SCHEMA.json"


def clear() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def pause(msg: str = "Press ENTER to continue...") -> None:
    input(f"\n{msg}")


def run_json(args: list[str]) -> dict:
    cp = subprocess.run([str(GAME), "--json", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr.strip() or cp.stdout.strip())
    return json.loads(cp.stdout)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    SAVE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_state() -> dict:
    if SAVE.exists():
        return load_json(SAVE)
    return {"current_book_id": "", "current_book_path": "", "current_page": 1, "score_total": 0, "rounds": []}


def header(title: str) -> None:
    clear()
    print("╔" + "═" * 76 + "╗")
    print("║" + "ROOT-414 GAUNTLET".center(76) + "║")
    print("║" + title.center(76) + "║")
    print("╚" + "═" * 76 + "╝\n")


def choose(options: list[str], prompt: str = "Choose") -> int | None:
    for i, item in enumerate(options, start=1):
        print(f"  {i:>2}. {item}")
    print("   q. back/quit")
    ans = input(f"\n{prompt}: ").strip().lower()
    if ans in {"q", "quit", "back", "b"}:
        return None
    try:
        n = int(ans)
        if 1 <= n <= len(options):
            return n - 1
    except ValueError:
        print("Invalid numeric choice.")
    print("Invalid choice.")
    pause()
    return None


def book_menu(state: dict) -> None:
    while True:
        header("BOOK LIBRARY")
        lib = run_json(["library"])
        books = lib.get("books", [])
        if not books:
            print(f"No books found in {lib.get('book_dir')}")
            pause(); return
        labels = [f"{b['name']} [{b['ext']}, {b['size_bytes']} bytes]" for b in books]
        idx = choose(labels, "Pick a book")
        if idx is None:
            return
        b = books[idx]
        state["current_book_id"] = b["id"]
        state["current_book_path"] = b["path"]
        state["current_page"] = 1
        save_state(state)
        print(f"\nSelected: {b['name']}")
        pause()
        return


def extract_current_page(state: dict) -> dict | None:
    if not state.get("current_book_path"):
        print("No book selected.")
        pause(); return None
    page = int(state.get("current_page", 1))
    try:
        return run_json(["extract-page", state["current_book_path"], str(page)])
    except Exception as e:
        print(f"Extraction failed: {e}")
        pause()
        return None


def show_wrapped(text: str, width: int = 96, max_lines: int = 42) -> None:
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, replace_whitespace=True))
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"\n... [{len(lines) - max_lines} more lines hidden — use case file for full text]")


def current_round(state: dict) -> dict | None:
    result = extract_current_page(state)
    if not result:
        return None
    case = load_json(Path(result["case_path"]))
    return {"extract": result, "case": case}


def play_round(state: dict) -> None:
    header("ROUND START")
    if not state.get("current_book_path"):
        print("Pick a book first.")
        pause(); return
    round_obj = current_round(state)
    if not round_obj:
        return
    case = round_obj["case"]
    ext = round_obj["extract"]
    header(f"ROUND: PAGE {state.get('current_page', 1)}")
    print(f"Book: {Path(state['current_book_path']).name}")
    print(f"Case: {case['benchmark_id']}")
    print(f"Extract: {ext['extract_method']} | chars={ext['chars']} | page_hash={ext['page_hash'][:12]}…")
    print("\n" + "─" * 80)
    show_wrapped(case.get("input_text", ""))
    print("─" * 80)
    print("\nJUDGE FOCUS:")
    for item in case.get("judge_focus", []):
        print(f" - {item}")
    pause("ENTER for judgment screen...")
    judge_screen(state, case)


def judge_screen(state: dict, case: dict) -> None:
    header("NORTHERN.STRIKE JUDGMENT")
    print(f"Case: {case['benchmark_id']}\n")
    decisions = ["approved", "needs_repair", "rejected", "comment"]
    idx = choose(decisions, "Decision")
    if idx is None:
        return
    decision = decisions[idx]
    while True:
        raw = input("Score 0-100: ").strip()
        try:
            score = max(0, min(100, int(raw)))
            break
        except ValueError:
            print("Enter a number.")
    notes = input("Notes / piss judgment: ").strip()
    repair = ""
    if decision in {"needs_repair", "rejected"}:
        repair = input("Repair instruction: ").strip()

    submission_id = f"human_round_p{int(state.get('current_page',1)):04d}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    result = run_json(["judge", case["benchmark_id"], submission_id, decision, str(score), notes or "", "--repair", repair])
    state["score_total"] = int(state.get("score_total", 0)) + score
    state.setdefault("rounds", []).append({
        "case_id": case["benchmark_id"],
        "page": state.get("current_page", 1),
        "decision": decision,
        "score": score,
        "judgment_path": result["judgment_path"],
    })
    save_state(state)
    print(f"\nSaved: {result['judgment_path']}")
    if input("Advance to next page? [Y/n]: ").strip().lower() not in {"n", "no"}:
        state["current_page"] = int(state.get("current_page", 1)) + 1
        save_state(state)


def status(state: dict) -> None:
    header("STATUS")
    print(f"Book: {Path(state.get('current_book_path','')).name if state.get('current_book_path') else '(none)'}")
    print(f"Page: {state.get('current_page', 1)}")
    print(f"Total score: {state.get('score_total', 0)}")
    print(f"Rounds judged: {len(state.get('rounds', []))}")
    print("\nRecent rounds:")
    for r in state.get("rounds", [])[-10:]:
        print(f" - p{r['page']}: {r['decision']} {r['score']} :: {r['case_id']}")
    pause()


def help_screen() -> None:
    header("HELP")
    print("This is the ROOT-414 reading gauntlet game.\n")
    print(f"Grading schema: {GRADING_SCHEMA}\n")
    print("1. Pick book from /home/mfspx/LUCIDOTA/BOOKS")
    print("2. Start round")
    print("3. It extracts ONLY current page/chunk")
    print("4. You judge: approved / needs_repair / rejected / comment")
    print("5. Judgment saves into benchmarks/judgments")
    print("6. Next page unlocks")
    print("\nMOBI note: without Calibre, extraction uses strings fallback = ugly hard mode.")
    pause()


def main() -> int:
    state = load_state()
    while True:
        header("MAIN MENU")
        print(f"Current book: {Path(state.get('current_book_path','')).name if state.get('current_book_path') else '(none)'}")
        print(f"Grading schema: {GRADING_SCHEMA}")
        print(f"Current page: {state.get('current_page', 1)} | Total score: {state.get('score_total', 0)} | Rounds: {len(state.get('rounds', []))}\n")
        opts = [
            "Pick book",
            "Start current round",
            "Show status",
            "Help",
            "Quit",
        ]
        idx = choose(opts, "Move")
        if idx is None or opts[idx] == "Quit":
            header("EXIT")
            print("Gauntlet paused. State saved.")
            return 0
        if idx == 0:
            book_menu(state)
        elif idx == 1:
            play_round(state)
        elif idx == 2:
            status(state)
        elif idx == 3:
            help_screen()


if __name__ == "__main__":
    raise SystemExit(main())
