#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "00_PROJECT_BRAIN" / "instruction_authority_registry.json"
OUT = ROOT / "05_OUTPUTS" / "instruction_hygiene"
BAD_PATTERNS = [
    ("receipt_collage_pass", re.compile(r"receipt\s+collage.*\b(pass|sufficient|proof)\b|\bcollage\b.*\bpass\b", re.I)),
    ("worker_self_report_sufficient", re.compile(r"worker\s+self[- ]?report\s+(is\s+)?sufficient|self[- ]?report\s+(is\s+)?sufficient\s+for\s+completion", re.I)),
    ("accepted_truth_without_proof", re.compile(r"accepted\s+truth\s+without\s+(proof|custody|oracle|audit)|accepted\s+truth.*no\s+(proof|custody|oracle|audit)", re.I)),
    ("markdown_only_completion", re.compile(r"markdown[- ]only\s+completion|markdown\s+(report\s+)?(is|counts\s+as)\s+completion|completion\s+without\s+evidence", re.I)),
    ("graph_materialization_bypass", re.compile(r"materiali[sz]e.*(without|bypass|skip).*(graph\s+write\s+barrier|command\s+envelope|evidence|operator\s+confirm)", re.I)),
    ("legacy_prompt_claims_current_authority", re.compile(r"legacy\s+prompt.*(current|canonical)\s+authority|old\s+prompt.*current\s+law", re.I)),
]
NEGATING_LAW_RE = re.compile(r"\b(no|not|never|do not|don't|reject|prevent|prohibit|forbid|forbidden|impossible|cannot)\b", re.I)


def now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def stamp() -> str: return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def rel(path: Path | str, root: Path = ROOT) -> str:
    try: return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        try: return str(Path(path).resolve().relative_to(ROOT.resolve()))
        except Exception: return str(path)

def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def read_text(path: Path) -> str:
    try: return path.read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""
def bad_pattern_hits(text: str) -> list[str]:
    hits: list[str] = []
    for kind, rx in BAD_PATTERNS:
        for line in text.splitlines():
            if rx.search(line):
                # "No markdown-only completion" and "no accepted truth without proof"
                # are active prohibitions, not permissions.
                if NEGATING_LAW_RE.search(line):
                    continue
                hits.append(kind)
                break
    return hits

def registry_paths(data: dict[str, Any], key: str) -> set[str]:
    out=set()
    for item in data.get(key) or []:
        if isinstance(item, str): out.add(item)
        elif isinstance(item, dict) and item.get("path"): out.add(str(item["path"]))
    return out

def check(data: dict[str, Any], *, root: Path) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    canonical = registry_paths(data, "canonical_files")
    archived = registry_paths(data, "superseded_files") | registry_paths(data, "legacy_reference_files")
    quarantined = registry_paths(data, "quarantined_files")
    manual = registry_paths(data, "manual_review_files")
    if canonical & archived:
        for p in sorted(canonical & archived): blockers.append({"kind":"archived_file_marked_active", "path":p})
    if canonical & quarantined:
        for p in sorted(canonical & quarantined): blockers.append({"kind":"quarantined_file_marked_active", "path":p})
    if canonical & manual:
        for p in sorted(canonical & manual): blockers.append({"kind":"manual_review_marked_active", "path":p})
    domains=[]
    for law in data.get("active_laws") or []:
        if isinstance(law, dict): domains.append(str(law.get("domain") or law.get("law_key") or "unknown"))
    for domain,count in Counter(domains).items():
        if domain != "unknown" and count > 1:
            blockers.append({"kind":"duplicate_active_canonical_authority", "domain":domain, "count":count})
    for p in sorted(canonical):
        path = root / p
        if not path.exists():
            blockers.append({"kind":"active_canonical_file_missing", "path":p}); continue
        text = read_text(path)
        low = p.lower()
        if "legacy" in low or "prompt" in low and "legacy_prompts" in low:
            blockers.append({"kind":"legacy_prompt_marked_active", "path":p})
        for kind in bad_pattern_hits(text):
            blockers.append({"kind":kind, "path":p})
        if re.search(r"accepted\s+truth", text, re.I) and not re.search(r"proof|custody|oracle|audit", text, re.I):
            blockers.append({"kind":"proofless_accepted_truth_instruction", "path":p})
    for p in sorted(archived | quarantined):
        if p in canonical: continue
        text = read_text(root / p)
        if re.search(r"current\s+law|canonical\s+authority|must\s+override", text, re.I):
            warnings.append({"kind":"archived_file_contains_authority_language", "path":p})
    return {
        "schema":"lucidota.instruction_conflict_scan.v1",
        "generated_at":now(),
        "registry": rel(REGISTRY),
        "active_canonical_files": sorted(canonical),
        "blockers": blockers,
        "warnings": warnings,
        "canonical_graph_materialization": False,
        "canonical_graph_writes": False,
        "verdict":"PASS" if not blockers else "FAIL",
    }

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--scan-root", default=str(ROOT))
    ap.add_argument("--output")
    a=ap.parse_args()
    reg_path=Path(a.registry)
    data=load_registry(reg_path)
    payload=check(data, root=Path(a.scan_root).resolve())
    payload["registry"] = rel(reg_path)
    OUT.mkdir(parents=True, exist_ok=True)
    out=Path(a.output) if a.output else OUT / f"instruction_conflict_scan_{stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload["report_path"]=rel(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    print("REPORT_PATH="+rel(out)); print("INSTRUCTION_CONFLICT_SCAN="+payload["verdict"])
    return 0 if payload["verdict"]=="PASS" else 4
if __name__ == "__main__": raise SystemExit(main())
