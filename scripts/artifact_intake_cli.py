from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from artifact_intake_worker import process_artifact  # noqa: E402


def _build_envelope_only(path: Path, location: str, groq_api_key: str) -> dict[str, Any]:
    import artifact_envelope
    import artifact_type_router
    import artifact_domain_classifier

    env = artifact_envelope.make_envelope(path, location)
    media_type_class: str = artifact_type_router.classify_media_type(path)
    text_snippet = ""
    if media_type_class in ("text_plain", "code", "structured"):
        try:
            text_snippet = path.read_text(errors="ignore")[:500]
        except OSError:
            pass
    domain, sub_domain = artifact_domain_classifier.classify_domain(
        path, text_snippet, groq_api_key
    )
    return {
        "artifact_id": str(env.artifact_id),
        "sha256": env.sha256,
        "domain": domain,
        "sub_domain": sub_domain,
        "media_type_class": media_type_class,
        "t_thing": str(env.t_thing),
        "t_possession": str(env.t_possession),
        "t_ingested": str(env.t_ingested),
        "xattr_stamped": env.xattr_stamped,
        "source_path": str(path),
        "dry_run": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Artifact intake CLI")
    ap.add_argument("--path", required=True, help="File or directory to ingest")
    ap.add_argument("--location", default="Laptop", help="Acquisition location")
    ap.add_argument("--limit", type=int, default=100, help="Max files (directory mode)")
    ap.add_argument("--dry-run", action="store_true", help="Classify only, skip DB write")
    ap.add_argument("--json", action="store_true", dest="json_out", help="JSONL output per file")
    args = ap.parse_args()

    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    target = Path(args.path)

    if target.is_file():
        files = [target]
    else:
        files = [p for p in target.rglob("*") if p.is_file()][: args.limit]

    processed = 0
    errors = 0
    domains: dict[str, int] = defaultdict(int)
    media_types: dict[str, int] = defaultdict(int)

    for f in files:
        try:
            if args.dry_run:
                receipt = _build_envelope_only(f, args.location, groq_api_key)
            else:
                receipt = process_artifact(f, args.location, groq_api_key)
            processed += 1
            domains[receipt["domain"]] += 1
            media_types[receipt["media_type_class"]] += 1
            if args.json_out:
                print(json.dumps(receipt))
        except Exception as exc:
            errors += 1
            if args.json_out:
                print(json.dumps({"error": str(exc), "path": str(f)}))
            else:
                print(f"ERROR {f}: {exc}", file=sys.stderr)

    summary = {
        "processed": processed,
        "domains": dict(domains),
        "media_types": dict(media_types),
        "errors": errors,
    }
    if args.json_out:
        print(json.dumps({"summary": summary}))
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

