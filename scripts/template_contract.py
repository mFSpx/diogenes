#!/usr/bin/env python3
"""Small deterministic Jinja-like template contract for LUCIDOTA outputs.

This is intentionally tiny: variable interpolation plus simple for-loops. It is
not an expression engine and performs no network/model calls.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import Any

from spine_common import receipt, rel, sha256_json

VAR_RE = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*}}")
LOOP_RE = re.compile(r"{%\s*for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+([A-Za-z_][A-Za-z0-9_.]*)\s*%}(.*?){%\s*endfor\s*%}", re.S)


def resolve(context: dict[str, Any], dotted: str, local: dict[str, Any] | None = None) -> Any:
    if local and dotted in local:
        return local[dotted]
    cur: Any = context
    for part in dotted.split("."):
        if local and part in local:
            cur = local[part]
            continue
        if isinstance(cur, dict):
            cur = cur.get(part, "")
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return ""
    return cur


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def render_template(template: str, context: dict[str, Any], local: dict[str, Any] | None = None) -> str:
    def loop(match: re.Match[str]) -> str:
        name, path, body = match.group(1), match.group(2), match.group(3)
        seq = resolve(context, path, local)
        if not isinstance(seq, list):
            return ""
        return "".join(render_template(body, context, {**(local or {}), name: item}) for item in seq)

    rendered = LOOP_RE.sub(loop, template)
    return VAR_RE.sub(lambda m: stringify(resolve(context, m.group(1), local)), rendered)


def load_context(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.context_json) if args.context_json else json.loads(Path(args.context_file).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("context must be a JSON object")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Render a deterministic Jinja-like template and write a receipt.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--template-string")
    src.add_argument("--template-file")
    ctx = ap.add_mutually_exclusive_group(required=True)
    ctx.add_argument("--context-json")
    ctx.add_argument("--context-file")
    ap.add_argument("--output", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    template = args.template_string if args.template_string is not None else Path(args.template_file).read_text(encoding="utf-8")
    context = load_context(args)
    rendered = render_template(template, context)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(rendered, encoding="utf-8")
    payload = {
        "schema": "lucidota.template_render_receipt.v1",
        "status": "PASSED",
        "template_contract": "jinja_like_minimal_v1",
        "output_path": rel(out),
        "context_sha256": sha256_json(context),
        "output_sha256": sha256_json({"rendered": rendered}),
        "model_calls_performed": False,
    }
    receipt("template_render", payload, root="05_OUTPUTS/templates")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    print("TEMPLATE_RENDER=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
