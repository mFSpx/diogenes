#!/usr/bin/env python3
"""indy_reads_swarm_router.py — persona-stamped Groq dispatch wrapper."""

import json, os, sys, urllib.request, urllib.error, datetime, pathlib

PERSONA_STAMP = {
    "schema": "indy_reads.worker_order.v1",
    "persona": "INDY_READS/LUCIDOTASOUL/DIOGENES/Northern.Strike co-pilot",
    "token_floor": 256,
    "token_ceiling": 768,
}

def route(intent: str, input_refs: list, model: str = None, target: str = "groq") -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    packet = {**PERSONA_STAMP, "intent": intent, "input_refs": input_refs}
    payload = json.dumps({
        "model": model,
        "max_tokens": PERSONA_STAMP["token_ceiling"],
        "messages": [{"role": "user", "content": json.dumps(packet)}],
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            status = "ok"
    except urllib.error.HTTPError as e:
        result = {"error": e.read().decode()}
        status = f"error-{e.code}"
    except Exception as e:
        result = {"error": str(e)}
        status = "error"

    utc = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path("05_OUTPUTS/indy_reads/swarm")
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = out_dir / f"{utc}_{intent[:20]}.json"
    receipt = {"status": status, "intent": intent, "target": target, "model": model,
               "input_refs": input_refs, "sampled_at": utc, "result_path": str(receipt_path)}
    receipt_path.write_text(json.dumps({"receipt": receipt, "response": result}, indent=2))
    return receipt


def read_governor() -> dict:
    try:
        return json.loads(pathlib.Path("/tmp/lucidota_governor_state.json").read_text())
    except Exception:
        return {}


if __name__ == "__main__":
    args = sys.argv[1:]
    intent, refs, i = None, [], 0
    while i < len(args):
        if args[i] == "--intent" and i + 1 < len(args):
            intent = args[i + 1]; i += 2
        elif args[i] == "--refs":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                refs.append(args[i]); i += 1
        else:
            i += 1
    if not intent or not refs:
        sys.exit("Usage: indy_reads_swarm_router.py --intent STR --refs PATH [PATH...]")
    print(json.dumps(route(intent, refs), indent=2))
