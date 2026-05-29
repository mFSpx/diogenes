#!/usr/bin/env python3
"""Bounded local-model overseer: health, RAM/VRAM guards, DB snapshot, JSON tasks."""
from __future__ import annotations
import argparse, json, os, re, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "model_runtime"
STACK = {
    "deepseek_r1_qwen_1p5b_gpu": (8080, "coder", "DeepSeek Qwen 1.5B GPU"),
    "mamba7b_ram": (8081, "db_watch", "Falcon3 Mamba 7B RAM"),
    "bonsai4b_ram": (8082, "review", "Ternary Bonsai 4B RAM"),
    "mamba7b_gpu_partial": (8083, "planner", "Falcon3 Mamba 7B partial VRAM"),
    "needle_0": (8090, "router", "Needle"),
    "needle_1": (8091, "router", "Needle"),
    "needle_2": (8092, "router", "Needle"),
    "needle_3": (8093, "router", "Needle"),
    "needle_4": (8094, "router", "Needle"),
    "needle_5": (8095, "router", "Needle"),
}

def sh(cmd: list[str], timeout: int = 5) -> str:
    return subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False).stdout.strip()

def stamp() -> str: return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

def http_json(port: int, path: str, payload: dict | None = None, timeout: int = 3) -> dict:
    try:
        data = json.dumps(payload).encode() if payload else None
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read(4096).decode("utf-8", "ignore")
        return {"ok": True, "raw": raw, "json": json.loads(raw) if raw[:1] in "[{" else None}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}:{str(exc)[:180]}"}

def mem() -> dict:
    rows = sh(["bash", "-lc", "free -m | awk 'NR==2{print $2,$3,$4,$7} NR==3{print $2,$3,$4}'"]).splitlines()
    gpu = sh(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    m = [int(x) for x in rows[0].split()] if rows else [0, 0, 0, 0]
    s = [int(x) for x in rows[1].split()] if len(rows) > 1 else [0, 0, 0]
    g = [int(x.strip()) for x in gpu.split(",")[:4]] if gpu else [0, 0, 0, 0]
    return {"ram_total": m[0], "ram_used": m[1], "ram_free": m[2], "ram_available": m[3], "swap_total": s[0], "swap_used": s[1], "gpu_total": g[0], "gpu_used": g[1], "gpu_free": g[2], "gpu_util": g[3]}

def pid_lines() -> str: return sh(["bash", "-lc", "pgrep -af 'llama-server|lucidota_needle_worker|lucidota_indy_reads_watcher' | grep -v pgrep || true"])

def queue_snapshot() -> dict:
    sql = "select status::text,count(*) from lucidota_control.absurd_queue_job group by 1 order by 1"
    out = sh(["bash", "-lc", f"psql postgresql://mfspx@/lucidota_state -Atqc {json.dumps(sql)}"], timeout=4)
    rows = {}
    for line in out.splitlines():
        if "|" in line:
            k, v = line.split("|", 1); rows[k] = int(v)
    return {"absurd_queue_job_by_status": rows, "raw": out[-1000:]}
def ontology() -> dict:
    try:
        o = json.loads((ROOT / "OFFICIAL_ONTOLOGY.json").read_text())
        return {"official_ontology": o.get("official_ontology"), "core_sentence": o.get("core_sentence"), "active_terms": o.get("active_terms", [])[:8]}
    except Exception:
        return {"official_ontology": "missing", "core_sentence": "", "active_terms": []}
def parse_model_json(raw: str):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I | re.M).strip()
    for m in re.finditer(r"\{[^{}]*\}", raw, flags=re.S):
        try:
            obj = json.loads(m.group(0))
            if {"action","test","risk"} <= set(obj): return obj
        except Exception: pass
    return None

def enforce(r: dict) -> list[dict]:
    acts = []
    if r["gpu_free"] < int(os.environ.get("LUCIDOTA_VRAM_RESERVE_MB", "768")):
        pid = (ROOT / "04_RUNTIME/inference_os/mamba7b_gpu.pid").read_text().strip() if (ROOT / "04_RUNTIME/inference_os/mamba7b_gpu.pid").exists() else ""
        if pid:
            subprocess.run(["kill", pid], check=False); acts.append({"kill": pid, "why": "gpu_free_below_reserve", "target": "mamba7b_gpu_partial"})
    if r["ram_available"] < 1800 or r["swap_used"] > 6500:
        acts.append({"defer": "new_model_loads", "why": "ram_or_swap_guard"})
    return acts

def task(port: int, role: str, q: dict) -> dict:
    want = {"action": role, "test": "online_json", "risk": "bounded"}
    prompt = {"messages": [
        {"role": "system", "content": "Return only minified JSON. No markdown, no prose."},
        {"role": "user", "content": "CTX=GO-25;EVIDENCE supports CLAIM. Return exactly "+json.dumps(want,separators=(",", ":"))}],
        "temperature": 0.0, "max_tokens": 320 if role == "coder" else 64, "stream": False}
    res = http_json(port, "/v1/chat/completions", prompt, timeout=180)
    msg = (((res.get("json") or {}).get("choices") or [{}])[0].get("message") or {}) if res.get("ok") else {}
    raw = msg.get("content") or msg.get("reasoning_content") or ""
    parsed = parse_model_json(raw)
    return {"ok": res.get("ok"), "json_ok": parsed is not None, "parsed": parsed, "raw": raw[:1200], "error": res.get("error")}

def once(assign: bool) -> dict:
    health = {name: {"port": port, "role": role, "health": http_json(port, "/health", timeout=1)} for name, (port, role, _) in STACK.items()}
    r, q = mem(), queue_snapshot()
    receipt = {"schema": "lucidota.model_turbine_overseer.v1", "at": stamp(), "policy": {"max_file_read_mb": 1, "no_giant_files": True, "graph_writes": "blocked", "tight_json": True}, "resources": r, "health": health, "queue": q, "pids": pid_lines(), "actions": enforce(r)}
    if assign:
        receipt["assignments"] = {k: task(p, role, q) for k, (p, role, _) in STACK.items() if role in {"coder", "planner", "db_watch", "review"} and health[k]["health"]["ok"]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"model_turbine_overseer_{stamp()}.json"; path.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    (OUT / "model_turbine_overseer_latest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return {"receipt_path": str(path.relative_to(ROOT)), "health_ok": {k: v["health"]["ok"] for k, v in health.items()}, "actions": receipt["actions"], "resources": r}

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--assign", action="store_true"); ap.add_argument("--watch", type=int, default=0); args = ap.parse_args()
    while True:
        print(json.dumps(once(args.assign), sort_keys=True), flush=True)
        if args.watch <= 0: return 0
        time.sleep(args.watch)
if __name__ == "__main__": raise SystemExit(main())
