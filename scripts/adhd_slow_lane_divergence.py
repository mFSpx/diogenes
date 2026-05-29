#!/usr/bin/env python3
"""ADHD slow lane: divergent frames, local lanes, Treelite pruning."""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
from treelite import gtil, model_builder as mb
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from local_model_chat_cli import probe_lane  # noqa: E402
OUT = ROOT / "05_OUTPUTS" / "adhd_slow_lane"
ROUTER_COUNT = 26
FRAMES = ["forensic", "speedrunner", "regulator", "red_team", "zero_budget", "systems"]
LANES = ["deepseek", "mamba_cpu", "bonsai"]
def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def router(i: int):
    meta = mb.Metadata(num_feature=5, task_type="kRegressor", average_tree_output=False, num_target=1, num_class=[1], leaf_vector_shape=(1, 1))
    ann = mb.TreeAnnotation(num_tree=1, target_id=[0], class_id=[0])
    b = mb.ModelBuilder(threshold_type="float32", leaf_output_type="float32", metadata=meta, tree_annotation=ann, postprocessor=mb.PostProcessorFunc(name="identity"), base_scores=[0.0])
    b.start_tree(); b.start_node(0)
    b.numerical_test(i % 5, 0.15 + 0.08 * (i % 4), default_left=True, opname="<", left_child_key=1, right_child_key=2)
    b.end_node(); b.start_node(1); b.leaf(0.0); b.end_node(); b.start_node(2); b.leaf(1.0 + i / 100); b.end_node(); b.end_tree()
    return b.commit()
ROUTERS = [router(i) for i in range(ROUTER_COUNT)]
def features(text: str) -> np.ndarray:
    low = text.lower()
    buckets = [
        min(len(text) / 180.0, 1.0),
        sum(t in low for t in ["evidence", "source", "receipt", "timeline", "verify"]) / 5,
        sum(t in low for t in ["first step", "run", "wire", "test", "ship"]) / 5,
        sum(t in low for t in ["risk", "blocker", "failure", "weak", "trap"]) / 5,
        sum(t in low for t in ["because", "therefore", "must", "exact", "json"]) / 5,
    ]
    return np.array([buckets], dtype=np.float32)
def score_branches(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for branch in branches:
        xs = features(branch["text"])
        scores = [float(gtil.predict(model, xs).reshape(-1)[0]) for model in ROUTERS]
        scored.append({**branch, "router_scores": scores, "treelite_score": sum(scores) / len(scores)})
    return sorted(scored, key=lambda b: b["treelite_score"], reverse=True)
def prune(scored: list[dict[str, Any]], survivors: int) -> list[dict[str, Any]]:
    return scored[: max(1, min(survivors, len(scored)))]
def telemetry(receipt: dict[str, Any]) -> dict[str, Any]:
    acct = receipt.get("token_accounting") or {}
    if acct.get("source") == "provider_usage":
        return {"metering": "provider_usage", "prompt_tokens": acct.get("prompt_tokens"), "completion_tokens": acct.get("completion_tokens"), "total_tokens": acct.get("total_tokens")}
    req = receipt.get("request") or {}
    return {"metering": "UNMETERED_HOST_EXECUTION", "prompt_bytes": req.get("prompt_chars"), "completion_bytes": len(receipt.get("text") or "")}
def call_branch(prompt: str, frame: str, lane: str, max_tokens: int, execute: bool, timeout: float) -> dict[str, Any]:
    system = f"ADHD divergent frame: {frame}. Isolate this branch; produce evidence, risk, and first step."
    receipt = probe_lane(lane=lane, prompt=prompt, system=system, max_tokens=max_tokens, temperature=0.35, timeout=timeout, execute=execute, log_prompts=False)
    if receipt.get("blockers"):
        raise RuntimeError(f"local_lane_failed:{lane}:{receipt['blockers']}")
    return {"frame": frame, "lane": lane, "text": receipt.get("text") or "", "receipt": receipt.get("report_path"), "telemetry": telemetry(receipt)}
def run(prompt: str, max_tokens: int, survivors: int, execute: bool, timeout: float) -> dict[str, Any]:
    if not execute:
        raise RuntimeError("execute_required")
    with cf.ThreadPoolExecutor(max_workers=len(FRAMES)) as pool:
        jobs = [pool.submit(call_branch, prompt, f, LANES[i % len(LANES)], max_tokens, execute, timeout) for i, f in enumerate(FRAMES)]
        branches = [job.result() for job in jobs]
    scored = score_branches(branches); kept = prune(scored, survivors)
    synth_prompt = json.dumps([{"frame": b["frame"], "text": b["text"]} for b in kept], sort_keys=True)
    synthesis = call_branch(synth_prompt, "synthesis_survivors_only", "mamba_cpu", max_tokens, execute, timeout)
    report = {"ok": True, "schema": "lucidota.adhd_slow_lane.v1", "source_method": "ADHD divergent frames + Treelite pruning", "router_count": ROUTER_COUNT, "branches_total": len(branches), "branches_survived": len(kept), "branches_pruned": len(branches) - len(kept), "scored": scored, "survivors": kept, "synthesis": synthesis, "canonical_graph_writes_performed": False}
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / f"adhd_slow_lane_{stamp()}.json"; report["report_path"] = str(path.relative_to(ROOT)); path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True); ap.add_argument("--max-tokens", type=int, default=64); ap.add_argument("--survivors", type=int, default=2); ap.add_argument("--timeout-sec", type=float, default=180.0); ap.add_argument("--execute", action="store_true"); ap.add_argument("--json", action="store_true")
    args = ap.parse_args(); report = run(args.prompt, args.max_tokens, args.survivors, args.execute, args.timeout_sec)
    print(json.dumps(report, sort_keys=True) if args.json else "REPORT_PATH=" + report["report_path"])
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
