#!/usr/bin/env python3
"""Execute queued Claw MOA chain nodes from the existing ABSURD adapter."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from spine_common import now, receipt, rel  # noqa: E402
from spine_job_adapter import ABSURDJobAdapter  # noqa: E402

DEFAULT_ABSURD_DIR = ROOT / "09_STORAGE" / "absurd" / "claw_moa"
DEFAULT_RECEIPT_ROOT = ROOT / "05_OUTPUTS" / "claw_moa_worker"
RUNTIME_DIR = ROOT / "04_RUNTIME" / "claw_moa"
LANE_PRIORITY = {
    "claw_moa.slow_queue_plan": 10,
    "claw_moa.groq_synthesis": 20,
    "claw_moa.promptflow_visual_prototype": 25,
    "claw_moa.vibes_delegate": 30,
}


def _node(job: dict[str, Any]) -> dict[str, Any]:
    return ((job.get("payload") or {}).get("node") or {})


def _prompt_for_job(job: dict[str, Any]) -> str:
    node = _node(job)
    return (
        "Execute this bounded Claw MOA task-chain node. Keep it terse and receipt-backed.\n\n"
        + json.dumps(
            {
                "job_id": job.get("job_id"),
                "lane": job.get("lane"),
                "node": node,
                "depends_on": job.get("depends_on", []),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _run_groq(job: dict[str, Any], *, execute_groq: bool, max_tokens: int, timeout_sec: float) -> dict[str, Any]:
    prompt = _prompt_for_job(job)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "groq_chat_cli.py"),
        "--prompt",
        prompt,
        "--system",
        "Return concise plain text only.",
        "--model",
        "llama-3.1-8b-instant",
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        "0",
    ]
    if execute_groq:
        cmd.append("--execute")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec)
    receipt_path = None
    status = None
    for line in proc.stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            receipt_path = line.split("=", 1)[1]
        if line.startswith("GROQ_CHAT="):
            status = line.split("=", 1)[1]
    return {
        "node_id": _node(job).get("node_id"),
        "provider": "groq",
        "provider_execution_mode": "execute" if execute_groq else "dry_run",
        "subreceipt_path": receipt_path,
        "provider_status": status,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def _write_vibes_prompt(job: dict[str, Any]) -> dict[str, Any]:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNTIME_DIR / f"vibes_delegate_{job['job_id'].replace(':', '_')}.prompt"
    prompt = (
        "You are the Mistral/Vibes code-work sidepiece for a bounded Claw MOA chain node.\n"
        "Use the operator note as Mistral/Vibes-only budget context: 200k session, ~32k active window.\n"
        "Do not mutate files unless the caller explicitly executes this prompt with file ownership.\n\n"
        + _prompt_for_job(job)
    )
    path.write_text(prompt, encoding="utf-8")
    return {
        "node_id": _node(job).get("node_id"),
        "provider": "mistral_vibes",
        "provider_execution_mode": "prompt_written",
        "prompt_path": rel(path),
        "execute_hint": f".venv/bin/vibe -p @{rel(path)} --agent auto-approve --trust --workdir {ROOT}",
    }


def _run_promptflow_prototype(job: dict[str, Any], *, execute_promptflow_prototype: bool, timeout_sec: float) -> dict[str, Any]:
    node = _node(job)
    flow = str(node.get("flow") or "04_RUNTIME/promptflow_smoke_flow")
    data = node.get("data") or "04_RUNTIME/promptflow_smoke_flow/data.jsonl"
    run_id = str(node.get("run_id") or f"claw_moa_pf_{job['job_id'].replace(':', '_')}")
    if not execute_promptflow_prototype:
        return {
            "node_id": node.get("node_id"),
            "provider": "promptflow",
            "provider_execution_mode": "dry_run",
            "role": "visual_prototype_only_not_correctness_gate",
            "flow": flow,
            "data": data,
            "run_id": run_id,
            "command": f"./claw flow run {flow} --batch-data {data} --run-id {run_id}",
        }
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "promptflow_eval_runner.py"),
        "--flow",
        flow,
        "--run-id",
        run_id,
    ]
    if data:
        cmd += ["--data", str(data)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec + 30)
    receipt_path = None
    for line in proc.stdout.splitlines():
        if "receipt=" in line:
            receipt_path = line.rsplit("receipt=", 1)[-1].strip()
    return {
        "node_id": node.get("node_id"),
        "provider": "promptflow",
        "provider_execution_mode": "execute",
        "role": "visual_prototype_only_not_correctness_gate",
        "flow": flow,
        "data": data,
        "run_id": run_id,
        "subreceipt_path": receipt_path,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }


def execute_job(
    job: dict[str, Any],
    *,
    execute_groq: bool,
    execute_vibes: bool,
    execute_promptflow_prototype: bool,
    max_tokens: int,
    timeout_sec: float,
) -> dict[str, Any]:
    lane = job.get("lane")
    if lane == "claw_moa.slow_queue_plan":
        return {
            "node_id": _node(job).get("node_id"),
            "provider": "deterministic",
            "provider_execution_mode": "execute",
            "plan": "slow lane job gate prepared; downstream provider lanes remain bounded sidepieces",
        }
    if lane == "claw_moa.groq_synthesis":
        return _run_groq(job, execute_groq=execute_groq, max_tokens=max_tokens, timeout_sec=timeout_sec)
    if lane == "claw_moa.promptflow_visual_prototype":
        return _run_promptflow_prototype(
            job,
            execute_promptflow_prototype=execute_promptflow_prototype,
            timeout_sec=timeout_sec,
        )
    if lane == "claw_moa.vibes_delegate":
        result = _write_vibes_prompt(job)
        result["execute_requested"] = bool(execute_vibes)
        # The default and tested path writes the prompt/handoff only. Direct Vibes
        # execution stays explicit because it may mutate code and spend model budget.
        return result
    return {
        "node_id": _node(job).get("node_id"),
        "provider": "unknown",
        "provider_execution_mode": "skipped",
        "note": f"no worker implementation for lane {lane}",
    }


def run_worker(
    *,
    absurd_dir: str | Path = DEFAULT_ABSURD_DIR,
    max_jobs: int = 1,
    receipt_root: str | Path = DEFAULT_RECEIPT_ROOT,
    execute_groq: bool = False,
    execute_vibes: bool = False,
    execute_promptflow_prototype: bool = False,
    groq_max_tokens: int = 160,
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    adapter = ABSURDJobAdapter(absurd_dir)
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _ in range(max_jobs):
        ready = sorted(
            [job for job in adapter.ready_jobs() if str(job.get("lane", "")).startswith("claw_moa.")],
            key=lambda job: (LANE_PRIORITY.get(str(job.get("lane")), 999), str(job.get("created_at", "")), str(job.get("job_id", ""))),
        )
        if not ready:
            break
        job = ready[0]
        try:
            if job["state"] == "CREATED":
                job = adapter.transition(job["job_id"], "QUEUED")
            job = adapter.transition(job["job_id"], "RUNNING")
            result = execute_job(
                job,
                execute_groq=execute_groq,
                execute_vibes=execute_vibes,
                execute_promptflow_prototype=execute_promptflow_prototype,
                max_tokens=groq_max_tokens,
                timeout_sec=timeout_sec,
            )
            if result.get("returncode", 0) != 0:
                raise RuntimeError(f"provider call failed for {job['lane']}: {result.get('returncode')}")
            done = adapter.transition(job["job_id"], "COMPLETED", result=result)
            completed.append(
                {
                    "job_id": done["job_id"],
                    "lane": done["lane"],
                    "node_id": result.get("node_id"),
                    "result": result,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            failures.append({"job_id": job.get("job_id"), "lane": job.get("lane"), "error": f"{type(exc).__name__}: {exc}"})
            try:
                adapter.transition(job["job_id"], "FAILED", error=f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            break
    report = {
        "schema": "lucidota.claw_moa_chain_worker.v1",
        "generated_at": now(),
        "status": "PASSED" if not failures else "FAILED",
        "absurd_state_path": rel(adapter.state_path),
        "jobs_completed": len(completed),
        "completed_jobs": completed,
        "failures": failures,
        "execute_groq": bool(execute_groq),
        "execute_vibes": bool(execute_vibes),
        "execute_promptflow_prototype": bool(execute_promptflow_prototype),
        "canonical_graph_writes_performed": False,
    }
    receipt("claw_moa_chain_worker", report, root=Path(receipt_root))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute queued Claw MOA chain jobs from an ABSURD adapter dir.")
    parser.add_argument("--absurd-dir", default=str(DEFAULT_ABSURD_DIR))
    parser.add_argument("--max-jobs", type=int, default=1)
    parser.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    parser.add_argument("--execute-groq", action="store_true")
    parser.add_argument("--execute-vibes", action="store_true")
    parser.add_argument("--execute-promptflow-prototype", action="store_true", help="Run PromptFlow as a visual prototype lane; never a hard gate.")
    parser.add_argument("--groq-max-tokens", type=int, default=160)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_worker(
        absurd_dir=args.absurd_dir,
        max_jobs=args.max_jobs,
        receipt_root=args.receipt_root,
        execute_groq=args.execute_groq,
        execute_vibes=args.execute_vibes,
        execute_promptflow_prototype=args.execute_promptflow_prototype,
        groq_max_tokens=args.groq_max_tokens,
        timeout_sec=args.timeout_sec,
    )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    print("CLAW_MOA_WORKER=" + report["status"])
    print("JOBS_COMPLETED=" + str(report["jobs_completed"]))
    return 0 if report["status"] == "PASSED" else 4


if __name__ == "__main__":
    raise SystemExit(main())
