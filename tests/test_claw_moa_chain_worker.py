from __future__ import annotations

import json


def test_claw_moa_chain_worker_completes_queued_nodes_with_dependency_order(tmp_path):
    from scripts.claw_moa_router import orchestrate_text
    from scripts.claw_moa_chain_worker import run_worker

    absurd_dir = tmp_path / "absurd"
    orchestrate_text(
        "audit the workflow, call Groq for bounded synthesis, then leave a Vibes code-work handoff",
        cache_key="pytest_moa_worker",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        enqueue_chain=True,
        absurd_dir=absurd_dir,
        execute_groq=False,
    )

    report = run_worker(
        absurd_dir=absurd_dir,
        max_jobs=10,
        receipt_root=tmp_path / "worker_receipts",
        execute_groq=False,
        execute_vibes=False,
    )

    assert report["status"] == "PASSED"
    assert report["jobs_completed"] == 3
    assert [job["lane"] for job in report["completed_jobs"]] == [
        "claw_moa.slow_queue_plan",
        "claw_moa.groq_synthesis",
        "claw_moa.vibes_delegate",
    ]

    state = json.loads((absurd_dir / "jobs_state.json").read_text(encoding="utf-8"))
    assert {job["state"] for job in state.values()} == {"COMPLETED"}

    by_lane = {job["lane"]: job for job in state.values()}
    assert by_lane["claw_moa.groq_synthesis"]["result"]["provider_execution_mode"] == "dry_run"
    prompt_path = by_lane["claw_moa.vibes_delegate"]["result"]["prompt_path"]
    assert prompt_path.endswith(".prompt")


def test_claw_moa_chain_worker_can_run_promptflow_visual_prototype_dry_run(tmp_path):
    import json

    from scripts.claw_moa_router import orchestrate_text
    from scripts.claw_moa_chain_worker import run_worker

    absurd_dir = tmp_path / "absurd_pf"
    orchestrate_text(
        "audit the workflow, call Groq for bounded synthesis, make a PromptFlow visual prototype, then leave a Vibes handoff",
        cache_key="pytest_moa_worker_pf",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        enqueue_chain=True,
        absurd_dir=absurd_dir,
        execute_groq=False,
        include_promptflow_prototype=True,
        promptflow_flow="04_RUNTIME/promptflow_smoke_flow",
        promptflow_data="04_RUNTIME/promptflow_smoke_flow/data.jsonl",
    )

    report = run_worker(
        absurd_dir=absurd_dir,
        max_jobs=10,
        receipt_root=tmp_path / "worker_receipts",
        execute_groq=False,
        execute_vibes=False,
        execute_promptflow_prototype=False,
    )

    assert report["status"] == "PASSED"
    assert report["jobs_completed"] == 4
    assert [job["lane"] for job in report["completed_jobs"]] == [
        "claw_moa.slow_queue_plan",
        "claw_moa.groq_synthesis",
        "claw_moa.promptflow_visual_prototype",
        "claw_moa.vibes_delegate",
    ]

    state = json.loads((absurd_dir / "jobs_state.json").read_text(encoding="utf-8"))
    pf = next(job for job in state.values() if job["lane"] == "claw_moa.promptflow_visual_prototype")
    assert pf["result"]["provider_execution_mode"] == "dry_run"
    assert pf["result"]["flow"] == "04_RUNTIME/promptflow_smoke_flow"
    assert pf["result"]["data"] == "04_RUNTIME/promptflow_smoke_flow/data.jsonl"

    vibes = next(job for job in state.values() if job["lane"] == "claw_moa.vibes_delegate")
    slow = next(job for job in state.values() if job["lane"] == "claw_moa.slow_queue_plan")
    assert vibes["depends_on"] == [slow["job_id"]]
