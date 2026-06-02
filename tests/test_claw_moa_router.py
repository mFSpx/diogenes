from __future__ import annotations


def test_claw_moa_router_dry_run_preserves_route_and_output_lanes(tmp_path):
    from scripts.claw_moa_router import orchestrate_text

    payload = orchestrate_text(
        "status check: keep this fast and receipt-backed",
        cache_key="pytest_moa",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
    )

    assert payload["schema"] == "lucidota.claw_moa_router.v1"
    assert payload["input_route"]["lane"] == "FASTLANE"
    assert payload["model_calls_performed"] is False
    assert payload["canonical_graph_writes_performed"] is False
    assert payload["hyperplex"]["outbound_state"] == "draft_only"
    assert payload["task_chain"]["schema"] == "lucidota.claw_moa_router.task_chain.v1"

    lane_names = [lane["lane"] for lane in payload["hyperplex"]["lanes"]]
    assert "tera_template" in lane_names
    assert "rag_quotes" in lane_names
    assert "deepseek_q4" in lane_names


def test_claw_moa_router_deep_work_routes_to_slowlane(tmp_path):
    from scripts.claw_moa_router import orchestrate_text

    payload = orchestrate_text(
        "please audit and refactor the model workflow with proof",
        cache_key="pytest_moa_slow",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
    )

    assert payload["input_route"]["lane"] == "SLOWLANE"
    assert payload["lane_plan"]["external_lanes"] == ["groq", "vibes"]
    assert payload["lane_plan"]["execution_state"] == "planned_not_executed"


def test_claw_moa_router_exports_slowlane_task_chain_jsonl(tmp_path):
    from scripts.claw_moa_router import orchestrate_text

    chain_path = tmp_path / "chains" / "slow_chain.jsonl"
    payload = orchestrate_text(
        "audit the workflow, call Groq for bounded synthesis, then leave a Vibes code-work handoff",
        cache_key="pytest_moa_chain",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        chain_jsonl_out=chain_path,
        execute_groq=False,
    )

    chain = payload["task_chain"]
    node_ids = [node["node_id"] for node in chain["nodes"]]
    assert node_ids[:3] == ["ingest_packet", "deterministic_route", "cache_packet"]
    assert "groq_synthesis" in node_ids
    assert "vibes_delegate" in node_ids
    assert "hyperplex_output" in node_ids
    assert chain["chain_jsonl_path"].endswith("slow_chain.jsonl")

    rows = [line for line in chain_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(chain["nodes"])
    groq_node = next(node for node in chain["nodes"] if node["node_id"] == "groq_synthesis")
    assert groq_node["depends_on"] == ["deterministic_route"]
    assert groq_node["state"] == "planned"


def test_claw_moa_router_enqueues_planned_chain_nodes_to_absurd_adapter(tmp_path):
    import json

    from scripts.claw_moa_router import orchestrate_text

    absurd_dir = tmp_path / "absurd"
    payload = orchestrate_text(
        "audit the workflow, call Groq for bounded synthesis, then leave a Vibes code-work handoff",
        cache_key="pytest_moa_enqueue",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        enqueue_chain=True,
        absurd_dir=absurd_dir,
        execute_groq=False,
    )

    enqueue = payload["task_chain"]["enqueue"]
    assert enqueue["status"] == "PASSED"
    assert enqueue["job_count"] == 3
    assert enqueue["absurd_state_path"].endswith("jobs_state.json")

    state = json.loads((absurd_dir / "jobs_state.json").read_text(encoding="utf-8"))
    lanes = {job["lane"] for job in state.values()}
    assert lanes == {
        "claw_moa.slow_queue_plan",
        "claw_moa.groq_synthesis",
        "claw_moa.vibes_delegate",
    }
    assert {job["state"] for job in state.values()} == {"QUEUED"}

    by_lane = {job["lane"]: job for job in state.values()}
    assert by_lane["claw_moa.vibes_delegate"]["depends_on"] == [
        by_lane["claw_moa.slow_queue_plan"]["job_id"]
    ]


def test_claw_moa_router_separates_strict_local_admission_from_external_provider_keys(tmp_path, monkeypatch):
    from scripts.claw_moa_router import orchestrate_text

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    payload = orchestrate_text(
        "audit routing fabric and call Groq if available",
        cache_key="pytest_provider_admission",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        execute_groq=True,
        enqueue_chain=False,
    )

    lane_plan = payload["lane_plan"]
    assert lane_plan["local_model_admission"]["mode"] == "strict_fail_closed"
    assert lane_plan["local_model_admission"]["startup_blocked_by_missing_provider_keys"] is False
    strict_services = {service["name"]: service for service in lane_plan["local_model_admission"]["services"]}
    assert strict_services["deepseek_r1_qwen_1p5b_gpu"]["switch_group"] == "reasoning_generation_slot"
    assert strict_services["bonsai4b_ram"]["switch_role"] == "ram_resident_gpu_switchable"
    assert lane_plan["provider_lanes"]["groq"]["lane_kind"] == "external_provider"
    assert lane_plan["provider_lanes"]["groq"]["status"] == "skipped"
    assert lane_plan["provider_lanes"]["groq"]["reason"] == "missing_GROQ_API_KEY"
    assert payload["model_synthesis"]["status"] == "skipped"
    assert payload["model_calls_performed"] is False
    assert "groq_synthesis_failed" not in payload["blockers"]


def test_claw_moa_router_route_targets_can_route_to_router_workflow_and_provider(tmp_path):
    from scripts.claw_moa_router import orchestrate_text

    payload = orchestrate_text(
        "audit ontology workflow and delegate slow work through nested routers",
        cache_key="pytest_router_targets",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        execute_groq=False,
        enqueue_chain=False,
    )

    targets = payload["route_targets"]
    target_kinds = {target["target_kind"] for target in targets}
    assert {"router", "workflow", "provider_lane"}.issubset(target_kinds)
    assert any(t["target_id"] == "language_router" for t in targets)
    assert any(t["target_id"] == "absurd_workflow_queue" for t in targets)
    assert all("decision_packet_id" in t for t in targets)


def test_claw_moa_router_enqueues_slowlane_chain_to_absurd_db_when_database_url_given(tmp_path):
    from scripts.claw_moa_router import orchestrate_text

    payload = orchestrate_text(
        "audit the workflow, call Groq for bounded synthesis, then leave a Vibes code-work handoff",
        cache_key="pytest_moa_db_enqueue",
        cache_dir=tmp_path / "lane_cache",
        receipt_root=tmp_path / "receipts",
        no_receipt=True,
        enqueue_chain=True,
        absurd_dir=tmp_path / "absurd",
        execute_groq=False,
        database_url="postgresql:///lucidota_state",
    )

    enqueue = payload["task_chain"]["enqueue"]
    assert enqueue["status"] == "PASSED"
    assert enqueue["db_writes_performed"] is True
    assert enqueue["db_enqueue"]["queue_name"] == "luci_operator"
    assert len(enqueue["db_enqueue"]["jobs"]) == enqueue["job_count"]
    assert all(job["job_uuid"] for job in enqueue["db_enqueue"]["jobs"])
