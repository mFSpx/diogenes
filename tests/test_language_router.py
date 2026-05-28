from scripts import language_router


def test_language_router_maps_sloppy_build_text_to_structured_work_order():
    r = language_router.route_text("fix the code, prove it, no slop, keep it tight")
    assert r["intent"] == "coding"
    assert r["lane"]["lane"] == "SLOWLANE"
    assert r["ontology_mode"] == "GO25_STRICT"
    assert "TOOL" in r["ontology_terms"]
    assert r["model_calls_performed"] is False


def test_language_router_maps_correspondence_and_renders_template():
    r = language_router.route_text("write a tight email reply with evidence", channel="email", template="Intent={{ intent }} Terms={{ ontology_terms }}")
    assert r["intent"] == "correspondence"
    assert r["outbound"]["audience"] == "email"
    assert "Intent=correspondence" in r["rendered"]


def test_language_router_helper_stays_under_100_loc():
    from pathlib import Path
    source = Path(language_router.__file__).read_text().splitlines()
    code_lines = [line for line in source if line.strip() and not line.lstrip().startswith("#")]
    assert len(code_lines) <= 100


def test_language_router_is_wired_to_recovery_matrix():
    from pathlib import Path
    assert "scripts/language_router.py" in Path("scripts/recovery_matrix.py").read_text()
    assert "lucidota_cli.py language" in Path("scripts/recovery_matrix.py").read_text()


def test_language_router_builds_absurd_work_order_and_model_hooks():
    r = language_router.route_text(
        "build CLI language routing; prove it; wire ABSURD and hypertimeline; no model unless necessary",
        verbosity="brief",
    )
    assert r["schema"] == "lucidota.language_subsystem.route.v2"
    assert r["work_order"]["status"] == "CREATED"
    assert r["work_order"]["cep"]["canonical_mutation_allowed"] is False
    assert r["absurd"]["required_for_workflow"] is True
    assert r["hypertimeline_hook"]["script"] == "scripts/hypertimeline_engine.py"
    assert r["model_route"]["policy"] == "deterministic_first"
    assert r["model_calls_performed"] is False
    assert r["percyphon"]["zero_vram"] is True
    assert r["ontology_release"] is False
    assert r["output_contract"]["verbosity"] == "brief"


def test_language_router_releases_ontology_only_at_workflow_end():
    r = language_router.route_text("workflow end: save this prompt and print at end", channel="operator")
    assert r["ontology_mode"] == "GO25_STRICT"
    assert r["ontology_release"] is True
    assert r["outbound"]["state"] == "final_print"


def test_language_router_can_refresh_ontology_from_db_path(monkeypatch):
    monkeypatch.setattr(language_router, "db_terms", lambda database_url=None: (["ENTITY", "CLAIM", "EVIDENCE", "TOOL"], {"source": "db", "database_url": "redacted"}))
    r = language_router.route_text("evidence claim tool", refresh_ontology=True)
    assert r["ontology_refresh"]["source"] == "db"
    assert r["ontology_terms"] == ["EVIDENCE", "CLAIM", "TOOL"]


def test_operator_cli_exposes_same_language_subsystem(tmp_path):
    import json, subprocess, sys
    proc = subprocess.run(
        [sys.executable, "scripts/lucidota_cli.py", "language", "--text", "status proof for operator", "--json"],
        text=True,
        capture_output=True,
        timeout=8,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.language_subsystem.route.v2"
    assert payload["source_surfaces"] == ["cli", "operator"]
