import json


def test_compiler_groups_workflows_into_batches(tmp_path):
    from scripts import groq_workorder_compiler as gwc

    src = tmp_path / "workflows.json"
    src.write_text(json.dumps({"workflows": [{"workflow_id": "a"}, {"workflow_id": "b"}, {"workflow_id": "c"}]}))
    report = gwc.compile_workorders(src, batch_size=2)
    assert len(report["batches"]) == 2
    assert report["batches"][0][0]["workflow_id"] == "a"
    assert report["workflow_count"] == 3
