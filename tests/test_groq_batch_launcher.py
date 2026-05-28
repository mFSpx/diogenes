import asyncio


def test_launcher_splits_batches_and_emits_receipts(monkeypatch):
    from scripts import groq_batch_launcher as gbl

    batches = [[{"workflow_id": "a"}], [{"workflow_id": "b"}]]
    report = gbl.plan_launches(batches, max_cloud_workers=2)
    assert report["launch_count"] == 2
    assert report["selected_workers"] == 2

    async def fake_launch(batch, **kwargs):
        return {"returncode": 0, "batch_size": len(batch), "report_path": "fake.json", "blockers": []}

    monkeypatch.setattr(gbl, "_launch_one_batch", fake_launch)
    out = asyncio.run(gbl.launch_batches(batches, execute=False, max_cloud_workers=2))
    assert out["selected_workers"] == 2
    assert len(out["launches"]) == 2
    assert all(item["returncode"] == 0 for item in out["launches"])
