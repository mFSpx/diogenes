from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REAL_WORK_LOOP_SCHEMA = "039_absurd_real_work_loop.sql"


def test_absurd_workers_bootstrap_real_work_loop_schema_before_runtime_columns():
    workers = [
        "scripts/absurd_queue_spine.py",
        "scripts/absurd_chrono_worker.py",
        "scripts/absurd_river_worker.py",
        "scripts/absurd_graph_promotion_worker.py",
        "scripts/absurd_intake_worker.py",
    ]
    missing = []
    for worker in workers:
        text = (ROOT / worker).read_text(encoding="utf-8")
        if REAL_WORK_LOOP_SCHEMA not in text:
            missing.append(worker)
    assert missing == []

