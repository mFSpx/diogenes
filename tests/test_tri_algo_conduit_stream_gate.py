from pathlib import Path

from ALGOS.tri_algo_conduit import decide_path


def test_decide_path_recovers_oversized_file_without_reading_full_payload(tmp_path: Path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * 128)
    decision = decide_path(p, observations=10, max_bytes=64, sample_bytes=8)
    assert decision.action == "recover"
    assert decision.reason == "payload_size_exceeds_max_bytes"
