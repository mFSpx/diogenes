from __future__ import annotations

import struct
from pathlib import Path

from pypeline.math.endpoint_health import EndpointCircuitBreaker, EndpointPool
from pypeline.math.model_pool import (
    ModelLoadError,
    ModelPool,
    TIER_T1_QWEN_0_5B,
    TIER_T2_REASONING,
    TIER_T2_TOOL,
    TIER_T3_QWEN_7B,
)
from pypeline.math.perceptual_dedupe import (
    cluster_by_phash,
    compute_dhash,
    compute_phash,
    hamming_distance,
)


def _write_tiny_png(path: Path) -> bytes:
    import zlib

    def _chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw_scanline = b"\x00\xff\x00\x00"
    idat = _chunk(b"IDAT", zlib.compress(raw_scanline))
    iend = _chunk(b"IEND", b"")
    raw = sig + ihdr + idat + iend
    path.write_bytes(raw)
    return raw


def test_perceptual_hashes_are_available_for_png(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    _write_tiny_png(path)
    phash = compute_phash(path)
    dhash = compute_dhash(path)
    assert isinstance(phash, str) and phash
    assert isinstance(dhash, str) and dhash
    assert hamming_distance(phash, phash) == 0


def test_cluster_by_phash_groups_identical_images(tmp_path: Path) -> None:
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    raw = _write_tiny_png(a)
    b.write_bytes(raw)
    clusters = cluster_by_phash([a, b], hamming_threshold=0)
    assert any({a, b}.issubset(set(cluster)) for cluster in clusters)


def test_endpoint_circuit_breaker_opens_at_threshold() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
    breaker.record_failure("claude-pro-max", "429")
    assert breaker.is_open("claude-pro-max")


def test_endpoint_pool_falls_back_to_ollama() -> None:
    pool = EndpointPool(failure_threshold=1)
    pool.record_failure("claude-pro-max", "429")
    assert pool.pick_endpoint("claude-pro-max") == "ollama-hoebrain"


def test_math_model_pool_lru_eviction_and_mutual_exclusion() -> None:
    pool = ModelPool(ram_ceiling_mb=TIER_T2_REASONING.ram_mb + 100)
    pool.load(TIER_T2_REASONING)
    pool.load_with_eviction(TIER_T2_TOOL)
    assert pool.is_loaded(TIER_T2_TOOL.name)
    assert not pool.is_loaded(TIER_T2_REASONING.name)

    pool = ModelPool()
    pool.load(TIER_T1_QWEN_0_5B)
    pool.load(TIER_T2_REASONING)
    try:
        pool.load(TIER_T3_QWEN_7B)
    except ModelLoadError:
        pass
    else:
        raise AssertionError("expected mutual exclusion error when loading T3 with T2 resident")
