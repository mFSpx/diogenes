import psycopg


def test_live_truth_priority_stack_helper_returns_canonical_stack():
    with psycopg.connect("postgresql:///lucidota_state") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT lucidota_control.live_truth_priority_stack()")
            (stack,) = cur.fetchone()

    assert stack == [
        "live_truth_surfaces",
        "deterministic_local_checks",
        "thin_packets",
        "local",
        "indy_reads",
        "codex",
        "vibe",
        "groq",
        "broader_cloud",
    ]
