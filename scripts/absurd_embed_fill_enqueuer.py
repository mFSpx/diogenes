import os, json
import psycopg2

STATE_DSN = os.getenv('LUCIDOTA_GO_STATE_DSN', 'postgresql:///lucidota_state')
STORAGE_DSN = os.getenv('LUCIDOTA_GO_STORAGE_DSN', 'postgresql:///lucidota_storage')
BATCH_SIZE = 50000

sc = psycopg2.connect(STATE_DSN)
stc = psycopg2.connect(STORAGE_DSN)

with sc, sc.cursor() as cur:
    cur.execute("""
        INSERT INTO lucidota_control.absurd_worker_contract
          (worker_key, queue_name, script_path, input_contract, output_contract,
           idempotency_rule, retry_policy, dead_letter_policy,
           canonical_graph_write_allowed, status, evidence_refs)
        VALUES ('embed_fill_batch','korpus','scripts/corpus_embed_fill.py',
          '{"job_kind":"embed_fill_batch","required_fields":["offset","limit"]}',
          '{"receipt_glob":"05_OUTPUTS/runtime/embed_fill*.json"}',
          'offset_limit','{"max_attempts":3,"backoff":"exponential"}',
          '{"policy":"quarantine"}', FALSE, 'implemented', '[]')
        ON CONFLICT (worker_key) DO NOTHING
    """)

with stc.cursor() as cur:
    cur.execute("SELECT count(*) FROM lucidota_korpus.corpus_chunk WHERE embedding IS NULL")
    total_null = cur.fetchone()[0]
stc.close()

jobs_enqueued = 0
with sc, sc.cursor() as cur:
    for offset in range(0, total_null, BATCH_SIZE):
        ikey = f"embed_fill_batch:offset:{offset}"
        payload = json.dumps({"offset": offset, "limit": BATCH_SIZE, "job_kind": "embed_fill_batch"})
        cur.execute("""
            INSERT INTO lucidota_control.absurd_queue_job
              (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts)
            VALUES ('korpus','embed-fill-pipeline','embed_fill_batch',%s,%s::jsonb,50,3)
            ON CONFLICT (queue_name, idempotency_key) DO NOTHING
        """, (ikey, payload))
        if cur.rowcount > 0:
            jobs_enqueued += 1

sc.close()
print(f"total_null={total_null} jobs_enqueued={jobs_enqueued} batch_size={BATCH_SIZE}")
