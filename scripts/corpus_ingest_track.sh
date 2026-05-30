#!/usr/bin/env bash
# One-shot CORPUS INGEST tracking snapshot.
# Live watch:  watch -n 5 scripts/corpus_ingest_track.sh
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
P="$ROOT/04_RUNTIME/corpus_extract_progress.json"
echo "── CORPUS INGEST @ $(date '+%H:%M:%S') ──"
if [[ -f "$P" ]]; then
  python3 - "$P" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    done=d.get('files_done',0); tot=d.get('total',0) or 0
    pct=(100.0*done/tot) if tot else 0.0
    print(f"  files {done}/{tot} ({pct:.1f}%)  chunks={d.get('chunks',0)}  embeds={d.get('embeds',0)}  deferred={d.get('deferred',0)}  errors={d.get('errors',0)}")
except Exception as e:
    print('  progress.json:',e)
PY
else
  echo "  (no progress file yet)"
fi
cnt=$(psql postgresql:///lucidota_storage -Atc "SELECT count(*) FROM lucidota_korpus.corpus_chunk;" 2>/dev/null)
echo "  DB corpus_chunk rows = ${cnt:-?}"
pgrep -fa corpus_groq_extractor >/dev/null 2>&1 && echo "  extractor: RUNNING" || echo "  extractor: not running"
up=0; for p in 8101 8102 8103 8104 8105 8106; do curl -fsS --max-time 1 "http://127.0.0.1:$p/health" >/dev/null 2>&1 && up=$((up+1)); done
echo "  BGE fleet up: $up/6"
free -m | awk 'NR==2{print "  RAM avail="$7"M  used="$3"M"}'
