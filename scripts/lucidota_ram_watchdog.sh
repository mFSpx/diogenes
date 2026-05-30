#!/usr/bin/env bash
# RAM WATCHDOG — keeps the box alive while the operator is away.
# If MemAvailable drops below FLOOR, SIGSTOP the crush(es) (FREEZE, never kill ->
# zero data loss); SIGCONT when memory recovers above RESUME. Heartbeat every tick.
# This is the GNU-parallel --memsuspend pattern and a Phase-0 sliver of the guvna:
# scale DOWN by suspending, never by culling. Bounded run, then exits.
set -uo pipefail
ROOT="${LUCIDOTA_HOME:-/home/mfspx/LUCIDOTA}"
LOG="$ROOT/04_RUNTIME/krampus_watchdog.log"
PROG="$ROOT/04_RUNTIME/krampus_archive_progress.json"
PAT="${WATCHDOG_PATTERN:-krampus_archive_crush}"
FLOOR="${WATCHDOG_FLOOR_MB:-700}"
RESUME="${WATCHDOG_RESUME_MB:-1200}"
TICKS="${WATCHDOG_TICKS:-150}"
INT="${WATCHDOG_INTERVAL:-20}"
paused=0
echo "$(date -Iseconds) WATCHDOG START floor=${FLOOR}M resume=${RESUME}M pat=$PAT ticks=$TICKS" >> "$LOG"
for ((i=0; i<TICKS; i++)); do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  prog=$(python3 -c "import json;d=json.load(open('$PROG'));print('chunks='+str(d.get('chunks',0)),'members='+str(d.get('members',0)),'slow='+str(d.get('slow',0)),'errors='+str(d.get('errors',0)))" 2>/dev/null || echo "no-progress")
  if [ "$avail" -lt "$FLOOR" ] && [ "$paused" -eq 0 ]; then
    pkill -STOP -f "$PAT" 2>/dev/null && paused=1
    echo "$(date -Iseconds) !! SUSPEND avail=${avail}M (<${FLOOR}) $prog" >> "$LOG"
  elif [ "$avail" -gt "$RESUME" ] && [ "$paused" -eq 1 ]; then
    pkill -CONT -f "$PAT" 2>/dev/null && paused=0
    echo "$(date -Iseconds) >> RESUME avail=${avail}M (>${RESUME}) $prog" >> "$LOG"
  fi
  echo "$(date -Iseconds) avail=${avail}M paused=$paused $prog" >> "$LOG"
  sleep "$INT"
done
# leave nothing suspended on exit
pkill -CONT -f "$PAT" 2>/dev/null || true
echo "$(date -Iseconds) WATCHDOG EXIT" >> "$LOG"
