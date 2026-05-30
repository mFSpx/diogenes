#!/usr/bin/env python3
"""Groq Patternalysis — N-lens pattern analysis over audit findings (429/403-resilient)."""
import asyncio, json, os, time, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime, timezone

GROQ_URL="https://api.groq.com/openai/v1/chat/completions"; MODEL="llama-3.3-70b-versatile"
NOISE=('BOOKS/','tests/poison_drop/','KNOWLEDGE_LIBRARY/','.claw/sessions/','/__pycache__/','07_SURFACES/')
KEY=os.environ.get("GROQ_API_KEY") or Path("/tmp/lucidota_groq_key").read_text().strip()
rows=[json.loads(l) for l in open("05_OUTPUTS/groq_scanner/findings.jsonl")]
real=[r for r in rows if not any(n in r['file'] for n in NOISE)]
digest="\n".join(f"{r['file']}: "+' '.join(r.get('finding','').split())[:140] for r in real)

LENSES=[
("Severity & volume","Real ROT/STENCH/WHIFF mix; what kind of bullshit dominates."),
("False-positive rate","How much is the audit's OWN noise/hallucination; characterize the false-positive patterns."),
("Swallowed exceptions","Map silent-failure sites (except: pass/return None); rank by risk."),
("Undefined/dead references","Real functional bugs: missing functions, broken calls, undefined names."),
("Theater by subsystem","Which directories concentrate fake-done; name the worst offenders."),
("Status-claim theater","Hardcoded OK / GO_ROUTE_OK / writes_performed:false style fake-success."),
("TODO/FIXME in load-bearing paths","Separate real incomplete critical work from noise."),
("Config/paths pointing nowhere","Dangling paths, dead endpoints, missing files referenced."),
("Duplicated/drifted logic","Forked code that has drifted out of sync."),
("Legacy/corpse code","Retired/legacy code still referenced from live paths."),
("Naming-integrity drift","Names/docstrings that contradict actual behavior."),
("Receipts/claims without effect","Proof-theater: claims of done with no real effect."),
("GONN spine integrity","Risks specifically to custody/queue/gate/graph-promotion correctness."),
("Highest-leverage fixes","The few fixes that would kill the most findings."),
("Quick wins","Cheap one-line fixes worth doing immediately."),
("Systemic root causes","The handful of root causes behind many findings."),
("What's NOT bullshit","False alarms to confidently dismiss."),
("Rust/PG/Python placement","Per the law (data->Postgres, stable->Rust, volatile->Python): what should move where."),
("Test/coverage gaps","Coverage gaps implied by the findings."),
("Executive verdict","One paragraph: real theater %, and overall honesty of the system."),
]
SYS=("You are a ruthless senior code-quality analyst reviewing bullshit-audit findings "
"(format 'file: finding') from a self-built AI system called LUCIDOTA. Analyze ONLY through the "
"requested lens. Be concrete, cite real files, no fluff, 4-7 tight bullets. If the findings "
"themselves are noise/false-positive under this lens, say so plainly.")

def call(t,p):
    payload={"model":MODEL,"temperature":0.2,"max_tokens":650,"messages":[
        {"role":"system","content":SYS},
        {"role":"user","content":f"LENS: {t}\n{p}\n\nFINDINGS ({len(real)}):\n{digest}"}]}
    data=json.dumps(payload).encode()
    for attempt in range(7):
        try:
            req=urllib.request.Request(GROQ_URL,data=data,headers={
                "Authorization":f"Bearer {KEY}","Content-Type":"application/json",
                "User-Agent":"groq-python/0.28.0","Accept":"application/json"})
            with urllib.request.urlopen(req,timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (429,403,503): time.sleep(4*(attempt+1)); continue
            return f"(lens failed: HTTP {e.code})"
        except Exception as e:
            time.sleep(3); continue
    return "(lens failed: exhausted retries)"

async def main():
    sem=asyncio.Semaphore(3)
    async def one(i,t,p):
        async with sem: return (i,t,await asyncio.to_thread(call,t,p))
    res=sorted(await asyncio.gather(*[one(i,t,p) for i,(t,p) in enumerate(LENSES)]))
    out=Path("05_OUTPUTS/groq_scanner/patternalysis.md"); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Groq Patternalysis — {len(LENSES)} lenses / {len(real)} real findings ({len(rows)} raw)\n{datetime.now(timezone.utc).isoformat()}\n\n")
        for i,t,c in res: f.write(f"## {i+1}. {t}\n{c}\n\n")
    ok=sum(1 for _,_,c in res if not c.startswith("(lens failed"))
    print(f"WROTE {out} | lenses_ok={ok}/{len(LENSES)} real={len(real)} raw={len(rows)}\n")
    for i,t,c in res: print(f"\n===== {i+1}. {t} =====\n{c}")
asyncio.run(main())
